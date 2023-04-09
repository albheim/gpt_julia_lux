using Lux, OneHotArrays, Optimisers, Zygote, NNlib
using Plots
using Random, Statistics, Distributions, LinearAlgebra

rng = Random.default_rng()
Random.seed!(rng, 37)

download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")

text = read("input.txt", String)

chars = sort(collect(Set(text)))
vocab_size = length(chars)
println("$vocab_size chars: $(join(chars))")

stoi = Dict(ch => i for (i,ch) in enumerate(chars))
itos = Dict(i => ch for (i,ch) in enumerate(chars))
encode(s) = [stoi[c] for c in s]
decode(l) = join(itos[i] for i in l)

data = encode(text)

n = round(Int, 0.9*length(data))
train_data = data[1:n]
val_data = data[n+1:end]

eval_iters = 200
eval_interval = 500
max_iters = 5000
batch_size = 32
block_size = 8
n_embd = 32
lr = 1e-3

function get_batch(rng, data; block_size, batch_size, vocab_size)
    ix = rand(rng, 1:length(data)-block_size, batch_size)
    x = stack(@view data[i:i+block_size-1] for i in ix)
    y = onehotbatch(stack(@view data[i+1:i+block_size] for i in ix), 1:vocab_size)
    x, y
end

xb, yb = get_batch(rng, train_data; batch_size, block_size, vocab_size)


struct Head{K,Q,V,M} <: Lux.AbstractExplicitContainerLayer{(:key_net, :query_net, :value_net)}
    key_net::K
    query_net::Q
    value_net::V
    triangular_matrix::M
end
    
function Head(n_embedding, head_size, block_size)
    mat = zeros(block_size, block_size)
    for i in axes(mat, 2)
        for j in i+1:size(mat, 1)
            mat[j, i] = -Inf
        end
    end

    Head(
        Dense(n_embedding => head_size),
        Dense(n_embedding => head_size),
        Dense(n_embedding => head_size),
        mat
    )
end

function (h::Head)(x::AbstractArray, ps, st::NamedTuple)
    yk, sk = h.key_net(x, ps.key_net, st.key_net)
    yq, sq = h.query_net(x, ps.query_net, st.query_net)
    yv, sv = h.value_net(x, ps.value_net, st.value_net)

    wei = batched_mul(permutedims(yk, (2, 1, 3)), yq) .* (size(x, 1)^-0.5)

    wei = wei .+ h.triangular_matrix
    wei = softmax(wei, dims=1)

    out = batched_mul(yv, wei)

    return out, (key_net = sk, query_net = sq, value_net=sv)
end

model = Head(n_embd, n_embd, block_size)
ps, st = Lux.setup(rng, model)

y, st = model(randn(rng, n_embd, block_size, batch_size), ps, st)
size(y)

struct MultiHeadAttention{H,D,L} <: Lux.AbstractExplicitLayer
    heads::Vector{H}
    projection::D
    layer_norm::L
end

function MultiHeadAttention(num_heads; n_embedding, head_size, block_size)
    MultiHeadAttention(
        [Head(n_embedding, head_size, block_size) for _ in 1:num_heads],
        Dense(n_embedding => n_embedding),
        LayerNorm((n_embedding, block_size), dims=1),
    )
end

Lux.initialparameters(rng::AbstractRNG, m::MultiHeadAttention) = (; 
    multiheadparams=[Lux.initialparameters(rng, h) for h in m.heads], 
    projection=Lux.initialparameters(rng, m.projection),
    layer_norm=Lux.initialparameters(rng, m.layer_norm)
)
Lux.initialstates(rng::AbstractRNG, m::MultiHeadAttention) = (; 
    multiheadstates=[Lux.initialstates(rng, h) for h in m.heads],
    projection=Lux.initialstates(rng, m.projection),
    layer_norm=Lux.initialstates(rng, m.layer_norm)
)
Lux.parameterlength(m::MultiHeadAttention) = length(m.heads) * Lux.parameterlength(m.heads[1]) + Lux.parameterlength(m.projection) + Lux.parameterlength(m.layer_norm)
Lux.statelength(m::MultiHeadAttention) = length(m.heads) * Lux.statelength(m.heads[1]) + Lux.statelength(m.projection) + Lux.statelength(m.layer_norm)

function (m::MultiHeadAttention)(x::AbstractArray, ps, st::NamedTuple)
    in, st_ln = m.layer_norm(x, ps.layer_norm, st.layer_norm)
    tmp = [m.heads[i](in, ps.multiheadparams[i], st.multiheadstates[i]) for i in eachindex(m.heads)]
    out = reduce(vcat, map(x -> x[1], tmp))
    out, st_pr = m.projection(out, ps.projection, st.projection)
    newstates = collect(map(x -> x[2], tmp))

    return out, (; multiheadstates=newstates, projection=st_pr, layer_norm=st_ln)
end

struct Block{H,F} <: Lux.AbstractExplicitContainerLayer{(:sa_head, :feed_forward)}
    sa_head::H
    feed_forward::F
end

function Block(n_embedding, n_head, block_size)
    head_size = n_embedding ÷ n_head
    Block(
        MultiHeadAttention(n_head; n_embedding, head_size, block_size),
        Chain(
            LayerNorm((n_embedding, block_size), dims=1),
            Dense(n_embedding => 4 * n_embedding, relu),
            Dense(4 * n_embedding => n_embedding),
        ),
    )
end

function (bl::Block)(x::AbstractArray, ps, st::NamedTuple)
    y, st_sa = bl.sa_head(x, ps.sa_head, st.sa_head)
    x += y
    y, st_ff = bl.feed_forward(x, ps.feed_forward, st.feed_forward)
    x += y
    return x, (sa_head=st_sa, feed_forward=st_ff)
end

struct BigramLanguageModel{E1,E2,B,H} <: Lux.AbstractExplicitContainerLayer{(:token_embedding, :position_embedding, :blocks, :lm_head)}
    token_embedding::E1
    position_embedding::E2
    blocks::B
    lm_head::H
end

BigramLanguageModel(n_embedding, vocabulary_size, block_size; n_head=4) = BigramLanguageModel(
    Embedding(vocabulary_size => n_embedding),
    Embedding(block_size => n_embedding),
    Chain(
        Block(n_embedding, n_head, block_size),
        Block(n_embedding, n_head, block_size),
        Block(n_embedding, n_head, block_size),
        LayerNorm((n_embedding, block_size), dims=1),
    ),
    Dense(n_embedding, vocabulary_size),
)

function (bl::BigramLanguageModel)(x::AbstractArray, ps, st::NamedTuple)
    tok_emb, st_tok = bl.token_embedding(x, ps.token_embedding, st.token_embedding)
    pos_emb, st_pos = bl.position_embedding(1:size(x, 1), ps.position_embedding, st.position_embedding)

    x = tok_emb .+ pos_emb
    x, st_bk = bl.blocks(x, ps.blocks, st.blocks)
    logits, st_lm = bl.lm_head(x, ps.lm_head, st.lm_head)

    return logits, (token_embedding=st_tok, position_embedding=st_pos, blocks=st_bk, lm_head=st_lm)
end

model = BigramLanguageModel(n_embd, vocab_size, block_size)
ps, st = Lux.setup(rng, model)

model(rand(rng, 1:vocab_size, block_size, batch_size), ps, st)

function logitcrossentropy(ypred, ytrue; dims=1)
    return mean(.-sum(ytrue .* Lux.logsoftmax(ypred; dims=dims); dims=dims))
end

function compute_loss(x, y, model, ps, st)
    ypred, st = model(x, ps, st)
    return logitcrossentropy(ypred, y), ypred, st
end

compute_loss(xb, yb, model, ps, st)[1]

function generate(rng, x, model, ps, st; new_tokens, block_size)
    for _ in 1:new_tokens
        crop_len = min(block_size, size(x, 1))
        x_recent = @view x[end+1-crop_len:end, :]
        logits, st = model(x_recent, ps, st)
        probs = Lux.softmax(logits[:, end, :]; dims=1)
        xn = [rand(rng, Categorical(probs[:, batch])) for batch in axes(probs, 2)]
        x = [x ; reshape(xn, 1, size(xn)...)]
    end
    x
end

generated = generate(rng, ones(Int, block_size, 1), model, ps, st; new_tokens=100, block_size)[:, 1]
decoded = decode(generated)

opt = Optimisers.ADAMW(lr)
opt_state = Optimisers.setup(opt, ps)

(loss, ypred, st), back = pullback(p -> compute_loss(xb, yb, model, p, st), ps)
gs = back((one(loss), nothing, nothing))[1]

function estimate_loss(rng, datas, model, ps, st; eval_iters, batch_size, block_size)
    out = []
    for data in datas
        loss = 0.0
        for k in 1:eval_iters
            x, y = get_batch(rng, data; block_size, batch_size, vocab_size)
            loss += compute_loss(x, y, model, ps, st)[1]
        end
        push!(out, loss / eval_iters)
    end
    out
end

for epoch in 1:max_iters
    x, y = get_batch(rng, data; block_size, batch_size, vocab_size)
    (loss, ypred, st), back = pullback(p -> compute_loss(x, y, model, p, st), ps)
    gs = back((one(loss), nothing, nothing))[1]
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
    if epoch % eval_interval == 0
        loss = estimate_loss(rng, (train_data, val_data), model, ps, st; eval_iters, block_size, batch_size=32)
        println("Epoch [$epoch]: Training loss $(loss[1]), validation loss $(loss[2])")
    end
end


decode(generate(rng, ones(Int, block_size, 1), model, ps, st; new_tokens=500, block_size)[:, 1])

