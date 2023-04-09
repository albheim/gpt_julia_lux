using Lux, OneHotArrays, Optimisers, Zygote
using Plots
using Random, Statistics, Distributions

rng = Random.default_rng()
Random.seed!(rng, 37)

download("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", "input.txt")

text = read("input.txt", String)

println(text[1:1000])

chars = sort(collect(Set(text)))
vocab_size = length(chars)
println("$vocab_size chars: $(join(chars))")

stoi = Dict(ch => i for (i,ch) in enumerate(chars))
itos = Dict(i => ch for (i,ch) in enumerate(chars))
encode(s) = [stoi[c] for c in s]
decode(l) = join(itos[i] for i in l)

encode("hii there")
decode(encode("hii there"))

data = encode(text)

n = round(Int, 0.9*length(data))
train_data = data[1:n]
val_data = data[n+1:end]

batch_size = 4
block_size = 8

function get_batch(rng, data; block_size, batch_size)
    ix = rand(rng, 1:length(data)-block_size, batch_size)
    x = stack(@view data[i:i+block_size-1] for i in ix)
    y = onehotbatch(stack(@view data[i+1:i+block_size] for i in ix), 1:vocab_size)
    x, y
end

xb, yb = get_batch(rng, train_data; batch_size, block_size)

model = Lux.Embedding(vocab_size => vocab_size)

ps, st = Lux.setup(rng, model)

yhat, st = model(xb, ps, st)

function logitcrossentropy(ypred, ytrue; dims=1)
    return mean(.-sum(ytrue .* Lux.logsoftmax(ypred; dims=dims); dims=dims))
end

function compute_loss(x, y, model, ps, st)
    ypred, st = model(x, ps, st)
    return logitcrossentropy(ypred, y), ypred, st
end

compute_loss(xb, yb, model, ps, st)[1]

function generate(rng, x, new_tokens, model, ps, st)
    for _ in 1:new_tokens
        logits, st = model(x, ps, st)
        probs = Lux.softmax(logits[:, end, :]; dims=1)
        xn = [rand(rng, Categorical(probs[:, batch])) for batch in axes(probs, 2)]
        x = [x ; reshape(xn, 1, size(xn)...)]
    end
    x
end

generated = generate(rng, ones(Int, 1, 1), 100, model, ps, st)[:, 1]
decoded = decode(generated)

opt = Optimisers.ADAMW(1e-3)
opt_state = Optimisers.setup(opt, ps)

(loss, ypred, st), back = pullback(p -> compute_loss(xb, yb, model, p, st), ps)
gs = back((one(loss), nothing, nothing))[1]

function estimate_loss(rng, datas, model, ps, st; eval_iters, batch_size, block_size)
    out = []
    for data in datas
        loss = 0.0
        for k in 1:eval_iters
            x, y = get_batch(rng, data; block_size, batch_size)
            loss += compute_loss(x, y, model, ps, st)[1]
        end
        push!(out, loss / eval_iters)
    end
    out
end

eval_iters = 200
eval_interval = 300

for epoch in 1:10000
    x, y = get_batch(rng, data; block_size, batch_size=32)
    (loss, ypred, st), back = pullback(p -> compute_loss(x, y, model, p, st), ps)
    gs = back((one(loss), nothing, nothing))[1]
    opt_state, ps = Optimisers.update(opt_state, ps, gs)
    if epoch % eval_interval == 0
        loss = estimate_loss(rng, (train_data, val_data), model, ps, st; eval_iters, block_size, batch_size=32)
        println("Epoch [$epoch]: Training loss $(loss[1]), validation loss $(loss[2])")
    end
end


decode(generate(rng, ones(Int, 1, 1), 500, model, ps, st)[:, 1])

