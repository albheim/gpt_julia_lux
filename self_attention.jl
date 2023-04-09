using NNlib
using Random, Statistics, LinearAlgebra

rng = Random.default_rng()
Random.seed!(rng, 37)

C, T, B = 2, 8, 4
x = randn(C, T, B)

function f1(x; T, B)
    xbow = zeros(C, T, B)
    for b in 1:B
        for t in 1:T
            xprev = x[:, 1:t, b]
            xbow[:, t, b] .= mean(xprev, dims=2)
        end
    end
    xbow
end
xbow = f1(x; T, B)

wei = UpperTriangular(ones(T, T))
wei = wei ./ sum(wei, dims=1)
function f2(x; wei)
    NNlib.batched_mul(x, wei)
end
xbow2 = f2(x; wei)

tril = UpperTriangular(ones(T, T))
wei = zeros(T, T)
wei[tril .== 0] .= -Inf
wei = softmax(wei, dims=1)
xbow3 = NNlib.batched_mul(x, wei)


# Single head
B,T,C = 4,8,32
head_size = 16
x = randn(C, T, B)

key = Dense(C => head_size)
query = Dense(C => head_size)
value = Dense(C => head_size)

pk, sk = Lux.setup(rng, key)
pq, sq = Lux.setup(rng, query)
pv, sv = Lux.setup(rng, value)

k, = key(x, pk, sk)
q, = query(x, pq, sq)

wei = batched_mul(permutedims(k, (2, 1, 3)), q)

tril = UpperTriangular(ones(T, T))
wei[tril .== 0, :] .= -Inf
wei = softmax(wei, dims=1)

v, = value(x, pv, sv)
out = NNlib.batched_mul(v, wei)

wei = batched_mul(permutedims(k, (2, 1, 3)), q)

function f1(wei, tril)
    wei[tril .== 0, :] .= -Inf
end

function f2(wei)
    for k in 1:size(wei, 3)
        for i in 1:size(wei, 2)
            for j in i+1:size(wei, 1)
                wei[j, i, k] = -Inf
            end
        end
    end
end

T = 500
@btime f1(x, tril) setup=begin x = randn(T, T, B); tril = UpperTriangular(ones(T, T)) end
@btime f2(x) setup=(x = randn(T, T, B))