#==*=*=*=*=*=*=*
    
    FIGURE 7
 
=*=*=*=*=*=*=*=#

using Plots
using Distributions
using AdaptiveMCMC
using JLD2
using Combinatorics
using .Threads
using Suppressor
using LinearAlgebra
using Optimization, OptimizationNLopt
using ForwardDiff

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("../defaults.jl")
include("../default_setup.jl")

## SETUP EXPERIMENTAL DESIGN

    n₀      = IC
    R       = 48
    d       = t -> true
    T       = 1:2:7  # Time points
    x₀      = Normal(0.0,β / sqrt(2ν))  # "Fully" sensitive cells
    d       = t -> true

## GENERATE EXPERIMENTAL DATA

    function proliferation_assay(t;R=R)
        [begin
            ibmᵢ = simulate_ibm(λ,v,d,β;n₀,x₀,tmax=t)
            ibm_statistics(ibmᵢ...)[1](t)
        end for _ = 1:R]
    end

    @time data = [proliferation_assay(t) for t in T]

## SETUP LIKELIHOOD FUNCTION, PRIOR, POSTERIOR

function ode(x,p,t)
    a,b,c,d = p
    a * sign(1 - x) + (1 - x) * (b + c * x + d * x^2)
end

function loglike(p)
    # Solve ODE
    x = solve(ODEProblem(ode,eps(),(0.0,7.0),p),Euler(),dt=0.01,verbose=false)
    # Exit early
    !all(isfinite.(x.u)) && return -Inf
    # Birth and death rates...
    rbirth(t) = max(0.0,λ(x(t),d(t)))
    rdeath(t) = -min(0.0,λ(x(t),d(t)))
    # Solve CME
    Ncme,Qcme = solve_cme(rbirth,rdeath;n₀,tmax=maximum(T));
    # Calculate likelihood
    sum(sum(log.(max.(0.0,Qcme(T[i])[data[i] .+ 1]))) for i in eachindex(T))
end

prior = Product(Uniform.(fill(-3.0,4),fill(3.0,4)))

## FIND THE MLE FOR EACH MODEL

function find_mle(idx;maxtime=30)
    # Variables that are NOT zero
    nidx = setdiff(1:4,idx)
    q2p = q -> begin
        p = zeros(4); p[nidx] = q; return p
    end
    obj = (x,p) -> -loglike(q2p(x))
    q = rand(Product(prior.v[nidx]))
    while !isfinite(obj(q,[]))
        q = rand(Product(prior.v[nidx]))
    end
    prob = OptimizationProblem(obj, rand(length(nidx)), [], lb = minimum.(prior.v)[nidx], ub = maximum.(prior.v)[nidx])
    sol = solve(prob,NLopt.GN_DIRECT();maxtime)
    q2p(sol.u),loglike(q2p(sol.u))
end

idxs = [[[]];collect(combinations(1:4))[1:end-1]]

P = similar(idxs)
L = zeros(length(idxs))

@time @threads for i = eachindex(idxs)
    P[i],L[i] = find_mle(idxs[i];maxtime=600)
end

## Refine global maximum if needed
if maximum(L) != L[1]
    obj = (x,p) -> -loglike(x)
    p = Float64.(P[findmax(L)[2]])
    prob = OptimizationProblem(obj, p, [], lb = minimum.(prior.v), ub = maximum.(prior.v))
    sol = solve(prob,NLopt.LN_BOBYQA();maxtime=600)
    P[1] = sol.u; L[1] = loglike(sol.u)
end

## Figure 7a (likelihood-ratio test)
fig7a = hline([0.0],lw=2.0,c=:black,ls=:dash,label="Maximum")

# Length 0
ik = (1:15)[length.(idxs) .== 0]
scatter!(fig7a,ik,max.(L[ik] .- L[1],-5.0),c=:black,label="")

# Length 1
ik = (1:15)[length.(idxs) .== 1]
scatter!(fig7a,ik,max.(L[ik] .- L[1],-5.0),c=:blue,label="")
hline!([-quantile(Chisq(1),0.95) / 2],c=:blue,ls=:dash,lw=2.0,label="k = 3")

# Length 2
ik = (1:15)[length.(idxs) .== 2]
scatter!(fig7a,ik,max.(L[ik] .- L[1],-5.0),c=:red,label="")
hline!([-quantile(Chisq(2),0.95) / 2],c=:red,ls=:dash,lw=2.0,label="k = 2")

# Length 3
ik = (1:15)[length.(idxs) .== 3]
scatter!(fig7a,ik,max.(L[ik] .- L[1],-5.0),c=:orange,label="")
hline!([-quantile(Chisq(3),0.95) / 2],c=:orange,ls=:dash,lw=2.0,label="k = 1")

lets = ["a","b","c","d"]
labs = ["["*join(lets[idx],",")*"]" for idx in idxs]

plot!(fig7a, xticks=(1:15,labs),xrotation=45,legend=:none)

## Plot inferred v(x)...

ik = [1,14,15]

# sgn fix for plotting
sgn(x) = x == 0 ? 1.0 : sign(x)

function v(x,p)
    a,b,c,d = p
    a * sgn(1 - x) + (1 - x)*(b + c * x + d * x^2)
end
function x(p)
    ode(x,p,t) = v(x,p)
    solve(ODEProblem(ode,eps(),(0.0,7.0),p),Euler(),dt=0.01)
end

fig7b = plot()
for i = eachindex(ik)
    plot!(fig7b,x -> v(x,P[ik[i]]),xlim=(0.0,1.0),label=labs[ik[i]])
end
plot!(fig7b,xlabel="x",ylabel="v(x)")

fig7c = plot()
for i = eachindex(ik)
    plot!(fig7c,x(P[ik[i]]),xlim=(0.0,7.0),label=labs[ik[i]])
end
plot!(fig7c,xlabel="t",ylabel="x(t)")

fig7 = plot(fig7a,fig7b,fig7c,layout=@layout([a{0.45w} b c]),size=(750,190))
add_plot_labels!(fig7)
savefig("fig7.svg")