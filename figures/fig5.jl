#==*=*=*=*=*=*=*
    
    FIGURE 5

=*=*=*=*=*=*=*=#

using Plots, StatsPlots
using Distributions
using AdaptiveMCMC
using JLD2
using Loess

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("defaults.jl")
include("../default_setup.jl")

## SIMULATE A HUGE EXPERIMENT, RECORD CELL BIRTHS AND DEATH TIMES
n₀ = 10000
x₀ = Normal(0.0,β / sqrt(2ν))
d  = t -> true
tmax = 7.0

@time ibmᵢ = simulate_ibm(λ,v,d,β;n₀,x₀,tmax);

# The "data" (total population, plus birth and death times)
n = ibm_statistics(ibmᵢ...)[1]
tbirth = ibmᵢ[2]; tbirth = tbirth[tbirth .> 0.0]
tdeath = ibmᵢ[3]; tdeath = tdeath[tdeath .< 7.0]

# Discretised data
T = range(0.0,tmax,200); T̂ = (T[1:end-1] + T[2:end]) / 2;
@time N = n.(T̂)
C₁ = [count(T[i] .< tbirth .< T[i+1]) for i = eachindex(T̂)]
C₂ = [count(T[i] .< tdeath .< T[i+1]) for i = eachindex(T̂)]

## SETUP LIKELIHOOD FUNCTION
function loglike(p)
    # Extract parameters
    γ₁,γ₂,γ₃,γ₄,lν,lβ = p
    # Proliferation rate function
    λ(x,d) = d ? γ₂ + (γ₄ - γ₂) * x : γ₁ + (γ₃ - γ₁) * x
    # Advection "strength"
    v(x,d) = -exp(lν) * (x - (d ? 1.0 : 0.0))
    # Solve PDE
    xpde,_,ppde = solve_pde(λ,v,d,exp(lβ);tmax,n₀,x₀);
    # Rates, discretised rates
    rbirth(t) = trap(ppde(t) .* max.(0.0,λ.(xpde(t),d(t))),xpde(t))
    rdeath(t) = -trap(ppde(t) .* min.(0.0,λ.(xpde(t),d(t))),xpde(t))
    R₁ = max.(0.0,rbirth.(T̂) .* N * diff(T)[1])
    R₂ = max.(0.0,rdeath.(T̂) .* N * diff(T)[1])
    # Poisson likelihood
    sum(logpdf.(Poisson.(R₁),C₁)) + sum(logpdf.(Poisson.(R₂),C₂))
end

## (a) MCMC for all parameters
logpost(p) = insupport(prior,p) ? loglike(p) + logpdf(prior,p) : -Inf

@time res2 = adaptive_rwm(p, logpost, 10000; algorithm=:aswam);
jldsave("fig5_res2.jld2"; res2)

## Figure 5

# Observed rates
R̃₁ = C₁ ./ N / diff(T)[1]
R̃₂ = C₂ ./ N / diff(T)[1]

# MAP
p̂ = res2.X[:,findmax(res2.D[1])[2]]

# Create figure
fig5a = plot(ywiden=false,xlabel="Time [d]",ylabel="Rate")

# Rug plot
height = 0.05
for i = 1:500
    plot!(fig5a,rand(tbirth)*[1,1],[0.0,height],c=:blue,α=0.1,label="")
    plot!(fig5a,rand(tdeath)*[1,1],[0.0,height],c=:red,α=0.1,label="")
end

# LOESS of rates
plot!(fig5a,T̂,predict(loess(T̂,R̃₁,span=0.5),T̂),c=:blue,lw=2.0,label="Birth")
plot!(fig5a,T̂,predict(loess(T̂,R̃₂,span=0.5),T̂),c=:red,lw=2.0,label="Death")

# Posterior
fig5b = stephist(res2.X[6,:],xlim=extrema(prior.v[6]),c=:black,α=0.5,lw=0.0,frange=0.0,
    xlabel="log(β)",widen=true,normalize=:pdf,label="Posterior")
plot!(fig5b,prior.v[6],c=:blue,lw=2.0,label="Prior",ylabel="Density")
vline!(fig5b,[p[6]],c=:black,lw=2.0,ls=:dash,label="True")
vline!(fig5b,[p̂[6]],c=:red,lw=2.0,ls=:dash,label="MAP",ywiden=false)

fig5 = plot(fig5a,fig5b,size=(600,220))
add_plot_labels!(fig5)
savefig("fig5.svg")