#==*=*=*=*=*=*=*
    
    FIGURE S2

=*=*=*=*=*=*=*=#

using Plots, StatsPlots
using Distributions
using AdaptiveMCMC
using JLD2
using .Threads

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("../defaults.jl")
include("../default_setup.jl")

## SETUP EXPERIMENTAL DESIGN

    n₀ = IC     # Number of cells in window 
    R  = 48     # Number of experimental replicates     
    T  = 1:2:7  # Time points
    X₀ = (ν,β) -> [Normal(0.0,β / sqrt(2ν)),  # "Fully" sensitive cells
                   Normal(1.0,β / sqrt(2ν))]  # "Fully" addicted cells
    D  = [t -> true,t -> false]   # Dosing on/off

    function proliferation_assay(d,x₀,t;R=R)
        [begin
            ibmᵢ = simulate_ibm(λ,v,d,β;n₀,x₀,tmax=t)
            ibm_statistics(ibmᵢ...)[1](t)
        end for _ = 1:R]
    end

## GENERATE EXPERIMENTAL DATA

function sample_posterior(T,R;iters=10000)

    # Generate data
    data = [proliferation_assay(d,x₀,t;R) for t in T, d in D, x₀ in X₀(ν,β)]

    # Likelihood function
    function loglike(p)
        # Extract parameters
        γ₁,γ₂,γ₃,γ₄,lν,lβ = p
        # Proliferation rate function
        λ(x,d) = d ? γ₂ + (γ₄ - γ₂) * x : γ₁ + (γ₃ - γ₁) * x
        # Advection "strength"
        v(x,d) = -exp(lν) * (x - (d ? 1.0 : 0.0))
        # Calculate log-likelihood
        ll = 0.0
        for (i,d) = enumerate(D), (j,x₀) = enumerate(X₀(exp(lν),exp(lβ)))
            # Solve PDE, CME
            xpde,npde,ppde = solve_pde(λ,v,d,exp(lβ);n₀,x₀,tmax=maximum(T));
            Ncme,Qcme = solve_cme(xpde,ppde,λ,d;n₀,tmax=maximum(T));
            # Compute log-likelihood
            ll += sum(sum(log.(max.(0.0,Qcme(T[k])[data[k,i,j] .+ 1]))) for k in eachindex(T))
        end
        return ll
    end

    # Posterior
    logpost(p) = insupport(prior,p) ? loglike(p) + logpdf(prior,p) : -Inf

    # MCMC
    adaptive_rwm(p, logpost, iters; algorithm=:aswam);

end

T = [[3],[1,5],[1,3,5,7],collect(1:7)]
R = 48

#res = Array{Any}(undef,length(T))

#@time @threads for i = eachindex(T)
for i = 4:4
    res[i] = sample_posterior(T[i],R;iters=10000)
end

## Produce plots...
pars = ["γ₁","γ₂","γ₃","γ₄","log ν","log β"]
lims = extrema.(prior.v)
lims[1] = (0.1,0.2)
lims[2] = (-1.0,0.0)
lims[3] = (0.0,0.2)
lims[4] = (0.05,0.15)

function posterior_plot(i,j)
    plt = stephist(res[i].X[j,:][1:10:end],c=:black,α=0.5,frange=0.0,lw=0.0,
        xlabel=pars[j],xwiden=true,ywiden=false,normalize=:pdf,label="")
    plot!(plt,prior.v[j],c=:blue,lw=2.0,label="",ylabel="Density",ywiden=false)
    vline!(plt,[p[j]],c=:black,lw=2.0,ls=:dash,label="")
    plot!(plt,xlim=lims[j])
    plt
end

plts = [posterior_plot(i,j) for j = eachindex(pars), i = eachindex(res)]

figS2 = plot(plts...,layout=grid(length(res),length(pars)),size=(1200,700),link=:x)
savefig("figS2.svg")