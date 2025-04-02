#==*=*=*=*=*=*=*
    
    FIGURE S3

    Inference using a lot of data (β becomes identifiable)

=*=*=*=*=*=*=*=#

using Plots, StatsPlots
using Distributions
using AdaptiveMCMC
using JLD2
using .Threads

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("defaults.jl")
include("../default_setup.jl")

## SETUP EXPERIMENTAL DESIGN

    n₀ = IC     # Number of cells in window 
    R  = 4*192  # Number of experimental replicates     
    T  = 0.5:0.5:7.0
    X₀ = (ν,β) -> [Normal(0.0,β / sqrt(2ν)),  # "Fully" sensitive cells
                   Normal(1.0,β / sqrt(2ν))]  # "Fully" addicted cells
    D  = [t -> true,t -> false]   # Dosing on/off

## GENERATE EXPERIMENTAL DATA

function proliferation_assay(d,x₀,t;R=R)
    [begin
        ibmᵢ = simulate_ibm(λ,v,d,β;n₀,x₀,tmax=t)
        ibm_statistics(ibmᵢ...)[1](t)
     end for _ = 1:R]
end

@time data = [proliferation_assay(d,x₀,t) for t in T, d in D, x₀ in X₀(ν,β)]

## SETUP LIKELIHOOD FUNCTION
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

## (a) MCMC for all parameters
logpost(p) = insupport(prior,p) ? loglike(p) + logpdf(prior,p) : -Inf
@time res1 = adaptive_rwm(p, logpost, 10000; algorithm=:aswam);
jldsave("figS3_res1.jld2"; res1)

## Figure 3

# Calculate mean (i.e., from the PDE) as a function of time for each
p̂ = res1.X[:,findmax(res1.D[1])[2]]
λ̂(x,d) = d ? p̂[2] + (p̂[4] - p̂[2]) * x : p̂[1] + (p̂[3] - p̂[1]) * x
v̂(x,d) = -exp(p̂[5]) * (x - (d ? 1.0 : 0.0))
n = [solve_pde(λ̂,v̂,d,exp(p̂[end]);n₀,x₀,tmax=maximum(T)+2)[2] for d in D, x₀ in X₀(exp.(p̂[5:6])...)];

# Data plot
figS3a = plot(xlabel="Time [d]",ylabel="Cell count",xticks=0:1:7,widen=true)
boxplot!(figS3a,collect(T .- 0.2)',data[:,1,1],bar_width=0.35,lc=:blue,α=0.5,c=:blue,label=["Drug On" "" "" ""])
boxplot!(figS3a,collect(T .+ 0.2)',data[:,2,1],bar_width=0.35,lc=:red,α=0.5,c=:red,label=["Drug Off" "" "" ""])
plot!(figS3a,[n[1,1],n[2,1]];c=[:blue :red],lw=2.0,xlim=(0.0,7.5),label="",ylim=(0.0,30.0))

figS3b = plot(xlabel="Time [d]",ylabel="Cell count",xticks=0:1:7,widen=true)
boxplot!(figS3b,collect(T .- 0.2)',data[:,1,2],bar_width=0.35,lc=:blue,α=0.5,c=:blue,label=["Drug On" "" "" ""])
boxplot!(figS3b,collect(T .+ 0.2)',data[:,2,2],bar_width=0.35,lc=:red,α=0.5,c=:red,label=["Drug Off" "" "" ""])
plot!(figS3b,[n[1,2],n[2,2]];c=[:blue :red],lw=2.0,xlim=(0.0,7.5),label="",ylim=(0.0,30.0))

figS3c = stephist(res1.X[5,:],xlim=extrema(prior.v[5]),c=:black,α=0.5,frange=0.0,lw=0.0,
    xlabel="log(ν)",widen=true,normalize=:pdf,label="Posterior")
plot!(figS3c,prior.v[5],c=:blue,lw=2.0,label="Prior",legend=:topleft,ylabel="Density",ywiden=false)
vline!(figS3c,[p[5]],c=:black,lw=2.0,ls=:dash,label="True")
vline!(figS3c,[p̂[5]],c=:red,lw=2.0,ls=:dash,label="MAP",ylim=(0.0,2.4))

figS3d = stephist(res1.X[6,:],xlim=extrema(prior.v[6]),c=:black,α=0.5,frange=0.0,lw=0.0,
    xlabel="log(β)",widen=true,normalize=:pdf,label="Posterior")
plot!(figS3d,prior.v[6],c=:blue,lw=2.0,label="Prior",ylabel="Density",ywiden=false)
vline!(figS3d,[p[6]],c=:black,lw=2.0,ls=:dash,label="True")
vline!(figS3d,[p̂[6]],c=:red,lw=2.0,ls=:dash,label="MAP",ylim=(0.0,1.5))

figS3 = plot(figS3a,figS3b,figS3c,figS3d,layout=grid(1,4),size=(900,180))
add_plot_labels!(figS3)
savefig("figS3.svg")
figS3