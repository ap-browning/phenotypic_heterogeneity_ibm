#==*=*=*=*=*=*=*
    
    FIGURE S6

    Figure 3, but accounting for the correlation structure

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
    R  = 192    # Number of experimental replicates     
    T  = 1:2:7  # Time points
    X₀ = (ν,β) -> [Normal(0.0,β / sqrt(2ν)),  # "Fully" sensitive cells
                   Normal(1.0,β / sqrt(2ν))]  # "Fully" addicted cells
    D  = [t -> true,t -> false]   # Dosing on/off

## GENERATE EXPERIMENTAL DATA

function correlated_proliferation_assay(d,x₀,t;R=R)
    [begin
        ibmᵢ = simulate_ibm(λ,v,d,β;n₀,x₀,tmax=maximum(t))
        ibm_statistics(ibmᵢ...)[1].(t)
     end for _ = 1:R]
end

data = [hcat(correlated_proliferation_assay(d,x₀,T)...) for d in D, x₀ in X₀(ν,β)]

## Likelihood (QME)
function loglike(p)

    # Extract parameters
    γ₁,γ₂,γ₃,γ₄,lν,lβ = p
    # Proliferation rate function
    λ(x,d) = d ? γ₂ + (γ₄ - γ₂) * x : γ₁ + (γ₃ - γ₁) * x
    # Advection "strength"
    v(x,d) = -exp(lν) * (x - (d ? 1.0 : 0.0))

    # Initiate likelihood
    ll = 0.0

    # Loop through conditions
    for (id,d) in enumerate(D), (ix,x₀) in enumerate(X₀(exp(lν),exp(lβ)))

        # Solve PDE
        xpde,npde,ppde = solve_pde(λ,v,d,exp(lβ);n₀,x₀,tmax=maximum(T));

        data_cond = data[id,ix]

        # Loop through times
        Ncme,Qcme = solve_cme(xpde,ppde,λ,d;n₀,tmin=0.0,tmax=T[1],saveat=T[1]);
        ll += sum(log.(max.(0.0,Qcme(T[1])[data_cond[1,:] .+ 1])))
        lleach = zeros(length(T) - 1)
        @threads for j = 2:(length(T) - 1)
            # Loop through unique initial conditions
            unique_ic = unique(data_cond[j-1,:])
            for n̂₀ in unique_ic
                Ncme,Qcme = solve_cme(xpde,ppde,λ,d;n₀=n̂₀,tmin=T[j-1],tmax=T[j],saveat=T[j]);
                idxs = findall(data_cond[j-1,:] .== n̂₀)
                lleach[j] = sum(log.(max.(0.0,Qcme(T[j])[data_cond[j,idxs] .+ 1])))
            end
        end
        ll += sum(lleach)

    end

    return ll

end

## MCMC for all parameters
logpost(p) = insupport(prior,p) ? loglike(p) + logpdf(prior,p) : -Inf
@time res1 = adaptive_rwm(p, logpost, 10000; algorithm=:aswam);


## Figure S6

# Solve CME at the MAP
p̂ = res1.X[:,findmax(res1.D[1])[2]]
λ̂(x,d) = d ? p̂[2] + (p̂[4] - p̂[2]) * x : p̂[1] + (p̂[3] - p̂[1]) * x
v̂(x,d) = -exp(p̂[5]) * (x - (d ? 1.0 : 0.0))
Ncme = [solve_cme(λ̂,v̂,d,exp(p̂[end]);n₀,x₀,tmax=maximum(T)+2)[1] for d in D, x₀ in X₀(exp.(p̂[5:6])...)];
Qcme = [solve_cme(λ̂,v̂,d,exp(p̂[end]);n₀,x₀,tmax=maximum(T)+2)[2] for d in D, x₀ in X₀(exp.(p̂[5:6])...)];

Nqnt = cme_quantiles.(Ncme,Qcme,[q.t for q in Qcme];p=[0.25,0.5,0.75])

# Calculate mean (i.e., from the PDE) as a function of time for each
p̂ = res1.X[:,findmax(res1.D[1])[2]]
λ̂(x,d) = d ? p̂[2] + (p̂[4] - p̂[2]) * x : p̂[1] + (p̂[3] - p̂[1]) * x
v̂(x,d) = -exp(p̂[5]) * (x - (d ? 1.0 : 0.0))
n = [solve_pde(λ̂,v̂,d,exp(p̂[end]);n₀,x₀,tmax=maximum(T)+2)[2] for d in D, x₀ in X₀(exp.(p̂[5:6])...)];

# Data plot
figS6a = plot(xlabel="Time [d]",ylabel="Cell count",xticks=0:1:7,widen=true,ylim=(0.0,45.0))
plot!(T,data[1,1],c=:red,lw=1.0,m=:circle,α=0.3,label="")
plot!(T,data[2,1],c=:blue,lw=1.0,m=:circle,α=0.3,label="")


figS6b = plot(xlabel="Time [d]",ylabel="Cell count",xticks=0:1:7,widen=true,ylim=(0.0,45.0))
plot!(T,data[1,2],c=:red,lw=1.0,m=:circle,α=0.3,label="")
plot!(T,data[2,2],c=:blue,lw=1.0,m=:circle,α=0.3,label="")

figS6c = stephist(res1.X[5,:],xlim=extrema(prior.v[5]),c=:black,α=0.5,frange=0.0,lw=0.0,
    xlabel="log(ν)",widen=true,normalize=:pdf,label="Posterior")
plot!(figS6c,prior.v[5],c=:blue,lw=2.0,label="Prior",legend=:topleft,ylabel="Density",ywiden=false)
vline!(figS6c,[p[5]],c=:black,lw=2.0,ls=:dash,label="True")
vline!(figS6c,[p̂[5]],c=:red,lw=2.0,ls=:dash,label="MAP",ylim=(0.0,2.4))

figS6d = stephist(res1.X[6,:],xlim=extrema(prior.v[6]),c=:black,α=0.5,frange=0.0,lw=0.0,
    xlabel="log(β)",widen=true,normalize=:pdf,label="Posterior")
plot!(figS6d,prior.v[6],c=:blue,lw=2.0,label="Prior",ylabel="Density",ywiden=false)
vline!(figS6d,[p[6]],c=:black,lw=2.0,ls=:dash,label="True")
vline!(figS6d,[p̂[6]],c=:red,lw=2.0,ls=:dash,label="MAP",ylim=(0.0,0.48))

figS6 = plot(figS6a,figS6b,figS6c,figS6d,layout=grid(1,4),size=(900,180))
add_plot_labels!(figS6)
savefig("figS6.svg")
figS6