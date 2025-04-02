#==*=*=*=*=*=*=*
    
    FIGURE 6

=*=*=*=*=*=*=*=#

using Plots
using Distributions
using AdaptiveMCMC
using JLD2
using .Threads
using Interpolations

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("defaults.jl")
include("../default_setup.jl")
include("../AltGamma.jl")

## FUNCTION TO TAKE CONVOLUTION
xwide_default = range(-2.0,2.0,401)
function add_noise(x,pd,d;xwide=xwide_default,ret=:marginal)
    p = linear_interpolation(x,max.(0.0,pd),extrapolation_bc=Line())
    f(x,y) = max.(0.0,p(x) * pdf.(d,y))
    if ret == :joint
        return (x,z) -> f(x,z - x)
    end
    q = [trap(f.(xwide,z .- xwide),xwide) for z in xwide]
    q /= trap(q,xwide)  # Fix issues integrating with numerical solution to PDE
    linear_interpolation(xwide,max.(0.0,q),extrapolation_bc=Line())
end

## SETUP EXPERIMENT, SAMPLE DATA
n₀      = 48 * 4
tmax    = 7.0
T       = 1:2:7  # Time points
x₀      = Normal(0.0,β / sqrt(2ν))
d       = t -> true
σ       = 0.25

@time ibm = [simulate_ibm(λ,v,d,β;n₀,x₀,tmax) for _ = eachindex(T)];

data = [ibm_statistics(ibm[i]...)[2](T[i]) + 
            σ * randn(length(ibm_statistics(ibm[i]...)[2](T[i]))) 
            for i in eachindex(T)]

## SETUP LIKELIHOOD FUNCTION (NORMAL NOISE)
function loglike(q)
    # Extract parameters (only β and σ are unknown now)
    lβ,lσ = q
    # Solve PDE
    xpde,_,ppde = solve_pde(λ,v,d,exp(lβ);tmax=7.0,n₀,x₀=Normal(0.0,exp(lβ) / sqrt(2ν)));
    # Add noise
    q = t -> add_noise(xpde(t),ppde(t),Normal(0.0,exp(lσ)))
    # Calculate log-likelihood
    sum(sum(log.(q(T[i]).(data[i]))) for i in eachindex(T))
end

# Extended set of parameters
q = [log(β),log(σ)]
prior_q = Product([prior.v[end],Uniform(-6.0,0.0)])

# MCMC for all parameters
logpost(q) = insupport(prior_q,q) ? loglike(q) + logpdf(prior_q,q) : -Inf

@time res3 = adaptive_rwm(q, logpost, 10000; algorithm=:aswam);
jldsave("fig6_res3.jld2"; res3)

## SETUP LIKELIHOOD FUNCTION (SKEWED NOISE)
function loglike2(q)
    # Extract parameters (only β and σ are unknown now)
    lβ,lσ,ω = q
    # Solve PDE
    xpde,_,ppde = solve_pde(λ,v,d,exp(lβ);tmax=7.0,n₀,x₀=Normal(0.0,exp(lβ) / sqrt(2ν)));
    # Add noise
    q = t -> add_noise(xpde(t),ppde(t),GammaAlt(0.0,exp(lσ),ω))
    # Calculate log-likelihood
    sum(sum(log.(q(T[i]).(data[i]))) for i in eachindex(T))
end

# Extended set of parameters
q2 = [log(β),log(σ),0.0]
prior_q2 = Product([prior.v[end]; Uniform(-6.0,0.0); Uniform(-1.0,1.0)])

# MCMC for all parameters
logpost2(q) = insupport(prior_q2,q) ? loglike2(q) + logpdf(prior_q2,q) : -Inf

@time res4 = adaptive_rwm(q2, logpost2, 10000; algorithm=:aswam);

## FIGURE 6

    # MEASUREMENT NOISE PARAMETERS / FIGURE
    xpde,npde,ppde = solve_pde(λ,v,d,β;tmax=28.0,n₀,x₀);
    xwide = range(-2.0,3.0,401)
    g = t -> add_noise(xpde(t),ppde(t),Normal(0.0,σ);xwide,ret=:joint)

    G₀ = [g(0.0)(xᵢ,zᵢ) for xᵢ in xwide, zᵢ in xwide]
    G₂ = [g(2.0)(xᵢ,zᵢ) for xᵢ in xwide, zᵢ in xwide]
    G₂₈ = [g(28.0)(xᵢ,zᵢ) for xᵢ in xwide, zᵢ in xwide]

    fig6a = plot(colorbar=false)
    plot!(fig6a,xwide,xwide,G₀',st=:contour,lw=2.0,xlim=(-0.6,1.6),ylim=(-0.6,1.6),c=:blue)
    plot!(fig6a,xwide,xwide,G₂',st=:contour,lw=2.0,xlim=(-0.6,1.6),ylim=(-0.6,1.6),c=:grey,α=0.5)
    plot!(fig6a,xwide,xwide,G₂₈',st=:contour,lw=2.0,xlim=(-0.6,1.6),ylim=(-0.6,1.6),c=:red)
    plot!(fig6a,[[],[],[]],lw=2.0,c=[:blue :grey :red],label=["Sensitive (t = 0 d)" "(t = 2 d)" "Addicted (t = 28 d)"],)
    plot!(fig6a,xlabel="Growth rate",ylabel="Marker")

    # Posterior (non-skewed)
    fig6b = scatter(res3.X[1,1:10:end],res3.X[2,1:10:end],c=:black,msw=0.0,α=0.1,
                xlim=extrema(prior_q.v[end-1]),ylim=(-2.5,-1.0),label="Posterior")
    scatter!(fig6b,[q[1]],[q[2]],c=:blue,m=:diamond,label="True value")
    plot!(fig6b,xlabel="log(β)",ylabel="log(σ)",legend=:bottomleft)

    # Posterior (skewed)
    fig6c = scatter(res4.X[1,1:10:end],res4.X[2,1:10:end],c=:black,msw=0.0,α=0.1,
                xlim=extrema(prior_q2.v[1]),ylim=(-2.5,-1.0),label="Posterior")
    scatter!(fig6c,[q2[1]],[q2[2]],c=:blue,m=:diamond,label="True value")
    plot!(fig6c,xlabel="log(β)",ylabel="log(σ)",legend=:bottomleft)

    # Figure 6
    fig6 = plot(fig6a,fig6b,fig6c,layout=@layout([a{0.4w} b c]),size=(720,200))
    add_plot_labels!(fig6)
    savefig("fig6.svg")