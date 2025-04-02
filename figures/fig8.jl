#==*=*=*=*=*=*=*
    
    FIGURE 8

=*=*=*=*=*=*=*=#

using Polynomials

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("../discrete_ibm.jl")
include("../discrete_cme_ibm.jl")
include("defaults.jl")

## Discrete model switching rates  (on drug only)
r₀₁ = 1.0
r₁₀ = 0.01

## Discrete model growth rates
λ₀ = -0.30
λ₁ =  0.10

## Intermediate calculations...
pmin = 0.0  # WLOG
pmax = maximum(roots(Polynomial([r₀₁,λ₁ - λ₀ - r₀₁ - r₁₀,λ₀ - λ₁])))

# Continuous model growth rates...
    # p = pmax corresponds to x = 1 in the continuous model
g₀ = λ₀
g₁ = λ₀ + (λ₁ - λ₀) * pmax

λ(x,d) = d ? g₀ + (g₁ - g₀) * x : 0.0   # only look at drug on...

# Advection "strength"
B = sqrt((r₀₁ + r₁₀ + λ₀ - λ₁)^2 + 4r₀₁ * (λ₁ - λ₀))
C = r₀₁ + r₁₀ + λ₀ - λ₁
v(x,d) = d ? (B + C) / 2 - C * x - (B - C) / 2 * x^2 : 0.0

## SIMULATE EACH MODEL

    # Convert to growth and death rates (ONLY FOR DRUG ON)
    g = [d -> max(0,λ₀), d -> max(0,λ₁)]
    δ = [d -> -min(0,λ₀), d -> -min(0,λ₁)]
    r = [d -> r₀₁, d -> r₁₀]

    # Final time, drug, monitoring times
    d = t -> true
    tmax = 7.0
    T = 0:1:7

    # Solve PDE for the continuous model
    β = 0.05
    xpde,npde,ppde = solve_pde(λ,v,d,β;n₀=mean(IC),x₀=Normal(0.0,β / sqrt(2*0.4)),tmax);

    # 2D initial condition
    n₀ = Product([IC,Binomial(0,0)])

    # Solve corresponding CMEs
    N_bin,Qcme_bin = solve_discrete_cme(g,δ,r,d;n₀,Nmax=50,tmax);
    N_con,Qcme_con = solve_cme(xpde,ppde,λ,d;n₀=IC,tmax);

## Figure 8

    # (a) - mean cell count
    μ_bin(t) = sum(N_bin .* Qcme_bin(t))
    μ_con(t) = sum(N_con .* Qcme_con(t))

    fig8a = plot(μ_bin,xlim=(0.0,7.0),c=:red,lw=2.0,label="Discrete CME")
    plot!(μ_con,xlim=(0.0,7.0),c=:black,ls=:dashdot,lw=2.0,label="Continuous CME")
    plot!(fig8a,widen=true,xticks=0:1:7,xlabel="Time [d]",ylabel="Mean cell count")

    fig8b = plot()
    plot!(fig8b,hist_coord(N_bin,Qcme_bin(1.0))...,frange=0.0,α=0.5,lw=0.0,label="Discrete CME")
    plot!(fig8b,hist_coord(N_con,Qcme_con(1.0))...,c=:black,lw=2.0,ls=:dash,label="Continuous CME")
    plot!(fig8b,xlim=(0.0,20.0),ylim=(0.0,0.16))

    fig8c = plot()
    plot!(fig8c,hist_coord(N_bin,Qcme_bin(3.0))...,frange=0.0,α=0.5,lw=0.0,label="Discrete CME")
    plot!(fig8c,hist_coord(N_con,Qcme_con(3.0))...,c=:black,lw=2.0,ls=:dash,label="Continuous CME")
    plot!(fig8c,xlim=(0.0,20.0),ylim=(0.0,0.16))

    fig8d = plot()
    plot!(fig8d,hist_coord(N_bin,Qcme_bin(7.0))...,frange=0.0,α=0.5,lw=0.0,label="Discrete CME")
    plot!(fig8d,hist_coord(N_con,Qcme_con(7.0))...,c=:black,lw=2.0,ls=:dash,label="Continuous CME")
    plot!(fig8d,xlim=(0.0,40.0),ylim=(0.0,0.1))

    fig8 = plot(fig8a,fig8b,fig8c,fig8d,xwiden=true,xlabel="Cell count",
        layout=grid(1,4),size=(800,180))
    plot!(fig8,subplot=1,xlabel="Time [d]")
    add_plot_labels!(fig8)
    savefig("fig8.svg")