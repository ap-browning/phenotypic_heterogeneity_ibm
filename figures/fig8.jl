#==*=*=*=*=*=*=*
    
    FIGURE 8

=*=*=*=*=*=*=*=#

using Polynomials
using ForwardDiff

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("../discrete_ibm.jl")
include("../discrete_cme_ibm.jl")
include("../default_setup.jl")
include("defaults.jl")


## Setup discrete model

    # Discrete model switching rates
        #          drug       off drug
    r₀₁ = d -> d ? 1.0      : 0.02
    r₁₀ = d -> d ? 0.01     : 0.50

    # Discrete model growth rates
        #          drug       off drug
    λ̃₀ = d -> d ? -0.30     : 0.15      # sensitive
    λ̃₁ = d -> d ?  0.10     : 0.10      # resistant

    # Range of x̃(t)
    pmin = minimum(roots(Polynomial([r₀₁(false),λ̃₁(false) - λ̃₀(false) - r₀₁(false) - r₁₀(false),λ̃₀(false) - λ̃₁(false)])))
    pmax = maximum(roots(Polynomial([r₀₁(true),λ̃₁(true) - λ̃₀(true) - r₀₁(true) - r₁₀(true),λ̃₀(true) - λ̃₁(true)])))

## Setup continuous model

    # Growth rates
        # Discrete model has x̃ ∈ [pmin,pmax]
        # Continuous model has x ∈ [0,1]
        # x̃ = pmin + (pmax - pmin) * x

    # Continuous model growth extrema
    λ₀ = d -> λ̃₀(d) * (1 - pmin) + λ̃₁(d) * pmin
    λ₁ = d -> λ̃₀(d) * (1 - pmax) + λ̃₁(d) * pmax

    # Continuous model
    λ(x,d) = λ₀(d) + (λ₁(d) - λ₀(d)) * x

    # Advection in the discrete model
    Ã(d) = r₀₁(d)
    B̃(d) = λ̃₁(d) - λ̃₀(d) - r₀₁(d) - r₁₀(d)
    C̃(d) = λ̃₀(d) - λ̃₁(d)

    ṽ(x̃,d) = Ã(d) + B̃(d) * x̃ + C̃(d) * x̃^2

    # Advection in the continuous model
    v(x,d) = ṽ(pmin + (pmax - pmin) * x,d) / (pmax - pmin)

## Simulate both models (vary drug schedule)

    # Convert to growth and death rates
    μ = [d -> max(0,λ̃₀(d)), d -> max(0,λ̃₁(d))]
    δ = [d -> -min(0,λ̃₀(d)), d -> -min(0,λ̃₁(d))]
    r = [d -> r₀₁(d), d -> r₁₀(d)]

    # Heterogeneity
    β = 0.05

    # Continuous model initial condition
    ν = -ForwardDiff.derivative(x -> v(x,false),0.0)
    x₀ = Normal(0.0,β / sqrt(2ν))

    # Discrete model initial condition (stationary proportion with drug off)
    n₀ = (pmin,IC)
    
    function simulate_both_models(d,tmax,tstops=0:1:tmax)

        # Solve PDE for the continuous model
        xpde,_,ppde = solve_pde(λ,v,d,β;n₀=mean(IC),x₀,tmax,tstops,npt=51,xqt=1e-12);

        # Solve corresponding CMEs
        N_bin,Qcme_bin = solve_discrete_cme(μ,δ,r,d;n₀,Nmax=80,tmax,dt=0.01);
        N_con,Qcme_con = solve_cme(xpde,ppde,λ,d;n₀=IC,tmax,tstops=0:1:14);

        μ_bin(t) = sum(N_bin .* Qcme_bin(t))
        μ_con(t) = sum(N_con .* Qcme_con(t))

        return μ_bin, μ_con, N_bin, N_con, Qcme_bin, Qcme_con

    end

## Figure 8 — Row 1

    # Continuous application of drug (7 days)
    d = t -> true
    tmax = 7.0
    
    μ_bin, μ_con, N_bin, N_con, Qcme_bin, Qcme_con = simulate_both_models(d,tmax);

    # (a) Mean cell count
    fig8a = plot(μ_bin,xlim=(0.0,tmax),c=:red,lw=2.0,label="Discrete CME")
    plot!(fig8a,μ_con,xlim=(0.0,tmax),c=:black,ls=:dashdot,lw=2.0,label="Continuous CME")
    plot!(fig8a,widen=true,xticks=0:1:7,xlabel="Time [d]",ylabel="Mean cell count")

    # (b-d): CME
    fig8b = plot()
    plot!(fig8b,hist_coord(N_bin,Qcme_bin(1.0))...,frange=0.0,α=0.5,lw=0.0,c=:red,label="Discrete CME")
    plot!(fig8b,hist_coord(N_con,Qcme_con(1.0))...,c=:black,lw=2.0,ls=:dash,label="Continuous CME")
    plot!(fig8b,xlim=(0.0,20.0),ylim=(0.0,0.16))

    fig8c = plot()
    plot!(fig8c,hist_coord(N_bin,Qcme_bin(3.0))...,frange=0.0,α=0.5,lw=0.0,c=:red,label="Discrete CME")
    plot!(fig8c,hist_coord(N_con,Qcme_con(3.0))...,c=:black,lw=2.0,ls=:dash,label="Continuous CME")
    plot!(fig8c,xlim=(0.0,20.0),ylim=(0.0,0.16))

    fig8d = plot()
    plot!(fig8d,hist_coord(N_bin,Qcme_bin(7.0))...,frange=0.0,α=0.5,lw=0.0,c=:red,label="Discrete CME")
    plot!(fig8d,hist_coord(N_con,Qcme_con(7.0))...,c=:black,lw=2.0,ls=:dash,label="Continuous CME")
    plot!(fig8d,xlim=(0.0,40.0),ylim=(0.0,0.1))

    fig8_row1 = plot(fig8a,fig8b,fig8c,fig8d,xwiden=true,xlabel="Cell count",
        layout=grid(1,4),size=(800,180))
    plot!(fig8_row1,subplot=1,xlabel="Time [d]")
    add_plot_labels!(fig8_row1)
    savefig("fig8_row1.svg")

## Figure 8 — Row 2

    # Intermittant with various periods
    ω = [1.0,2.0,4.0]
    fig8e = [plot() for ωᵢ in ω]

    for (i,ωᵢ) in enumerate(ω)

        d = t -> mod(t,2ωᵢ) < ωᵢ ? true : false
        tmax = 14.0

        # Simulate
        μ_bin, μ_con, N_bin, N_con, Qcme_bin, Qcme_con = simulate_both_models(d,tmax);

        # Plot schedule
        W = 0:2ωᵢ:tmax
        for j in eachindex(W)
            plot!(fig8e[i],W[j] .+ [0.0,ωᵢ],[40.0,40.0],frange=0.0,c=:blue,α=0.5,lw=0.0,label=j == 1 ? "Drug on" : "")
        end

        # Plot model solutions
        plot!(fig8e[i],μ_bin,xlim=(0.0,tmax),c=:red,lw=2.0,label="Discrete CME")
        plot!(fig8e[i],μ_con,xlim=(0.0,tmax),c=:black,ls=:dashdot,lw=2.0,label="Continuous CME")
        plot!(fig8e[i],widen=true,xticks=0:1:14,xlabel="Time [d]",ylabel="Mean cell count")

    end

    fig8_row2 = plot(fig8e...,layout=grid(1,3),size=(800,180),widen=true,
        ylim=(5.0,22.0),xticks=0:2:14)
    add_plot_labels!(fig8_row2,offset=3)
    savefig("fig8_row2.svg")

## DEBUG LOSS OF MASS

plot(t -> sum(Qcme_bin(t)),xlim=(0.0,14.0))