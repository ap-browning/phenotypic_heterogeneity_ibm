#==*=*=*=*=*=*=*
    
    FIGURE S1
    
    Additional panels from Fig. 2, looking at the full CME distribution.

        (a)
        (b)

=*=*=*=*=*=*=*=#

using Plots
using Distributions

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("defaults.jl")
include("../default_setup.jl")

## SETUP TREATMENT SCHEDULE(S)

    dcon(t) = true
    dint(t) = mod(t,14) .< 7.0
    tmax    = 28.0
    tstop   = sort([0:7:28; (7:7:28) .- 1e-3])

## SETUP INITIAL CONDITION
    
    # Initially, all in the drug naive stationary state
    x₀ = Normal(0.0,β / sqrt(2ν))
    n₀ = IC

## SOLVE SDE, PDE, CME

    rep = 1000   # Number of replicates for comparison plot

    # Results for continuous treatment
    @time ibm_con = [simulate_ibm(λ,v,dcon,β;n₀,x₀,tmax) for _ = 1:rep];
    ibm_con_stat  = [ibm_statistics(ibmᵢ...) for ibmᵢ in ibm_con];
    xpde_con,npde_con,ppde_con = solve_pde(λ,v,dcon,β;n₀,x₀,tmax);

    @time Ncme_con,Qcme_con = solve_cme(xpde_con,ppde_con,λ,dcon;n₀,tmax);

    # Results for intermittent treatment
    @time ibm_int = [simulate_ibm(λ,v,dint,β;n₀,x₀,tmax) for _ = 1:rep];
    ibm_int_stat  = [ibm_statistics(ibmᵢ...) for ibmᵢ in ibm_int];
    xpde_int,npde_int,ppde_int = solve_pde(λ,v,dint,β;n₀,x₀,tmax);

    @time Ncme_int,Qcme_int = solve_cme(xpde_int,ppde_int,λ,dint;n₀,tmax);

## PLOT 

    Ncon(t) = [stat[1](t) for stat in ibm_con_stat]
    Nint(t) = [stat[1](t) for stat in ibm_int_stat]

    # Empirical histograms
    Qibm_con(t) = [count(Ncon(t) .== n) for n in Ncme_con] / rep
    Qibm_int(t) = [count(Nint(t) .== n) for n in Ncme_int] / rep

    # Plot
    T = 0.0:7.0:28.0
    plt_con = [plot(title="t = $(Int(t))") for t in T]
    plt_int = [plot(title="t = $(Int(t))") for t in T]
    lims = [(0.0,20.0),(0.0,20.0),(0.0,50.0),(0.0,80.0),(0.0,150.0)]
    ymax1 = [0.2,0.2,0.1,0.05,0.02]
    ymax2 = [0.2,0.2,0.06,0.08,0.03]
    for (i,t) in enumerate(T)
        plot!(plt_con[i],hist_coord(Ncme_con,Qibm_con(t))...,xlim=lims[i],ylim=(0.0,ymax1[i]),yticks=[0.0,ymax1[i]],c=:blue,frange=0.0,lw=0.0,α=0.5,label="ibm")
        plot!(plt_con[i],hist_coord(Ncme_con,Qcme_con(t))...,xlim=lims[i],ylim=(0.0,ymax1[i]),yticks=[0.0,ymax1[i]],c=:black,ls=:dash,label="cme")
        plot!(plt_int[i],hist_coord(Ncme_int,Qibm_int(t))...,xlim=lims[i],ylim=(0.0,ymax2[i]),yticks=[0.0,ymax2[i]],c=:blue,frange=0.0,lw=0.0,α=0.5,label="")
        plot!(plt_int[i],hist_coord(Ncme_int,Qcme_int(t))...,xlim=lims[i],ylim=(0.0,ymax2[i]),yticks=[0.0,ymax2[i]],c=:black,ls=:dash,label="")
    end

    figS1 = plot(plt_con...,plt_int...,layout=grid(2,5),ywiden=false,xwiden=true,
        xlabel="Cell count", ylabel="Density", size=(800,400))
    savefig("figS1.svg")