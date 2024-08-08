#==*=*=*=*=*=*=*
    
    FIGURE 2

=*=*=*=*=*=*=*=#

using Plots, StatsPlots
using Distributions

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("../defaults.jl")
include("../default_setup.jl")

## SETUP TREATMENT SCHEDULE(S)

    dcon(t) = true
    dint(t) = mod(t,14) .< 7.0
    tmax    = 28.0
    tstops  = sort([0:7:28; (7:7:28) .- 1e-3])

## SETUP INITIAL CONDITION

    # Initially, all in the drug naive stationary state
    x₀ = Normal(0.0,β / sqrt(2ν))
    n₀ = IC

## SOLVE SDE, PDE, CME

    rep = 10    # Number of replicates for population plot
    n̄₀ = 500   # Number of cells for PDE comparison plot 

    # Results for continuous treatment
    @time ibm_con = [simulate_ibm(λ,v,dcon,β;n₀,x₀,tmax) for _ = 1:rep];
    ibm_con_stat  = [ibm_statistics(ibmᵢ...) for ibmᵢ in ibm_con];
    xpde_con,npde_con,ppde_con = solve_pde(λ,v,dcon,β;n₀,x₀,tmax);

    @time n̄ibm_con,x̄ibm_con = ibm_statistics(simulate_ibm(λ,v,dcon,β;n₀=n̄₀,x₀,tmax)...);
    @time x̄pde_con,n̄pde_con,p̄pde_con = solve_pde(λ,v,dcon,β;n₀=n̄₀,x₀,tmax,tstops);

    @time Ncme_con,Qcme_con = solve_cme(xpde_con,ppde_con,λ,dcon;n₀,tmax);

    # Results for intermittent treatment
    @time ibm_int = [simulate_ibm(λ,v,dint,β;n₀,x₀,tmax) for _ = 1:rep];
    ibm_int_stat  = [ibm_statistics(ibmᵢ...) for ibmᵢ in ibm_int];
    xpde_int,npde_int,ppde_int = solve_pde(λ,v,dint,β;n₀,x₀,tmax);

    @time n̄ibm_int,x̄ibm_int = ibm_statistics(simulate_ibm(λ,v,dint,β;n₀=n̄₀,x₀,tmax)...);
    @time x̄pde_int,n̄pde_int,p̄pde_int = solve_pde(λ,v,dint,β;n₀=n̄₀,x₀,tmax,tstops);   

    @time Ncme_int,Qcme_int = solve_cme(xpde_int,ppde_int,λ,dint;n₀,tmax);

## PLOT 

    # Trace plots...
    fig2a = ibm_traceplot(rand(ibm_con)...;c=:blue,mc=:black,ms=5,α=0.5,lw=2.0)
    plot!(fig2a,ylim=(-0.25,1.25),yticks=-0.2:0.2:1.2,xticks=0:7:28)

    fig2b = ibm_traceplot(rand(ibm_int)...;c=:blue,mc=:black,ms=5,α=0.2,lw=2.0)
    plot!(fig2b,ylim=(-0.25,1.25),yticks=-0.2:0.2:1.2,xticks=0:7:28)

    # Cell count plots
    fig2c = cme_quantile_plot(Ncme_con,Qcme_con)
    plot!(fig2c,[stat[1] for stat in ibm_con_stat],xlim=(0.0,28.0),label=["ibm" fill("",rep-1)...],c=:blue,α=0.4)
    plot!(fig2c,npde_con,c=:black,lw=2.0,label="pde",ylim=(0,50))

    fig2d = cme_quantile_plot(Ncme_int,Qcme_int)
    plot!(fig2d,[stat[1] for stat in ibm_int_stat],xlim=(0.0,28.0),label=["ibm" fill("",rep-1)...],c=:blue,α=0.4)
    plot!(fig2d,npde_int,c=:black,lw=2.0,label="pde",ylim=(0,50))

    plot(fig2a,fig2b,fig2c,fig2d,layout=grid(1,4))

    # PDE comparison
    fig2e = plot(); fig2f = plot()
    for (i,t) = enumerate([0.0,1.0,2.0,4.0,7.0])
        # Continuous treatment (for t ≤ 7; identical to intermittent)
        density!(fig2e,x̄ibm_con(t),lw=0.0,label="t = $(Int(t))",c=palette(:GnBu_8)[i+1],frange=0.0)
        idx = p̄pde_con(t) .> 1e-3
        plot!(fig2e,x̄pde_con(t)[idx],p̄pde_con(t)[idx],c=:black,ls=:dash,lw=2.0,label="")

        # Intermittent treatment (for t ≥ 7)
        density!(fig2f,x̄ibm_int(t+7),lw=0.0,label="t = $(Int(t+7))",c=palette(:RdPu_8)[i+1],frange=0.0)
        idx = p̄pde_int(t+7) .> 1e-3
        plot!(fig2f,x̄pde_int(t+7)[idx],p̄pde_int(t+7)[idx],c=:black,ls=:dash,lw=2.0,label="")
    end

    # Figure 2
    [plot!(fig,xlabel="Time [d]",xticks=0:7:28,widen=true) for fig = [fig2a,fig2b,fig2c,fig2d]]
    plot!(fig2a,ylabel="Phenotype")
    plot!(fig2c,ylabel="Cell count")
    [plot!(fig,xlabel="Phenotype",ylim=(0,9),widen=true) for fig = [fig2e,fig2f]]
    plot!(fig2e,ylabel="Density")

    fig2 = plot(fig2a,fig2b,fig2c,fig2d,fig2e,fig2f,
        layout=@layout([a b c d; e f]),size=(800,350))

    add_plot_labels!(fig2)
    savefig(fig2,"fig2.svg")
