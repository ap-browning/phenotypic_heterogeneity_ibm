#==*=*=*=*=*=*=*
    
    FIGURE 4
    
=*=*=*=*=*=*=*=#

using Plots
using Distributions
using AdaptiveMCMC
using JLD2
using Optimization, OptimizationNLopt
using .Threads

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("defaults.jl")
include("../default_setup.jl")

# Setup problem
n₀ = 5.0
d = t -> true
tmax = 28.0

# Solve PDE to get intensity functions
function get_rates(β)
    x₀ = Normal(0.0,β / sqrt(2ν))
    xpde,_,ppde = solve_pde(λ,v,d,β;tmax,n₀,x₀,npt=1001,xlim=(-1.0,2.0));
    rbirth(t) = trap(ppde(t) .* max.(0.0,λ.(xpde(t),d(t))),xpde(t))
    rdeath(t) = -trap(ppde(t) .* min.(0.0,λ.(xpde(t),d(t))),xpde(t)) 
    return rbirth,rdeath
end

α = [0.1,0.2,0.5,1.0,2.0]
rate_fcns = [get_rates(β * αᵢ) for αᵢ in α];

fig4a = plot([r[1] for r in rate_fcns],xlim=(0.0,14.0),lw=2.0,palette=palette(:YlGn)[3:end],label=α')
plot!(fig4a,rate_fcns[4][1],lw=2.0,c=:black,ls=:dash,label="")
plot!(ylabel="Birth rate")

fig4b = plot([r[2] for r in rate_fcns],xlim=(0.0,14.0),lw=2.0,palette=palette(:YlOrRd)[3:end],label=α')
plot!(fig4b,rate_fcns[4][2],lw=2.0,c=:black,ls=:dash,label="")
plot!(ylabel="Death rate")

fig4 = plot(fig4a,fig4b,size=(600,220),xticks=0:2:14,widen=true,xlabel="Time [d]")
add_plot_labels!(fig4)
savefig("fig4.svg")
