#==*=*=*=*=*=*=*
    
    FIGURE S4

    Demonstrate the noise process

=*=*=*=*=*=*=*=#

using Plots, StatsPlots
using Distributions
using ColorSchemes

include("../pde_ibm.jl")
include("../sde_ibm.jl")
include("../cme_ibm.jl")
include("defaults.jl")
include("../default_setup.jl")

## SETUP EXPERIMENTAL DESIGN

    n₀ = IC     # Number of cells in window 
    R  = 4*192  # Number of experimental replicates     
    T  = 3.0
    x₀ = Normal(0.0,β / sqrt(2ν))
    d  = t -> true

## SOLVE CME TO GET EXACT DISTRIBUTION

@time x,n,p = solve_pde(λ,v,d,β;n₀,x₀,tmax=T);
@time N,Q = solve_cme(x,p,λ,d;n₀,tmax=T);

## (a) : noise for various different values of n

function ℙe_I_n(n;α=0.1,n₀=5)
    n̄ = 2round((4α^2 * n^2 + n₀) / 2)
    Binomial(n̄,0.5) - Int(n̄ / 2)
end

e = -15.0:15.0
n = [5,20,50]

figS4a = plot(palette=reverse(palette(:Set2_3)))
for i = reverse(eachindex(n))
    d = pdf.(ℙe_I_n(n[i]),e)
    plot!(figS4a,hist_coord(e,d)...,frange=0.0,α=0.8,label="n = $(n[i])",lw=0.0)
end
plot!(figS4a,ywiden=false,xlabel="Error",ylabel="Mass")

## (b) : exact vs noisy distribution

figS4b = plot()
plot!(figS4b,hist_coord(N,Q(T))...,c=:black,frange=0.0,fα=0.2,lw=0.0,label="Precise")
plot!(figS4b,hist_coord(N,add_noise(N,Q(T)))...,c=:red,frange=0.0,fα=0.3,lw=0.0,label="Noisy")
plot!(figS4b,ywiden=false,xlabel="Cell count",ylabel="Mass")

## Figure S4
figS4 = plot(figS4a,figS4b,size=(600,220))
add_plot_labels!(figS4)
savefig("figS4.svg")
figS4