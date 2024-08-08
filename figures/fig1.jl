#==*=*=*=*=*=*=*
    
    FIGURE 1

=*=*=*=*=*=*=*=#

using Plots
using Interpolations

include("defaults.jl")

# Dose-response curve data from Hamis et al. 
λ₀ = [0.1374,0.1826, 0.1539, 0.2050, 0.1430, 0.2010,
      0.1240,0.0208,-0.2023,-0.4009,-0.7865,-0.6913]
λ₁ = [0.0707,0.0862, 0.1006, 0.1030, 0.1358, 0.1249,
      0.1766,0.1973, 0.1713,-0.0577,-0.2005,-0.1314] 

dose = [0.01,0.1,0.3,1,3,10,30,100,300,1000,3000,10000]
dosel = copy(dose); dosel[1] = 0.0
    # dose 0.01 corresponds to 0.0 (just for plotting)

# Calculate 500 uM dose
f₀ = d -> linear_interpolation(log.(dose),λ₀).(log.(d))
f₁ = d -> linear_interpolation(log.(dose),λ₁).(log.(d))

fig1a = hline([0.0],lw=2.0,ls=:dash,c=:black,α=0.5,label="")
plot!(fig1a,dose,λ₀,xaxis=:log,c=:blue,lw=2.0,label="sensitive")
plot!(fig1a,dose,λ₁,xaxis=:log,c=:red,lw=2.0,label="addicted")
scatter!([0.01,0.01],[λ₀[1],λ₁[1]],c=:black,label="")
scatter!([500.0,500.0],[f₀(500.0),f₁(500.0)],c=:black,label="")
plot!(fig1a,
    xticks=(dose,dosel),
    xrotation=45,
    yticks=-0.8:0.2:0.2,
    legend=:bottomleft,
    xlabel="Dose [nM]",
    ylabel="Net growth rate [d⁻¹]",
    size=(300,230)
)
savefig("fig1.svg")
