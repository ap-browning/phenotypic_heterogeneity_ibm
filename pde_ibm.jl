using Distributions
using DifferentialEquations

# On a fixed mesh
function solve_pde_std(λ,v,d,β;t₀=0.0,tmax=28.0,n₀=1,x₀=Uniform(0.0,1.0),npt=201,xlim=(-0.5,1.5))
    # Domain
    x = range(xlim...,npt)
    Δx = diff(x)[1]
    # Initial condition
    if isa(n₀,Distribution)
        u = mean(n₀) * pdf.(x₀,x)   # Really only interested in the dynamics of x from the PDE
    else
        u = n₀ * pdf.(x₀,x)
    end
    function pde!(du,u,p,t)
        #∇² = CenteredDifference(2,2,Δx,npt+1)
        #∂ₓ = CenteredDifference(1,2,Δx,npt+1)
        #bc = Dirichlet0BC(Float64)
        #du .= β^2 / 2 * (∇² * bc * u) - ∂ₓ * bc * (v.(x,d(t)) .* u) + λ.(x,d(t)) .* u
        du .= β^2 / 2 * ∇²(u,Δx) - ∂ₓ(v.(x,d(t)) .* u,Δx) + λ.(x,d(t)) .* u
    end
    # Solve
    sol = solve(ODEProblem(pde!,u,(t₀,tmax)))
    # Obtain integral as a function of t
    n = t -> sum(sol(t)[1:end-1] + sol(t)[2:end]) * Δx / 2
    # Obtain PDF as a function of t
    p = t -> sol(t) / n(t)
    return x,n,p
end 


# On a moving domain (far better numerical performance)
function solve_pde(λ::Function,v::Function,d::Function,β::Number;t₀=0.0,tmax=28.0,n₀=1,x₀=Normal(0.0,0.1),npt=101,xqt=1e-10,tstops=[],xlim=nothing)
    # Create grid
    y = range(quantile.(x₀,[xqt,1-xqt])...,npt) .- mean(x₀)
    #x = t -> x̄(t) .+ y
    Δx = Δy = diff(y)[1]
    # Initial condition
    if isa(n₀,Distribution)
        ũ₀ = [mean(x₀); mean(n₀) * pdf.(x₀,mean(x₀) .+ y)]
    else
        ũ₀ = [mean(x₀); n₀ * pdf.(x₀,mean(x₀) .+ y)]
    end
    # PDE
    function pde!(dg,g,p,t)
        x̄ = g[1]  # Centre of the moving domain
        ũ = @view g[2:end]
        #∇² = CenteredDifference(2,2,Δy,npt+1)
        #∂ₓ = CenteredDifference(1,2,Δy,npt+1)
        #bc = Dirichlet0BC(Float64)
        dg[1] = v(x̄,d(t))
        #dg[2:end] .= β^2 / 2 * (∇² * bc * ũ) - ∂ₓ * bc * (v.(x̄ .+ y,d(t)) .* ũ) + v(x̄,d(t)) * (∂ₓ * bc * ũ) + λ.(x̄ .+ y,d(t)) .* ũ
        dg[2:end] .= β^2 / 2 * ∇²(ũ,Δy) - ∂ₓ(v.(x̄ .+ y,d(t)) .* ũ,Δy) + v(x̄,d(t)) * ∂ₓ(ũ,Δy) + λ.(x̄ .+ y,d(t)) .* ũ
    end
    # Solve
    sol = solve(ODEProblem(pde!,ũ₀,(t₀,tmax));tstops)
    # Obtain integral as a function of t
    n = t -> sum(sol(t)[2:end-1] + sol(t)[3:end]) * Δx / 2
    # Obtain PDF as a function of t
    p = t -> sol(t)[2:end] / n(t)
    # Get moving domain
    x = t -> sol(t)[1] .+ y
    return x,n,p
end

# Derivative operators (Dirichlet boundary conditions)
function ∂ₓ(u,Δ)
    du = diff([0.0;u;0.0]) / Δ
    return (du[1:end-1] + du[2:end]) / 2
end
function ∇²(u,Δ)
    diff(diff([0.0;u;0.0])) / Δ^2
end