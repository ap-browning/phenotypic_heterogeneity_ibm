using Distributions
using DifferentialEquations
using Interpolations

# Integration
function trap(y,x)
    Δx = x[2] - x[1]
    sum(y[1:end-1] + y[2:end]) * Δx / 2
end

function solve_cme(x::Function,p,λ,d;kwargs...)

    # Function to get birth and death rate at each time-step
    rbirth(t) = trap(p(t) .* max.(0.0,λ.(x(t),d(t))),x(t))
    rdeath(t) = -trap(p(t) .* min.(0.0,λ.(x(t),d(t))),x(t))   # Could interpolate to make quicker!

    solve_cme(rbirth,rdeath;kwargs...)

end
solve_cme(x::Number,args...;kwargs...) = solve_cme(t -> x,args...;kwargs...)
function solve_cme(rbirth,rdeath;n₀=1,Nmax=150,tmin=0.0,tmax=28.0,saveat=(),tstops=[])

        # Domain and initial condition
        N = 0:Nmax
        if isa(n₀,Distribution)
            q₀ = pdf.(n₀,N)
        else
            q₀ = zeros(size(N)); q₀[Int(n₀+1)] = 1.0
        end

        # Chemical master equation
        function rhs!(dq,q,p,t)
            # Implement boundaries using smart referencing
            Q(n) = 0 ≤ n ≤ Nmax ? q[n+1] : 0.0
            # Get event rates
            rb,rd = rbirth(t),rdeath(t)
            # Fill out RHS or CME
            for i = eachindex(q)
                n = i - 1
                dq[i] = (n-1) * rb * Q(n-1) +
                        (n+1) * rd * Q(n+1) -
                        n * (rb + rd) * Q(n)
            end
        end
    
        # Solve
        sol = solve(ODEProblem(rhs!,q₀,(tmin,tmax));saveat,tstops)
        
        return N,sol

end

function cme_quantiles(N,Q,T;p=[0.025,0.25,0.5,0.75,0.975])
    function get_quantiles(t)
        n̂ = [0; (N[1:end-1] + N[2:end]) / 2; N[end]; N[end]+1]
        F̂ = [0; cumsum(Q(t)); 1.0]
        Interpolations.deduplicate_knots!(F̂,move_knots=true)
        itp = linear_interpolation(F̂,n̂)
        itp.(p)
    end
    hcat([get_quantiles(t) for t in T]...)
end

function cme_quantile_plot(N,Q;c=:black)
    qnt = cme_quantiles(N,Q,Q.t;p=[0.025,0.25,0.5,0.75,0.975])
    plt = plot(Q.t,qnt[end,:],frange=qnt[1,:];c,α=0.1,lw=0.0,label="cme")
    plot!(plt,Q.t,qnt[end-1,:],frange=qnt[2,:];c,α=0.1,lw=0.0,label="")
end

function solve_cme(λ::Function,v::Function,d::Function,β::Number;n₀=1,Nmax=150,tmax=28.0,kwargs...)
    # Solve PDE
    xpde,npde,ppde = solve_pde(λ,v,d,β;kwargs...)
    # Solve CME
    return solve_cme(xpde,ppde,λ,d;n₀,tmax,Nmax)
end
