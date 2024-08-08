using DifferentialEquations

function solve_discrete_cme(λ,δ,r,d;n₀=[1,0],Nmax=100,tmax=28.0)

    # Domain and initial condition
    N = 0:Nmax
    if isa(n₀,Distribution)
        q₀ = [pdf(n₀,[i,j]) for i in N, j in N]
    else
        n₁₀,n₂₀ = n₀
        q₀ = zeros(length(N),length(N)); q₀[n₁₀+1,n₂₀+1] = 1.0
    end

    # Chemical master equation
    function rhs!(dq,q,p,t)

        # Implement boundaries using smart referencing
        Q(n₁,n₂) = ((0 ≤ n₁ ≤ Nmax) & (0 ≤ n₂ ≤ Nmax)) ? q[n₁+1,n₂+1] : 0.0
        # Fill out RHS or CME
        for i = eachindex(N), j = eachindex(N)
            n₁,n₂ = i - 1, j - 1
            λ₁,λ₂   = λ[1](d(t)),λ[2](d(t))
            δ₁,δ₂   = δ[1](d(t)),δ[2](d(t))
            r₁₂,r₂₁ = r[1](d(t)),r[2](d(t))

            dq[i,j] = (n₁ - 1) * λ₁ * Q(n₁ - 1,n₂) + 
                    (n₁ + 1) * δ₁ * Q(n₁ + 1,n₂) +
                    (n₂ - 1) * λ₂ * Q(n₁,n₂ - 1) + 
                    (n₂ + 1) * δ₂ * Q(n₁,n₂ + 1) + 
                    (n₁ + 1) * r₁₂ * Q(n₁ + 1,n₂ - 1) + 
                    (n₂ + 1) * r₂₁ * Q(n₁ - 1,n₂ + 1) - 
                    (n₁ * (λ₁ + δ₁) + n₂ * (λ₂ + δ₂) + r₁₂ * n₁ + r₂₁ * n₂) * Q(n₁,n₂)
        end
    end

    sol = solve(ODEProblem(rhs!,q₀,(0.0,tmax)),Heun(),dt=0.01)

    q = t -> begin
        Q = sol(t)
        [sum(Q[i,n+2-i] for i = 1:n+1) for n = 0:Nmax]
    end

    return N, q

end

