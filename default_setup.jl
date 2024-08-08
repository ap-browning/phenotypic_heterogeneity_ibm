## SETUP MODEL PARAMETERS

    # Fitness "surface" (single drug level for now)
    γ₁ =  0.15
    γ₂ = -0.30
    γ₃ =  0.10
    γ₄ =  0.10

    λ(x,d) = d ? γ₂ + (γ₄ - γ₂) * x : γ₁ + (γ₃ - γ₁) * x

    # Advection "strength"
    ν = 0.4
    v(x,d) = -ν * (x - (d ? 1.0 : 0.0))

    # Diffusivity
    β = 0.05

    # Parameters (in a vector, on infinite scale)
    p = [γ₁,γ₂,γ₃,γ₄,log(ν),log(β)]

    # Prior
    prior = Product(Uniform.(
        [-1.0,-1.0,-1.0,-1.0,-6.0,-6.0],
        [ 1.0, 1.0, 1.0, 1.0, 1.0,-1.0]    
    ))

## SETUP INITIAL CONDITION

using Distributions
A₁ = π * 4500^2 # Area of a well (9mm diameter)
A₂ = 817 * 614  # Area of a FOV

IC = Binomial(1000, A₂ / A₁)

## Binomial noise
function add_noise(N,Q;α=0.1,n₀=5)
    function ℙy_I_n(n)
        n̄ = 2round((4α^2 * n^2 + n₀) / 2)
        Truncated(Binomial(n̄,0.5) - Int(n̄ / 2) + n,0,Inf)
    end
    max.(0.0,sum(pdf(ℙy_I_n(n),N) * Q[n+1] for n in N))
end

function add_noise(x;α=0.1,n₀=5)
    function ℙy_I_n(n)
        n̄ = Int(2round((4α^2 * n^2 + n₀) / 2))
        Truncated(Binomial(n̄,0.5) - Int(n̄ / 2) + n,0,Inf)
    end
    rand.(ℙy_I_n.(x))
end