using DifferentialEquations
using Distributions
using Plots

function simulate_ibm(λ,v,d,β;tmax=28.0,n₀=1,x₀=Uniform(0.0,1.0),Δt=0.01)

    # SDE for an agent path in phenotypic space
    f(x,p,t) = v(x,d(t))
    g(x,p,t) = β

    # Function to simulate an agent forward from a time point
    function simulate_path(t₀,x₀)
        prob = SDEProblem(f,g,x₀,(t₀,tmax))
        solve(prob,EM(),dt=Δt)
    end

    # Simulate birth and death events
    function get_birthdeath(x)
        X,T = x.u, x.t
        Λ = λ.(X,d.(T)) * Δt

        R = abs.(Λ)                 # Rates
        S = sign.(Λ)                # Events (i.e., birth or death)
        P = (1 .- exp.(-R))         # Probabilities
        I = findall(P .> rand(length(R)))   # Occurences

        # Find death (if any)
        death_idx = findfirst(S[I] .< 0.0)
        if death_idx === nothing
            tdeath = tmax
            tbirth = T[I]
        elseif death_idx == 1
            tdeath = T[I[1]]
            tbirth = []
        else
            tdeath = T[I[death_idx]]
            tbirth = T[I[1:death_idx-1]]
        end

        return tbirth,tdeath
    end

    # Simulate a cell
    function simulate_cell(t₀,x₀)
        x = simulate_path(t₀,x₀)
        tbirth,tdeath = get_birthdeath(x)
        return x,tbirth,tdeath
    end

    function simulate_population(x₀dist,n₀)
        t₀ = zeros(n₀)  # Array storing birth time of each cell
        t₁ = fill(NaN,n₀)   # NaN death time indicates simulation not run
        x₀ = rand(x₀dist,n₀)
        X  = Array{Any}(undef,n₀)
    
        while any(isnan.(t₁))
            i = findfirst(isnan.(t₁))
            x,tbirth,tdeath = simulate_cell(t₀[i],x₀[i])
            X[i] = x
            t₁[i] = tdeath
            t₀ = [t₀; tbirth]
            t₁ = [t₁; tbirth * NaN]
            x₀ = [x₀; x.(tbirth)]
            X  = [X; Array{Any}(undef,length(tbirth))]
        end
        return X,t₀,t₁
    
    end

    if isa(n₀,Distribution)
        return simulate_population(x₀,rand(n₀))
    else
        return simulate_population(x₀,n₀)
    end

end

# Function to get n(t), p(t)
function ibm_statistics(X,t₀,t₁)
    n = t -> count(t₀ .≤ t .≤ t₁)
    p = t -> [x(t) for x in X[findall(t₀ .≤ t .≤ t₁)]]
    return n,p
end

# Function to plot trace
function ibm_traceplot(X,t₀,t₁;Δt=0.1,kwargs...)
    plt = plot()
    hline!(plt,[0.0,1.0],ls=:dash,lw=2.0,α=0.5,c=:black,label="")
    for i = eachindex(X)
        tplt = range(t₀[i],t₁[i],1 + Int(ceil((t₁[i] - t₀[i]) / Δt)))
        plot!(plt,tplt,X[i].(tplt);label="",kwargs...)
        t₀[i] > 0.0 && scatter!(plt,[t₀[i]],[X[i](t₀[i])];kwargs...,α=1.0,msw=0.0,label="")
        t₁[i] < 28.0 && scatter!(plt,[t₁[i]],[X[i](t₁[i])];kwargs...,α=1.0,msw=0.0,shape=:square,label="")
    end
    return plt
end

# Diagnostic plot for v
function ibm_mean_diagnostic(v;t₁=10.0)
    ode1(x,p,t) = v(x,true)
    ode2(x,p,t) = v(x,false)
    sol1 = solve(ODEProblem(ode1,0.0,(0.0,t₁)))
    sol2 = solve(ODEProblem(ode2,1.0,(0.0,t₁)))
    plt = plot(sol1,xlim=(0.0,t₁),label="On Drug")
    plot!(plt,sol2,xlim=(0.0,t₁),label="Off Drug")
end