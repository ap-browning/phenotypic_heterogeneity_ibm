using Statistics, StatsBase, Interpolations

function simulate_ibm(λ,δ,r,d;n₀=[1,0],tmax=28.0)

    # Birth rate function
    rbirth = (t,st) -> st > 0 ? λ[st](d(t)) : 0.0
    rdeath = (t,st) -> st > 0 ? δ[st](d(t)) : 0.0
    rswtch = (t,st) -> st > 0 ? r[st](d(t)) : 0.0

    # Codes: 1/2 state, 0 - dead
    if isa(n₀,Distribution)
        n̂₀ = rand(n₀)
        x = [fill(1,n̂₀[1]); fill(2,n̂₀[2])]
    else
        x = [fill(1,n₀[1]); fill(2,n₀[2])]
    end

    # Storage
    t = 0.0
    T = [t]
    N = n₀

    while t < tmax

        # Rates
        R = [rbirth.(t,x), rdeath.(t,x), rswtch.(t,x)]

        # Total event rate
        revent = [sum(y) for y = R]
        rtotal = sum(revent)

        # Time until next event
        t += rand(Exponential(1 / rtotal))

        # Which event?
        eidx = sample(eachindex(revent),Weights(revent))

        # Which cell?
        cidx = sample(eachindex(x),Weights(R[eidx]))

        # Action the event
        if eidx == 1
            # Birth
            x = [x; x[cidx]]
        elseif eidx == 2
            # Death
            x[cidx] = 0
        else 
            # Switch
            x[cidx] = x[cidx] == 1 ? 2 : 1
        end

        # Save population counts
        N = [N [sum(x .== c) for c = [1,2]]]
        T = [T;t]

    end

    if T[end] > tmax
        T[end] = tmax
    end

    return T,N

end

function interpolate_prev(x,y)
    idx(xᵢ) = xᵢ ≥ x[end] ? length(x) : (findfirst(x .> xᵢ) - 1)
    x -> y[idx(x)]
end

function ibm_statistics(T,N)
    N̄ = sum(N,dims=1)[:]
    n = interpolate_prev(T,N̄)
    p = interpolate_prev(T,N[2,:] ./ sum(N,dims=1)[:])
    return n,p
end