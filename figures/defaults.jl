## Default plot settinhgs

gr()
default()
default(
    fontfamily="Helvetica",
    tick_direction=:out,
    guidefontsize=9,
    annotationfontfamily="Helvetica",
    annotationfontsize=10,
    annotationhalign=:left,
    box=:on,
    msw=0.0,
    lw=1.5
)

alphabet = "abcdefghijklmnopqrstuvwxyz"

function add_plot_labels!(plt;offset=0)
    n = length(plt.subplots)
    for i = 1:n
        plot!(plt,subplot=i,title="($(alphabet[i+offset]))")
    end
    plot!(
        titlelocation = :left,
        titlefontsize = 10,
        titlefontfamily = "Helvetica"
    )
end

# Colours
col_prior1 = "#2d79ff"
col_prior2 = "#af52de"
col_posterior = "#f63a30"

    # Function to create the histogram
    function hist_coord(n,q;cutoff=1e-3 * maximum(q))
        # Trim (relative to maximum)
        idxmax = min(findlast(q .> maximum(q) * cutoff) + 1,length(n))
        n̂ = n[1:idxmax]; q̂ = q[1:idxmax]
        x = [n̂[1] .- 0.5; [n̂ .- 0.5 n̂ .+ 0.5]'[:]; n̂[end] .+ 0.5]
        y = [0; [q̂ q̂]'[:]; 0]
        return x,y
    end

function sample_noinf(prior,logpost)
    p = rand(prior)
    while isinf(logpost(p))
        p = rand(prior)
    end
    return p
end