using Distributions
using Random

import Base: rand, minimum, maximum
import Distributions: pdf, cdf, kurtosis, logpdf, loglikelihood
import Statistics: mean, var, std, quantile
import StatsBase: skewness, sample, kurtosis

##############################################################
## Alternative parameterisation of the gamma distribution
##############################################################
# Due to numerically stability, returns a normal distribution if ω ≤ 1e-4
const ω_threshold = 1e-4
struct GammaAlt{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    ω::T
    θ::NamedTuple
    d::Gamma
end
GammaAlt(μ,σ,ω) = abs(ω) > ω_threshold ? GammaAlt(μ,σ,ω,gamma_moments_inv([μ,σ,ω]),Gamma(4 / ω^2)) : Normal(μ,σ)

function gamma_moments_inv(m)
    μ,σ,ω = m
    k = 4 / ω^2
    θ = σ * ω / 2
    s = μ - 2σ / ω
    return (k = k,θ = θ,s = s)
end

#### Evaluation
rand(rng::AbstractRNG, d::GammaAlt) = d.θ.θ * rand(rng,d.d) + d.θ.s
pdf(d::GammaAlt,x::Real) = pdf(d.d,(x - d.θ.s) / d.θ.θ) / abs(d.θ.θ)
logpdf(d::GammaAlt,x::Real) = logpdf(d.d,(x - d.θ.s) / d.θ.θ) - log(abs(d.θ.θ))
cdf(d::GammaAlt,x::Real) = d.θ.θ > 0.0 ? 
    cdf(d.d,(x - d.θ.s) / d.θ.θ) : 
    1 - cdf(d.d,(x - d.θ.s) / d.θ.θ)
quantile(d::GammaAlt,p::AbstractArray) = d.θ.θ > 0.0 ? 
    d.θ.θ * quantile(d.d,p) .+ d.θ.s : 
    d.θ.θ * quantile(d.d,1 .- p) .+ d.θ.s
quantile(d::GammaAlt,p::Number) = d.θ.θ > 0.0 ? 
    d.θ.θ * quantile(d.d,p) .+ d.θ.s : 
    d.θ.θ * quantile(d.d,1 - p) .+ d.θ.s
minimum(d::GammaAlt) = quantile(d,0.0)
maximum(d::GammaAlt) = quantile(d,1.0)

mean(d::Union{GammaAlt}) = d.μ
std(d::Union{GammaAlt}) = d.σ
var(d::Union{GammaAlt}) = std(d)^2
skewness(d::Union{GammaAlt}) = d.ω
kurtosis(d::Union{GammaAlt}) = 6 / d.θ.k