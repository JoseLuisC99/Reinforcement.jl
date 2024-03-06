export  BanditsEnv, BanditsAction, BanditsState, drift, optimal_action, plot_bandits_testbed
using Random, Distributions, Plots, StatsPlots

const BanditsAction = Int64
const BanditsState = Int64

struct BanditsEnv <: AbstractEnv
    means::Vector{Float64}
end

function BanditsEnv(n::Int64; μ::Float64 = 0.0, σ::Float64 = 1.0)::BanditsEnv
    @assert n > 0
    dist = Normal(μ, σ)
    means = rand(dist, n)

    return BanditsEnv(means)
end

function get_states(env::BanditsEnv)::Vector{BanditsState}
    return [1]
end

function get_actions(env::BanditsEnv, state::BanditsState)::Vector{BanditsAction}
    return return collect(1:length(env.means))
end

function get_transitions(env::BanditsEnv, state::BanditsState, action::BanditsAction)::Vector{Tuple{BanditsState, Float64}}
    @assert action ∈ env.actions
    return [(action, 1.0)]
end

function get_reward(env::BanditsEnv, state::BanditsState, action::BanditsAction, next_state::BanditsState)::Float64
    μ = env.means[action]
    dist = Normal(μ, 1.0)
    return rand(dist, 1)[1]
end

is_terminal(env::BanditsEnv, state::BanditsState)::Bool = false

get_discount_factor(env::BanditsEnv)::Float64 = 0.0

get_initial_state(env::BanditsEnv)::BanditsState = 1

get_goal_states(env::BanditsEnv)::Dict{BanditsState, Real} = Dict{BanditsState, Real}()

function drift(env::BanditsEnv, means::Vector{Float64})
    @assert size(env.means) == size(means)
    empty!(env.means)
    append!(env.means, means)
end

function drift(env::BanditsEnv; μ::Float64 = 0.0, σ::Float64 = 1.0)
    dist = Normal(μ, σ)
    means = rand(dist, length(env.means))
    drift(env, means)
end

function optimal_action(env::BanditsEnv)::BanditsAction
    return argmax(env.means)
end

function plot_bandits_testbed(env::BanditsEnv; μ::Real = 0.0)
	len = length(env.means)
	plot(legend=false, xlabel="Action", ylabel="Reward", xticks=1:len)
	
	for mean ∈ env.means
		dist = Normal(mean, 1.0)
		sample = rand(dist, 1000)
		violin!(sample, side=:right, show=true, alpha=0.3)
	end

	len += 1
	plot!(1:len, zeros(len) .+ μ, ls=:dash, linewidth=3)
end