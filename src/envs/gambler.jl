export Gambler

struct Gambler <: AbstractEnv
    goal::Int64
    prob_head::Float64
end

# function Gambler(goal::Int64, prob_head::Float64)
#     @assert 0 <= prob_head <= 1
#     @assert goal > 0

#     return Gambler(goal, prob_head)
# end

function get_states(env::Gambler)::Vector{Int64}
    return collect(1:(env.goal - 1))
end

function get_actions(env::Gambler, state)::Vector{Int64}
    return collect(1:(min(state, env.goal - state)))
end

function get_transitions(env::Gambler, state::Int64, action::Int64)::Vector{Tuple{Int64, Float64}}
    if state == env.goal
        return [(state, 1.0)]
    else
        return [(state + action, env.prob_head), (state - action, 1 - env.prob_head)]
    end
end

function get_reward(env::Gambler, state::Int64, action::Int64, next_state::Int64)::Float64
    if next_state == env.goal
        return 1.0
    end
    return 0.0
end

is_terminal(env::Gambler, state)::Bool = state == env.goal || state < 1
get_discount_factor(env::Gambler)::Int64 = 1.0
get_initial_state(env::Gambler)::Int64 = 1
get_goal_states(env::Gambler)::Dict{Int64, Real} = Dict(env.goal => 1.0)