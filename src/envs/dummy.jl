export  DummyEnv, DummyAction, DummyState

const DummyAction = Int64
const DummyState = Int64

struct DummyEnv <: AbstractEnv
    actions::Vector{DummyAction}
    probs::Vector{Float64}

    DummyEnv(n::Int64) = new(collect(1:n), rand(n))
    DummyEnv(probs::Vector{Float64}) = new(collect(1:length(probs)), probs)
end

function get_states(env::DummyEnv)::Vector{DummyState}
    return env.actions
end

function get_actions(env::DummyEnv, state::DummyState)::Vector{DummyAction}
    return env.actions
end

function get_transitions(env::DummyEnv, state::DummyState, action::DummyAction)::Vector{Tuple{DummyState, Float64}}
    @assert action ∈ env.actions
    return [(action, 1.0)]
end

function get_reward(env::DummyEnv, state::DummyState, action::DummyAction, next_state::DummyState)::Float64
    return rand() < env.probs[action] ? 5 : 0
end

is_terminal(env::DummyEnv, state::DummyState)::Bool = false

get_discount_factor(env::DummyEnv)::Float64 = 0.0

get_initial_state(env::DummyEnv)::DummyState = 1

get_goal_states(env::DummyEnv)::Dict{DummyState, Real} = Dict{DummyState, Real}()

function drift(env::DummyEnv, probs::Vector{Float64})
    @assert size(env.probs) == size(probs)
    empty!(env.probs)
    append!(env.probs, probs)
end