export AbstractModelFreeLearner, init!, QLearning, SARSA
""" `AbstractModelFreeLearner{T}`
    The follow functions must be defined:
    - `get_environment(learner<:AbstractModelFreeLearner)::AbstractEnv`
    - `get_bandit_algorithm(learner<:AbstractModelFreeLearner)::MultiArmedBandit`
    - `get_Q_function(learner<:AbstractModelFreeLearner)::QFunction`
    - `value_function(learner<:AbstractModelFreeLearner, state)::Float64`
"""
abstract type AbstractModelFreeLearner end

function init!(learner::AbstractModelFreeLearner; α::Float64 = 0.1, max_episodes::Int64 = 100, max_episode_length::Union{Int64, Nothing} = nothing)
    env = get_environment(learner)
    nbandits = get_bandit_algorithm(learner)
    Q = get_Q_function(learner)

    for _ ∈ 1:max_episodes
        state = get_initial_state(env)
        actions = get_actions(env, state)
        action = select(nbandits, state, actions, Q)

        while !is_terminal(env, state)
            (next_state, reward) = execute(env, state, action)
            actions = get_actions(env, next_state)
            next_action = select(nbandits, next_state, actions, Q)
            q_value = get_Q_value(Q, state, action)
            
            next_state_value = value_function(learner, next_state, next_action)
            δ = α * (reward + get_discount_factor(env) * next_state_value - q_value)
            update!(Q, state, action, δ)

            state = next_state
            action = next_action

            if max_episode_length !== nothing
                max_episode_length -= 1
                if max_episode_length == 0
                    break
                end
            end
        end
    end
end

struct QLearning <: AbstractModelFreeLearner
    env::AbstractEnv
    nbandit::MultiArmedBandit
    Q::QFunction
end

get_environment(learner::QLearning)::AbstractEnv = learner.env
get_bandit_algorithm(learner::QLearning)::MultiArmedBandit = learner.nbandit
get_Q_function(learner::QLearning)::QFunction = learner.Q

function value_function(learner::QLearning, state::T, action::U)::Float64 where {T, U}
    (_, max_q_value) = get_max_Q(learner.Q, state, get_actions(learner.env, state))
    return max_q_value
end

struct SARSA <: AbstractModelFreeLearner
    env::AbstractEnv
    nbandit::MultiArmedBandit
    Q::QFunction
end

get_environment(learner::SARSA)::AbstractEnv = learner.env
get_bandit_algorithm(learner::SARSA)::MultiArmedBandit = learner.nbandit
get_Q_function(learner::SARSA)::QFunction = learner.Q

function value_function(learner::SARSA, state::T, action::U)::Float64 where {T, U}
    return get_Q_value(learner.Q, state, action)
end