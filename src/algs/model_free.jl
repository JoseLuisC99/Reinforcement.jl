export AbstractModelFreeLearner, NStepLearner, run!, QLearning, SARSA
""" `AbstractModelFreeLearner{T}`
    The follow functions must be defined:
    - `get_environment(learner<:AbstractModelFreeLearner)::AbstractEnv`
    - `get_bandit_algorithm(learner<:AbstractModelFreeLearner)::MultiArmedBandit`
    - `get_Q_function(learner<:AbstractModelFreeLearner)::QFunction`
    - `value_function(learner<:AbstractModelFreeLearner, state)::Float64`
"""
abstract type AbstractModelFreeLearner end

function run!(learner::AbstractModelFreeLearner; α::Float64 = 0.1, max_episodes::Int64 = 100, max_episode_length::Int64 = typemax(Int64))
    env = get_environment(learner)
    nbandits = get_bandit_algorithm(learner)
    Q = get_Q_function(learner)

    for _ ∈ 1:max_episodes
        state = get_initial_state(env)
        actions = get_actions(env, state)
        action = select(nbandits, state, actions, Q)
        episode_length = 0

        while !is_terminal(env, state) && episode_length < max_episode_length
            (next_state, reward) = execute(env, state, action)
            actions = get_actions(env, next_state)
            next_action = select(nbandits, next_state, actions, Q)
            q_value = get_Q_value(Q, state, action)
            
            next_state_value = value_function(learner, next_state, next_action)
            δ = α * (reward + get_discount_factor(env) * next_state_value - q_value)
            update!(Q, state, action, δ)

            state = next_state
            action = next_action

            episode_length += 1
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

struct NStepLearner
    td::AbstractModelFreeLearner
    n::Int64
end

function run!(learner::NStepLearner; α::Float64 = 0.1, max_episodes::Int64 = 100, max_episode_length::Int64 = typemax(Int64))
    env = get_environment(learner.td)
    nbandits = get_bandit_algorithm(learner.td)
    Q = get_Q_function(learner.td)

    for _ ∈ 1:max_episodes
        state = get_initial_state(env)
        actions = get_actions(env, state)
        action = select(nbandits, state, actions, Q)
        episode_length = 0

        rewards = Float64[]
        states = [state]
        actions = [action]
        next_state, next_action = nothing, nothing

        while length(states) > 0 && episode_length < max_episode_length
            if !is_terminal(env, state)
                (next_state, reward) = execute(env, state, action)
                push!(rewards, reward)

                if !is_terminal(env, next_state)
                    next_action = select(nbandits, next_state, get_actions(env, next_state), Q)
                    push!(states, next_state)
                    push!(actions, next_action)
                end
            end

            if length(rewards) == learner.n || is_terminal(env, state)
                nstep_reward = sum([
                    get_discount_factor(env) ^ i * rewards[i] for i ∈ eachindex(rewards)
                ])

                if !is_terminal(env, state)
                    next_value = value_function(learner.td, next_state, next_action)
                    nstep_reward += get_discount_factor(env) ^ learner.n * next_value
                end

                q_value = get_Q_value(Q, states[1], actions[1])
                update!(Q, states[1], actions[1], α * (nstep_reward - q_value))

                popat!(rewards, 1)
                popat!(states, 1)
                popat!(actions, 1)
            end
            episode_length += 1
            state = next_state
            action = next_action
        end
    end
end