export AbstractValueFunc, update!, get_value, get_Q_value, extract_policy, TabularValueFunction

""" Every value function must be a subtype of `AbstractValueFunc` to have compatability with the 
    rest of the API. For every `ValueFunc <: AbstractValueFunc`, the follow functions must be defined:
    - `update!(V::ValueFunc{T}, state::T, value::Real)::Nothing where {T}`
    - `get_value(V::ValueFunc{T}, state::T)::Real where {T}`
    - `merge!(V::ValueFunc{T}, new_V::ValueFunc{T})::Nothing where {T}`
"""
abstract type AbstractValueFunc{T} end

function get_Q_value(V::AbstractValueFunc{T}, env::AbstractEnv, state::T, action::U)::Real where {T, U}
    Q_value = 0.0
    for (new_state, prob) in get_transitions(env, state, action)
        reward = get_reward(env, state, action, new_state)
        Q_value += prob * (reward + get_discount_factor(env) * get_value(V, new_state))
    end

    return Q_value
end

function extract_policy(U::DataType, V::AbstractValueFunc{T}, env::AbstractEnv; tie_criterion::Function = rand)::AbstractPolicy where {T}
    policy = TabularPolicy{T, U}()

    for state in get_states(env)
        max_q = -Inf
        argmax_q = []

        for action ∈ get_actions(env, state)
            q_value = get_Q_value(V, env, state, action)
            if q_value == max_q
                push!(argmax_q, action)
            elseif q_value > max_q
                argmax_q = [action]
                max_q = q_value
            end
        end

        update!(policy, state, tie_criterion(argmax_q))
    end

    return policy
end

struct TabularValueFunction{T} <: AbstractValueFunc{T}
    table::Dict{T, Real}
    default::Real

    TabularValueFunction{T}() where {T} = new{T}(Dict{T, Real}(), 0.0)
end

function update!(V::TabularValueFunction{T}, state::T, value::Number)::Real where {T}
    V.table[state] = value
end

function get_value(V::TabularValueFunction{T}, state::T)::Real where {T}
    return haskey(V.table, state) ? V.table[state] : V.default
end

function merge!(V::TabularValueFunction{T}, new_V::TabularValueFunction{T})::Nothing where {T}
    for state ∈ keys(new_V.table)
        update!(V, state, get_value(new_V, state))
    end
end