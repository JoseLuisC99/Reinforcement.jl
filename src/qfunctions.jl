export QFunction, update!, merge!, get_Q_value, get_max_Q, extract_policy, QTable

abstract type QFunction{T, U} end

# """ Update the Q-value of (state, action) by delta """
# function update!(Q::QFunction{T, U}, state::T, action::U, Δ::Real)::Nothing where {T, U}
#     throw(ErrorException("Method not implemented on the Q function $(typeof(Q))"))
# end

# """ Get a Q value for a given state-action pair """
# function get_Q_value(Q::QFunction{T, U}, state::T, action::U)::Real where {T, U}
#     throw(ErrorException("Method not implemented on the Q function $(typeof(Q))"))
# end


""" Return a pair containing the action and Q-value, where the
    action has the maximum Q-value in state
"""
function get_max_Q(Q::QFunction{T, U}, state::T, actions::Vector{U}; tie_criterion::Function = rand)::Tuple{U, Real} where {T, U}
    max_q = -Inf
    arg_max_q = U[]

    for action ∈ actions
        value = get_Q_value(Q, state, action)

        
        if max_q == value
            push!(arg_max_q, action)
        elseif max_q < value
            arg_max_q = [action]
            max_q = value
        end
    end

    return (tie_criterion(arg_max_q), max_q)
end

""" Extract a policy for this Q-function  """
function extract_policy(Q::QFunction{T, U}, env::AbstractEnv)::AbstractPolicy{T} where {T, U}
    policy = TabularPolicy{T, U}()

    for state ∈ get_states(env)
        (action, _) = get_max_Q(Q, state, get_actions(env, state))
        update!(policy, state, action)
    end

    return policy
end


""" Tabular deterministic Q function """
struct QTable{T, U} <: QFunction{T, U}
    table::Dict{Tuple{T, U}, Real}
    default::Real

    QTable{T, U}() where {T, U} = new(Dict{Tuple{T, U}, Real}(), 0.0)
    QTable{T, U}(x::Real) where {T, U} = new(Dict{Tuple{T, U}, Real}(), x)
end

function update!(Q::QTable{T, U}, state::T, action::U, Δ::Real)::Nothing where {T, U}
    if haskey(Q.table, (state, action))
        Q.table[(state, action)] += Δ
    else
        Q.table[(state, action)] = Q.default + Δ
    end
    return
end

function get_Q_value(Q::QTable{T, U}, state::T, action::U)::Real where {T, U}
    return haskey(Q.table, (state, action)) ? Q.table[(state, action)] : Q.default
end