export AbstractPolicy, select_action, TabularPolicy, update!

abstract type AbstractPolicy{T, U} end

# function select_action(policy::AbstractPolicy{T, U}, state::T)::U where {T, U}
#     throw(ErrorException("Unimplemented method in policy $(typeof(policy))"))
# end

""" Deterministic tabular policy.
    This policy keeps a table that maps from each state to the action for that state
"""
struct TabularPolicy{T, U} <: AbstractPolicy{T, U}
    table::Dict{T, U}
    default_action::Union{U, Nothing}

    TabularPolicy{T, U}() where {T, U} = new(Dict{T, U}(), nothing)
end

""" State to Action table. If state is not present we return default action."""
function select_action(policy::TabularPolicy{T, U}, state::T)::Union{U, Nothing} where {T, U}
    return haskey(policy.table, state) ? policy.table[state] : policy.default_action
end

function update!(policy::TabularPolicy{T, U}, state::T, action::U)::Nothing where {T, U}
    policy.table[state] = action
    return
end