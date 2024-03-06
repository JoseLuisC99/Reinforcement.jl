export MultiArmedBandit, EpsilonGreedy, EpsilonDecreasing, SoftmaxBandit, UpperConfidenceBounds, select, reset!
using Random

""" A multi-armed bandit is defined by a set of random variables that we select iteratively, observing the reward 
    from the arm after each round, and adjusting the strategy each time. 

    The following methods must be implemented for every subtype of `MultiArmedBandit`:
        - select(nbandit<:MultiArmedBandit, state::T, actions::Vector{U}, qfunction::QFunction{T, U}): select 
          an action for this state given from a list given a Q-function
"""
abstract type MultiArmedBandit end

mutable struct EpsilonGreedy <: MultiArmedBandit
    ϵ::Float64

    EpsilonGreedy(ϵ) = new(ϵ)
    EpsilonGreedy() = new(0.1)
end

function select(nbandit::EpsilonGreedy, state::T, actions::Vector{U}, qfunction::QFunction{T, U})::U where {T, U}
    if rand() < nbandit.ϵ
        return rand(actions)
    end
    (argmax_q, _) = get_max_Q(qfunction, state, actions)
    return argmax_q
end

function reset!(nbandit::EpsilonGreedy)::Nothing
    return
end

struct EpsilonDecreasing <: MultiArmedBandit
    ϵ::Float64
    α::Float64
    ϵ_greedy::EpsilonGreedy

    EpsilonDecreasing(ϵ::Float64, α::Float64) = new(ϵ, α, EpsilonGreedy(ϵ))
end

function select(nbandit::EpsilonDecreasing, state::T, actions::Vector{U}, qfunction::QFunction{T, U})::U where {T, U}
    argmax_q = select(nbandit.ϵ_greedy, state, actions, qfunction)
    nbandit.ϵ_greedy.ϵ *= nbandit.α
    return argmax_q
end

function reset!(nbandit::EpsilonDecreasing)::Nothing
    nbandit.ϵ_greedy.ϵ = nbandit.ϵ
    return
end

struct SoftmaxBandit <: MultiArmedBandit
    τ::Float64

    SoftmaxBandit(τ::Float64) = new(τ)
    SoftmaxBandit() = new(1.0)
end

function select(nbandit::SoftmaxBandit, state::T, actions::Vector{U}, qfunction::QFunction{T, U})::U where {T, U}
    total = 0.0
    for action ∈ actions
        total += exp(get_Q_value(qfunction, state, action) / nbandit.τ)
    end

    cumulative_prob = 0.0
    r = rand()
    for action ∈ actions
        prob = exp(get_Q_value(qfunction, state, action) / nbandit.τ) / total
        if cumulative_prob <= r <= cumulative_prob + prob
            return action
        end
        cumulative_prob += prob
    end
end

function reset!(nbandit::SoftmaxBandit)::Nothing
    return
end

mutable struct UpperConfidenceBounds{U} <: MultiArmedBandit
    t::Int64
    c::Real
    N::Dict{U, Int64}

    UpperConfidenceBounds{U}() where {U}= new(0, 1.0, Dict{U, Int64}())
    UpperConfidenceBounds{U}(c::Real) where {U}= new(0, c, Dict{U, Int64}())
end

function select(nbandit::UpperConfidenceBounds{U}, state::T, actions::Vector{U}, qfunction::QFunction{T, U})::U where {T, U}
    for action ∈ actions
        if !haskey(nbandit.N, action)
            nbandit.N[action] = 1
            nbandit.t += 1
            return action
        end
    end

    max_actions = U[]
    max_value = -Inf

    for action ∈ actions
        value = get_Q_value(qfunction, state, action) + nbandit.c * sqrt(2 * log(nbandit.t) / nbandit.N[action])
        if value > max_value
            max_actions = [action]
            max_value = value
        elseif value == max_value
            push!(max_actions, action)
        end
    end

    selected_action = rand(max_actions)
    nbandit.N[selected_action] += 1
    nbandit.t += 1

    return selected_action
end

function reset!(nbandit::UpperConfidenceBounds)::Nothing
    nbandit.t = 0
    empty!(nbandit.N)
    return
end