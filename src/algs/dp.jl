export ValueIteration, PolicyIteration, run!

struct ValueIteration{T}
    env::AbstractEnv
    V::TabularValueFunction{T}
end

function run!(A::DataType, val_iteration::ValueIteration{T}; max_iters::Int64 = 100, θ::Float64 = 0.001, tie_criterion::Function = rand) where {T}
    for i ∈ 1:max_iters
        Δ = 0.0
        qtable = QTable{T, A}()

        for state ∈ get_states(val_iteration.env)
            for action ∈ get_actions(val_iteration.env, state)
                new_value = 0.0
                for (next_state, prob) ∈ get_transitions(val_iteration.env, state, action)
                    reward = get_reward(val_iteration.env, state, action, next_state)
                    new_value += prob * (reward + (get_discount_factor(val_iteration.env) * get_value(val_iteration.V, next_state)))
                end
                update!(qtable, state, action, new_value)
            end

            (_, max_q) = get_max_Q(qtable, state, get_actions(val_iteration.env, state), tie_criterion=tie_criterion)
            Δ = max(Δ, abs(get_value(val_iteration.V, state) - max_q))
            update!(val_iteration.V, state, max_q)
        end

        if Δ < θ
            println("Break in $i iterations")
            break
        end
    end
end

struct PolicyIteration{T, U}
    env::AbstractEnv
    policy::AbstractPolicy{T, U}
end

function run!(
        policy_iter::PolicyIteration{T, U};
        max_iters::Int64 = 0, 
        max_eval_iters = 1000,
        θ::Float64 = 0.001, 
        tie_criterion::Function = rand
    ) where {T, U}

    v = TabularValueFunction{T}()
    stable = false

    current_iter = 0
    while !stable
        policy_evaluation(policy_iter, v, max_eval_iters, θ)
        stable = policy_improvement(policy_iter, v, θ, tie_criterion)

        current_iter += 1
        if current_iter > 0 && current_iter == max_iters
            break
        end
    end

	return v
end

function policy_evaluation(policy_iter::PolicyIteration, value_func::TabularValueFunction, max_iters::Int64, θ::Float64)
    for i ∈ 1:max_iters
        Δ = 0.0

        for state ∈ get_states(policy_iter.env)
            new_value = 0.0
            action = select_action(policy_iter.policy, state)
            for (next_state, prob) ∈ get_transitions(policy_iter.env, state, action)
                reward = get_reward(policy_iter.env, state, action, next_state)
                new_value += prob * (reward + (get_discount_factor(policy_iter.env) * get_value(value_func, next_state)))
            end

            Δ = max(Δ, abs(get_value(value_func, state) - new_value))
            update!(value_func, state, new_value)
        end

        if Δ < θ
            println("Break in $i iterations")
            break
        end
    end
end

function policy_improvement(policy_iter::PolicyIteration{T, U}, value_func::TabularValueFunction, θ::Float64, tie_criterion::Function = rand) where {T, U}
    stable = true
    qtable = QTable{T, U}()

    for state ∈ get_states(policy_iter.env)
        for action ∈ get_actions(policy_iter.env, state)
            new_value = 0.0
            for (next_state, prob) ∈ get_transitions(policy_iter.env, state, action)
                reward = get_reward(policy_iter.env, state, action, next_state)
                new_value += prob * (reward + (get_discount_factor(policy_iter.env) * get_value(value_func, next_state)))
            end
            update!(qtable, state, action, new_value)
        end

        (argmax_q, max_q) = get_max_Q(qtable, state, get_actions(policy_iter.env, state), tie_criterion=tie_criterion)
        
        if select_action(policy_iter.policy, state) != argmax_q && abs(max_q - get_value(value_func, state)) < θ
            update!(policy_iter.policy, state, argmax_q)
            stable = false
        end
    end

    return stable
end
