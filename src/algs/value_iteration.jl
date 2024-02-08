export ValueIteration, run!

struct ValueIteration{T}
    env::AbstractEnv
    V::TabularValueFunction{T}
end

function run!(A::DataType, val_iteration::ValueIteration{T}; max_iters::Int64 = 100, θ::Float64 = 0.001) where {T}
    for i ∈ 1:max_iters
        Δ = 0.0
        new_values = TabularValueFunction{T}()

        for state ∈ get_states(val_iteration.env)
            qtable = QTable{T, A}()

            for action ∈ get_actions(val_iteration.env, state)
                new_value = 0.0
                for (next_state, prob) ∈ get_transitions(val_iteration.env, state, action)
                    reward = get_reward(val_iteration.env, state, action, next_state)
                    new_value += prob * (reward + (get_discount_factor(val_iteration.env) * get_value(val_iteration.V, next_state)))
                end
                update!(qtable, state, action, new_value)
            end

            (_, max_q) = get_max_Q(qtable, state, get_actions(val_iteration.env, state))
            Δ = max(Δ, abs(get_value(val_iteration.V, state) - max_q))
            update!(new_values, state, max_q)
        end

        merge!(val_iteration.V, new_values)

        if Δ < θ
            println("Break in $i iterations")
            break
        end
    end
end