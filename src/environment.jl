export  AbstractEnv, get_states, get_actions, get_transitions, get_reward, is_terminal, 
    get_discount_factor, get_initial_state, get_goal_states

abstract type AbstractEnv end

# """ Return all states of this environment """
# function get_states(env::AbstractEnv)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end

# """ Return all actions with non-zero probability from this state """
# function get_actions(env::AbstractEnv, state)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end

# """ Return all non-zero probability transitions for this action
#     from this `state`, as a list of (`state`, `probability`) pairs
# """
# function get_transitions(env::AbstractEnv, state, action)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end

# """ Return the reward for transitioning from `state` to `next_state` via action
# """
# function get_reward(env::AbstractEnv, state, action, next_state)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end

# """ Return true if and only if `state` is a terminal state of this environment """
# function is_terminal(env::AbstractEnv, state)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end

# """ Return the discount factor for this environment """
# function get_discount_factor(env::AbstractEnv)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end

# """ Return the initial state of this environment """
# function get_initial_state(env::AbstractEnv)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end

# """ Return all goal states of this environment """
# function get_goal_states(env::AbstractEnv)
#     throw(ErrorException("Method not implemented on the environment $(typeof(env))"))
# end