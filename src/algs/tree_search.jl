export MonteCarloTreeSearch

node_id = 0

struct Node{T, U}
    id::Int64
    state::T
    action::U
    reward::Float64

    parent::Node
    childrens::Dict{U, Dict{Node, Float64}}

    env::AbstractEnv
    Q::QFunction
    ndbandit::MultiArmedBandit

    Node(state::T, action::U, reward::Float64, parent::Node, env::AbstractEnv, Q::QFunction, nbandit::MultiArmedBandit) where {T, U} = new(
        node_id += 1, state, action, reward, parent, Dict(), env, Q, nbandit
    )
    Node(state::T, action::U, reward::Float64, node::Node) where {T, U} = Node(
        state, action, node, reward, node.env, node.Q, node.ndbandit
    )
end

function is_expanded(node::Node)::Bool
    actions = get_actions(node.env, node.state)
    return length(actions) == length(node.childrens)
end

function get_child(node::Node{T, U}, action::U) where {T, U}
    (next_state, reward) = execute(node.env, node.state, action)
    for (child, _) ∈ node.childrens[action]
        if next_state == child.state
            return child
        end
    end

    new_child = Node(next_state, action, reward, node)
    for (outcome, prob) ∈ get_transitions(env, node.state, action)
        if outcome == next_state
            node.childrens[action] = [(new_child, prob)]
            return new_child
        end
    end
end

function select(node::Node)::Node
    if !is_expanded(node) || is_terminal(node.env, node.state)
        return node
    else
        actions = collect(keys(node.childrens))
        action = select(node.ndbandit, node.state, actions, node.Q)
        return select(get_child(node, action))
    end
end

struct MonteCarloTreeSearch
    env::AbstractEnv
    Q::QFunction
    ndbandit::MultiArmedBandit
end

# function run!(mcst::MonteCarloTreeSearch; iterations::Int64 = 100_000)
#     root_node = create_root_node(mcst)
    
#     for _ in 1:iterations
#         selected_node = select(root_node)
#     end
# end