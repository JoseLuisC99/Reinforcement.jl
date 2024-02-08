export GridAction, GridState, GridTerminalState, GridWorld, Up, Right, Left, Down, Terminal,
    show_policy, show_value_function, maze
using Luxor, Printf

@enum GridAction Up Right Left Down Terminal
const GridState = Tuple{Int64, Int64}
const GridTerminalState = (0, 0)

struct GridWorld <: AbstractEnv
    ϵ::Float64
    width::Int64
    height::Int64
    γ::Float64
    action_cost::Float64
    blocked_states::Vector{GridState}
    initial_state::GridState
    goals::Dict{GridState, Real}

    GridWorld(width::Int64, height::Int64, blocked_states::Vector{GridState}, initial_state::GridState, goals::Dict{GridState, Real}) = new(
        0.1, width, height, 0.9, 0.0, blocked_states, initial_state, goals
    )
    GridWorld(width::Int64, height::Int64) = new(
        0.1, width, height, 0.9, 0.0,
        [(2, 2)], (1, 1), Dict(
            (width, height) => 1, (width, height - 1) => -1
        )
    )
    GridWorld() = GridWorld(4, 3)
end

function maze(n::Int64, m::Int64, start::Tuple{Int64, Int64})
    vector_maze, exit = maze_dfs(n, m, start)
    bloked_states = Vector{GridState}[]

    for i ∈ 1:n
        for j ∈ 1:m
            if vector_maze[i, j] == 1
                push!(bloked_states, (i, j))
            end
        end
    end

    return GridWorld(n, m, bloked_states, start, Dict(
        exit => 1
    ))
end

function get_states(env::GridWorld)::Vector{GridState}
    states = [GridTerminalState]
    for i in 1:env.width
        for j in 1:env.height
            if (i, j) ∉ env.blocked_states
                push!(states, (i, j))
            end
        end
    end

    return states
end

function get_actions(env::GridWorld, state::Union{GridState, Nothing})::Vector{GridAction}
    actions = [Up, Right, Down, Left, Terminal]

    if state === nothing
        return actions
    end

    valid_actions = GridAction[]
    for action ∈ actions
        for (_, prob) ∈ get_transitions(env, state, action)
            if prob > 0
                push!(valid_actions, action)
                break
            end
        end
    end

    return valid_actions
end

function valid_state(env::GridWorld, state::GridState, new_state::GridState, prob::Float64)::Union{Tuple{GridState, Float64}, Nothing}
    if prob == 0.0
        return nothing
    end

    if new_state ∈ env.blocked_states
        return (state, prob)
    end

    (x, y) = new_state
    if x > 0 && x <= env.width && y > 0 && y <= env.height
        return (new_state, prob)
    end 

    return (state, prob)
end

function push_if_not_nothing!(v, item)
    if item !== nothing
        push!(v, item)
    end
end

function get_transitions(env::GridWorld, state::GridState, action::GridAction)::Vector{Tuple{GridState, Float64}}
    transitions = Vector{Tuple{GridState, Real}}()

    if state == GridTerminalState
        if action == Terminal
            return [(state, 1.0)]
        else
            return []
        end
    end

    straight = 1 - 2 * env.ϵ
    (x, y) = state
    if haskey(env.goals, state)
        if action == Terminal
            push!(transitions, (GridTerminalState, 1.0))
        end
    elseif action == Up
        push_if_not_nothing!(transitions, valid_state(env, state, (x, y + 1), straight))
        push_if_not_nothing!(transitions, valid_state(env, state, (x - 1, y), env.ϵ))
        push_if_not_nothing!(transitions, valid_state(env, state, (x + 1, y), env.ϵ))
    elseif action == Down
        push_if_not_nothing!(transitions, valid_state(env, state, (x, y - 1), straight))
        push_if_not_nothing!(transitions, valid_state(env, state, (x - 1, y), env.ϵ))
        push_if_not_nothing!(transitions, valid_state(env, state, (x + 1, y), env.ϵ))
    elseif action == Right
        push_if_not_nothing!(transitions, valid_state(env, state, (x + 1, y), straight))
        push_if_not_nothing!(transitions, valid_state(env, state, (x, y - 1), env.ϵ))
        push_if_not_nothing!(transitions, valid_state(env, state, (x, y + 1), env.ϵ))
    elseif action == Left
        push_if_not_nothing!(transitions, valid_state(env, state, (x - 1, y), straight))
        push_if_not_nothing!(transitions, valid_state(env, state, (x, y - 1), env.ϵ))
        push_if_not_nothing!(transitions, valid_state(env, state, (x, y + 1), env.ϵ))
    end

    merged_probs = Dict()
    for (state, prob) in transitions
        if haskey(merged_probs, state)
            merged_probs[state] += prob
        else
            merged_probs[state] = prob
        end
    end

    transitions = []
    for (state, prob) in merged_probs
        push!(transitions, (state, prob))
    end

    return transitions
end

function get_reward(env::GridWorld, state::GridState, action::GridAction, next_state::GridState)::Float64
    reward = 0.0

    if haskey(env.goals, state) && next_state == GridTerminalState
        reward = env.goals[state]
    else
        reward = env.action_cost
    end

    return reward
end

is_terminal(env::GridWorld, state::GridState) = state == GridTerminalState
get_discount_factor(env::GridWorld) = env.γ
get_initial_state(env::GridWorld) = env.initial_state
get_goal_states(env::GridWorld) = env.goals

function draw_arrow(center::Point, dir::Union{GridAction, Nothing}; length::Float64 = 20.0)
	sethue("skyblue4")

	if dir === Up::GridAction
		arrow_begin = center + Point(0, length / 2)
		arrow_end = center - Point(0, length / 2)
		arrow(arrow_begin, arrow_end)
	elseif dir === Down::GridAction
		arrow_begin = center - Point(0, length / 2)
		arrow_end = center + Point(0, length / 2)
		arrow(arrow_begin, arrow_end)
	elseif dir === Left::GridAction
		arrow_begin = center + Point(length / 2, 0)
		arrow_end = center - Point(length / 2, 0)
		arrow(arrow_begin, arrow_end)
	elseif dir === Right::GridAction
		arrow_begin = center - Point(length / 2, 0)
		arrow_end = center + Point(length / 2, 0)
		arrow(arrow_begin, arrow_end)
	end
end

function plot_grid_world(grid_world::GridWorld; width::Float64 = 50.0, height::Float64 = 50.0)
    background("transparent")

    for goal in grid_world.goals
        goal[2] < 0 ? sethue("firebrick") : sethue("chartreuse3")
        (x, y) = goal[1]
        x, y = x - 1, grid_world.height - y
        box(x * width, y * height, width, height, :fill)
    end

    for block_state in grid_world.blocked_states
        sethue("gray50")
        (x, y) = block_state
        x, y = x - 1, grid_world.height - y
        box(x * width, y * height, width, height, :fill)
    end
	
    for i in 1:grid_world.width
        for j in 1:grid_world.height
            sethue("grey20")
            x, y = i - 1, grid_world.height - j
            box(x * width, y * height, width, height, :stroke)
        end
    end
end

function show_value_function(grid_world::GridWorld, V::AbstractValueFunc; width::Float64 = 50.0, height::Float64 = 50.0)
    plot_grid_world(grid_world, width = width, height = height)
	
    sethue("black")
    fontsize(height / 4)
    for i in 1:grid_world.width
        for j in 1:grid_world.height
            (i, j) ∈ grid_world.blocked_states && continue
            x, y = i - 1, grid_world.height - j
            str = @sprintf "%.2f" get_value(V, (i, j))
            text(str, Point(x *  width, y * height), halign=:center, valign=:middle)
        end
    end
end

function show_policy(grid_world::GridWorld, policy::AbstractPolicy; width::Float64 = 50.0, height::Float64 = 50.0)
    plot_grid_world(grid_world, width = width, height = height)

    sethue("black")
    fontsize(height / 4)
    for goal in grid_world.goals
        (x, y) = goal[1]
        x, y = x - 1, grid_world.height - y
        str = @sprintf "%.2f" goal[2]
        text(str, Point(x *  width, y * height), halign=:center, valign=:middle)
    end
	
    for i in 1:grid_world.width
        for j in 1:grid_world.height
            x, y = i - 1, grid_world.height - j
            draw_arrow(Point(x * width, y * height), select_action(policy, (i, j)))
        end
    end
end

function neighbors(maze::Matrix{Int64}, i::Int64, j::Int64)::Vector{Tuple{Int64, Int64}}
	n, m = size(maze)
	@assert 1 <= i <= n
	@assert 1 <= j <= m

	_neighbors = Tuple{Int64, Int64}[]
	for (x, y) ∈ shuffle([(-2, 0), (2, 0), (0, -2), (0, 2)])
		if 1 <= i + x <= n && 1 <= j + y <= m
			push!(_neighbors, (i + x, j + y))
		end
	end

	return _neighbors
end

function maze_dfs(n::Int64, m::Int64, start::Tuple{Int64, Int64})::Tuple{Matrix{Int64}, Tuple{Int64, Int64}}
	maze = ones(Int64, (n, m))
	visited = Dict{Tuple{Int64, Int64}, Bool}()
	stack = [start]
	maze[start...] = 0
	last = nothing

	while !isempty(stack)
		next = pop!(stack)
		for v ∈ neighbors(maze, next...)
			maze[v...] == 0 && continue
			maze[v...] = 0
			push!(stack, v)

			x1, y1 = next
			x2, y2 = v
			x, y = ([x1, y1] + [x2, y2]) .÷ 2
			maze[x, y] = 0
			last = v
		end
	end

	return maze, last
end