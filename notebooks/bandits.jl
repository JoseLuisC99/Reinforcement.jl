### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ fdc8f97e-c50b-11ee-15d8-d54ccb91fc5b
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	using Reinforcement
	using Luxor, PlutoUI, Plots
end

# ╔═╡ 42e2e6ee-e35d-4218-9359-b28d04d8619d
function run_bandit(bandit::MultiArmedBandit; episode_length::Int64 = 500, drift::Bool = true)
	env = DummyEnv([0.1, 0.3, 0.7, 0.2, 0.1])
	times_selected = zeros(Int64, 5)
	qtable = QTable{DummyState, DummyAction}()
	
	rewards = Float64[]
	for step ∈ 1:episode_length
		if drift && step == episode_length / 2
			Reinforcement.drift(env, [0.5, 0.2, 0.0, 0.3, 0.3])
		end

		action = select(bandit, 1, get_actions(env, 1), qtable)
		reward = get_reward(env, 1, action, 1)
		push!(rewards, reward)
		times_selected[action] += 1

		update!(qtable, 1, action, 
			(reward - get_Q_value(qtable, 1, action)) / times_selected[action]
		)
	end

	return rewards
end

# ╔═╡ 66136a1b-0c04-4c65-a995-9ae9775601fd
function test_bandits(drift::Bool = false)
	plot()
	ϵ = 0.1
	α = 0.99
	τ = 1.0

	algs = [
		(EpsilonGreedy(ϵ), "Epsilon-Greedy"),
		(EpsilonDecreasing(ϵ, α), "Epsilon-Decreasing"),
		(SoftmaxBandit(τ), "Softmax Bandit"),
		(UpperConfidenceBounds{DummyAction}(), "UCB1")
	]

	for (alg, name) ∈ algs
		rewards = run_bandit(alg, drift=drift, episode_length=1000)
		rewards = accumulate((x, y) -> (1 - α) * y + α * x, rewards, dims=1)
		plot!(rewards, label=name)
	end
	plot!(title="Bandits algorithm")
end

# ╔═╡ 39791c57-f9ca-4ac8-b95a-c0b0887d8396
test_bandits()

# ╔═╡ 2ba8fd8f-221c-4356-8697-25188b4436ea
test_bandits(true)

# ╔═╡ Cell order:
# ╠═fdc8f97e-c50b-11ee-15d8-d54ccb91fc5b
# ╠═42e2e6ee-e35d-4218-9359-b28d04d8619d
# ╠═66136a1b-0c04-4c65-a995-9ae9775601fd
# ╠═39791c57-f9ca-4ac8-b95a-c0b0887d8396
# ╠═2ba8fd8f-221c-4356-8697-25188b4436ea
