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
	using Luxor, PlutoUI, Plots, Statistics
end

# ╔═╡ 48f447ed-37c0-4b12-927f-fb7835fb9054
begin
	env = BanditsEnv(10)
	plot_bandits_testbed(env)
end

# ╔═╡ aa42128d-b939-4e5a-a3f5-b2d1b45b4471
md"""
## ϵ-greedy
"""


# ╔═╡ 42e2e6ee-e35d-4218-9359-b28d04d8619d
function run_bandit(bandit::MultiArmedBandit, env::BanditsEnv; episode_length::Int64 = 500, drift::Bool = false, optimistic_value::Real = 0.0)
	# println(optimal_action(env))
	times_selected = zeros(Int64, length(env.means))
	qtable = QTable{BanditsState, BanditsAction}(optimistic_value)
	
	rewards = Float64[]
	optimals = Float64[]
	optimals_count = 0
	
	for step ∈ 1:episode_length
		if drift && step == episode_length / 2
			Reinforcement.drift(env)
		end

		action = select(bandit, 1, get_actions(env, 1), qtable)
		reward = get_reward(env, 1, action, 1)

		if action == optimal_action(env)
			optimals_count += 1
		end
		push!(rewards, reward)
		push!(optimals, optimals_count / length(optimals))
		times_selected[action] += 1

		update!(qtable, 1, action, 
			(reward - get_Q_value(qtable, 1, action)) / times_selected[action]
		)
	end

	return rewards, optimals
end

# ╔═╡ d29012a6-a5c4-488e-a8e5-c2dbba873968
function average_performance(bandit::MultiArmedBandit; runs=2000, episode_length=1000, optimistic_value::Real = 0.0)
	average_reward = zeros(episode_length)
	average_optimals = zeros(episode_length)

	average = (cumulatives, vals, n) -> begin
		cumulatives = cumulatives .* (n - 1)
		cumulatives += vals
		cumulatives = cumulatives ./ n

		return cumulatives
	end

	# lk = Threads.SpinLock()
	for n ∈ 1:runs
		reset!(bandit)
		env = BanditsEnv(10)
		
		rewards, optimals = run_bandit(bandit, env, 
			episode_length=episode_length, optimistic_value=optimistic_value)

		average_reward = average(average_reward, rewards, n)
		average_optimals = average(average_optimals, optimals, n)
	end

	return average_reward, average_optimals
end

# ╔═╡ 46e3f97e-8e6e-4eab-816f-13b535f0986e
begin
	runs = 3_000
	episode_length = 1_000
	rewards1, optimals1 = average_performance(EpsilonGreedy(0.1), 
		runs=runs, episode_length=episode_length)
	rewards2, optimals2 = average_performance(EpsilonGreedy(0.01), 
		runs=runs, episode_length=episode_length)
	rewards3, optimals3 = average_performance(EpsilonGreedy(0.0), 
		runs=runs, episode_length=episode_length)
end

# ╔═╡ 26de948b-c168-4921-80e3-f40ee0c8851b
begin
	plot(xlabel="Steps", ylabel="Average reward")
	plot!(rewards1, label="ϵ = 0.1")
	plot!(rewards2, label="ϵ = 0.01")
	plot!(rewards3, label="ϵ = 0 (greedy)")
end

# ╔═╡ 5a78f147-b914-48c0-a036-05bfa4653a8d
begin
	plot(xlabel="Steps", ylabel="% Optimal action")
	plot!(optimals1, label="ϵ = 0.1")
	plot!(optimals2, label="ϵ = 0.01")
	plot!(optimals3, label="ϵ = 0 (greedy)")
end

# ╔═╡ 16b6111c-a6aa-42f0-acf4-83b07786a252
md"""
## Optimistic Initial Values
"""

# ╔═╡ 7a7789e2-ec16-4541-86f9-3b23aa1aadde
begin
	rewards4, optimals4 = average_performance(EpsilonGreedy(0.0), 
		runs=runs, episode_length=episode_length, optimistic_value=5.0)
	rewards5, optimals5 = average_performance(EpsilonGreedy(0.1), 
		runs=runs, episode_length=episode_length)
end

# ╔═╡ ac72359c-e65e-4240-b307-c408f714030b
begin
	plot(xlabel="Steps", ylabel="Average reward")
	plot!(rewards4, label="Q = 5, ϵ = 0.0")
	plot!(rewards5, label="Q = 0, ϵ = 0.1")
end

# ╔═╡ 2c210d18-0790-4bf1-bd54-44bee5b220c6
begin
	plot(xlabel="Steps", ylabel="% Optimal action")
	plot!(optimals4, label="Q = 5, ϵ = 0.0")
	plot!(optimals5, label="Q = 0, ϵ = 0.1")
end

# ╔═╡ 1eb1ef68-8947-4b37-94d8-cf3ef48915b8
md"""
## Upper-Confidence-Bound
"""

# ╔═╡ 65d4918e-51bc-4696-8faa-fc136c544671
begin
	rewards6, optimals6 = average_performance(
		UpperConfidenceBounds{BanditsAction}(2.0), 
		runs=runs, episode_length=episode_length)
	rewards7, optimals7 = average_performance(EpsilonGreedy(0.1), 
		runs=runs, episode_length=episode_length)
end

# ╔═╡ c94760c1-9270-4294-8409-025160cd3c17
begin
	plot(xlabel="Steps", ylabel="Average reward")
	plot!(rewards6, label="UCB c = 2.0")
	plot!(rewards7, label="ϵ-greedy ϵ = 0.1")
end

# ╔═╡ bcb24967-02c1-4081-9a41-84e47c17e31e
begin
	plot(xlabel="Steps", ylabel="% Optimal action")
	plot!(optimals4, label="UCB c = 2.0")
	plot!(optimals5, label="ϵ-greedy ϵ = 0.1")
end

# ╔═╡ d66e88dc-ad00-40eb-b5c3-ae3e2d89c8ea
md"""
## Exploration vs Exploitation
"""

# ╔═╡ 4e859718-d3a6-4b93-ab28-2e8ddd76ae36
function exploration_exploitation(alg, parameter_space::Vector{Float64}, optimistic_value::Bool = false)
	avg_rewards = Float64[]
	for param ∈ parameter_space
		rewards, _ = average_performance(
			optimistic_value ? alg(0.0) : alg(param), 
			runs=runs, 
			optimistic_value=optimistic_value ? param : 0.0)
		push!(avg_rewards, mean(rewards))
	end

	return avg_rewards
end

# ╔═╡ 03fb6601-6e02-4e14-9505-2ba9ad420289
begin
	parameter_space = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4]
	epsilon_greedy = exploration_exploitation(EpsilonGreedy, parameter_space[1:6])
	ucb = exploration_exploitation(
		UpperConfidenceBounds{BanditsAction}, parameter_space[4:10])
	optimistic = exploration_exploitation(EpsilonGreedy, parameter_space[6:10], true)
end

# ╔═╡ f561e0af-d59e-457e-9c07-4b6afe623215
begin
	labels = ["1/128", "1/64", "1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"]
	plot(xlabel="ϵ α c Q₀", ylabel="Average reward",
		xticks=(1:length(parameter_space), labels), legend=:bottomright)
	plot!(1:6, epsilon_greedy, label="ϵ-greedy")
	plot!(4:10, ucb, label="UCB")
	plot!(6:10, optimistic, label="Optimistic initialization")
end

# ╔═╡ Cell order:
# ╠═fdc8f97e-c50b-11ee-15d8-d54ccb91fc5b
# ╠═48f447ed-37c0-4b12-927f-fb7835fb9054
# ╟─aa42128d-b939-4e5a-a3f5-b2d1b45b4471
# ╠═42e2e6ee-e35d-4218-9359-b28d04d8619d
# ╠═d29012a6-a5c4-488e-a8e5-c2dbba873968
# ╠═46e3f97e-8e6e-4eab-816f-13b535f0986e
# ╠═26de948b-c168-4921-80e3-f40ee0c8851b
# ╠═5a78f147-b914-48c0-a036-05bfa4653a8d
# ╟─16b6111c-a6aa-42f0-acf4-83b07786a252
# ╠═7a7789e2-ec16-4541-86f9-3b23aa1aadde
# ╠═ac72359c-e65e-4240-b307-c408f714030b
# ╠═2c210d18-0790-4bf1-bd54-44bee5b220c6
# ╟─1eb1ef68-8947-4b37-94d8-cf3ef48915b8
# ╠═65d4918e-51bc-4696-8faa-fc136c544671
# ╠═c94760c1-9270-4294-8409-025160cd3c17
# ╠═bcb24967-02c1-4081-9a41-84e47c17e31e
# ╟─d66e88dc-ad00-40eb-b5c3-ae3e2d89c8ea
# ╠═4e859718-d3a6-4b93-ab28-2e8ddd76ae36
# ╠═03fb6601-6e02-4e14-9505-2ba9ad420289
# ╠═f561e0af-d59e-457e-9c07-4b6afe623215
