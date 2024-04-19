### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 738d3a80-fd10-11ee-08b7-95b2dd27d0ba
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	using Reinforcement
end

# ╔═╡ 956f6fc6-0391-4f75-81fc-04ecebeb1674
using Plots

# ╔═╡ 901a615e-0cb8-42ff-993a-de1e77492df4
begin
	plot()
	for i in [1, 2, 3, 32]
		env = Gambler(100, 0.4)
		V = TabularValueFunction{Int64}()
	
		alg = ValueIteration(env, V)
		run!(Int64, alg, max_iters=i, θ=0.0)

		values = Float64[]
		for i in 1:99
			push!(values, get_value(V, i))
		end
		plot!(values, label="sweep $i")
	end
	plot!()
end

# ╔═╡ d7f6fb5f-f195-437e-9aa5-5bd1af47291d
begin
	subplots = []
	for i in [1, 2, 3, 32]
		env = Gambler(100, 0.3)
		V = TabularValueFunction{Int64}()
	
		alg = ValueIteration(env, V)
		run!(Int64, alg, max_iters=i, θ=0.0, tie_criterion=minimum)
		
		actions = Int64[]
		value_policy = extract_policy(Int64, V, env, tie_criterion=minimum)
		for i in 1:99
			push!(actions, select_action(value_policy, i))
		end

		push!(subplots, bar(actions, label="sweep $i"))
	end
	plot(subplots..., layout=(4, 1), size=(600, 800))
end

# ╔═╡ 19d228b0-b551-4dd4-a9a6-86701657793e
begin
	x = range(0, 10, length=100)
	y1 = @. exp(-0.1x) * cos(4x)
	y2 = @. exp(-0.3x) * cos(4x)
	y3 = @. exp(-0.5x) * cos(4x)
	[y1 y2 y3]
end

# ╔═╡ d917dd72-7301-45a0-8cd9-6a08f834f783
hcat(subplots...)

# ╔═╡ 9fc80996-e034-46f5-b895-493dfdd9fce3
begin
	plot()
	for i in [1, 2, 3, 32]
		env = Gambler(100, 0.4)
	
		policy = TabularPolicy{Int64, Int64}(1)
		policy_iter = PolicyIteration(env, policy)
	
		V = run!(policy_iter, θ=0.001, max_iters=i, max_eval_iters=i, tie_criterion=rand)

		values = Float64[]
		for i in 1:99
			push!(values, get_value(V, i))
		end
		plot!(values, label="sweep $i")
	end
	plot!()
end

# ╔═╡ ad23be33-a7c9-45bc-8ca0-fd06db80b4bb
begin
	subplots_policy = []
	for i in [1, 2, 3, 32]
		env = Gambler(100, 0.3)
		policy = TabularPolicy{Int64, Int64}(1)
		policy_iter = PolicyIteration(env, policy)
		run!(policy_iter, θ=0.0001, max_iters=i, max_eval_iters=i, tie_criterion=rand)
		
		actions = Int64[]
		for i in 1:99
			push!(actions, select_action(policy, i))
		end

		push!(subplots_policy, bar(actions, label="sweep $i"))
	end
	plot(subplots_policy..., layout=(4, 1), size=(600, 800))
end

# ╔═╡ Cell order:
# ╠═738d3a80-fd10-11ee-08b7-95b2dd27d0ba
# ╠═956f6fc6-0391-4f75-81fc-04ecebeb1674
# ╠═901a615e-0cb8-42ff-993a-de1e77492df4
# ╠═d7f6fb5f-f195-437e-9aa5-5bd1af47291d
# ╠═19d228b0-b551-4dd4-a9a6-86701657793e
# ╠═d917dd72-7301-45a0-8cd9-6a08f834f783
# ╠═9fc80996-e034-46f5-b895-493dfdd9fce3
# ╠═ad23be33-a7c9-45bc-8ca0-fd06db80b4bb
