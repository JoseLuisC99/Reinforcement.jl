### A Pluto.jl notebook ###
# v0.19.38

using Markdown
using InteractiveUtils

# ╔═╡ c8b9478e-c5f6-11ee-1547-4dc156fe4847
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	using Reinforcement
	using Luxor, PlutoUI, Printf
end

# ╔═╡ 4347d389-6e6d-4ee0-8a79-20ba0992f90b
begin
	env = GridWorld()
	nbandit = EpsilonGreedy()
	qfunction = QTable{GridState, GridAction}()

	alg = SARSA(env, nbandit, qfunction)
	init!(alg)
end

# ╔═╡ 9f40301b-cd95-45d8-9ed6-a1a1b983d91a
@draw begin 
	origin(Point(50, 50))
	show_Q_function(env, qfunction, width=100, height=100)
end 400 300

# ╔═╡ d35d9214-3884-43a1-869f-a83d032be55f
@draw begin 
	origin(Point(50, 50))
	policy = extract_policy(qfunction, env)
	show_policy(env, policy, width=100, height=100)
end 400 300

# ╔═╡ Cell order:
# ╠═c8b9478e-c5f6-11ee-1547-4dc156fe4847
# ╠═4347d389-6e6d-4ee0-8a79-20ba0992f90b
# ╠═9f40301b-cd95-45d8-9ed6-a1a1b983d91a
# ╠═d35d9214-3884-43a1-869f-a83d032be55f
