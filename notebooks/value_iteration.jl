### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 71ddf18e-c504-11ee-077f-a972bc454518
begin
	import Pkg
	Pkg.activate(Base.current_project())
	Pkg.instantiate()

	using Reinforcement
	using Luxor, PlutoUI, Distributions, Printf
end

# ╔═╡ b221c7cd-6b18-4e20-9b28-e05cbc5ec47b
begin
	env = GridWorld()
	V = TabularValueFunction{GridState}()

	alg = ValueIteration(env, V)
	run!(GridAction, alg)
end

# ╔═╡ 6cc4a477-dbdb-4c94-a6aa-d84ebf87a016
policy = extract_policy(GridAction, V, env)

# ╔═╡ 094b94eb-297e-4ccf-972e-6adb4a595ba9
@draw begin 
	origin(Point(50, 50))
	show_policy(env, policy, width=100.0, height=100.0)
end 400 300

# ╔═╡ 99d57c1c-05e7-4140-ab2c-536b8e4a90ee
@draw begin 
	origin(Point(50, 50))
	show_V_function(env, V, width=100.0, height=100.0)
end 400 300

# ╔═╡ Cell order:
# ╠═71ddf18e-c504-11ee-077f-a972bc454518
# ╠═b221c7cd-6b18-4e20-9b28-e05cbc5ec47b
# ╠═6cc4a477-dbdb-4c94-a6aa-d84ebf87a016
# ╠═094b94eb-297e-4ccf-972e-6adb4a595ba9
# ╠═99d57c1c-05e7-4140-ab2c-536b8e4a90ee
