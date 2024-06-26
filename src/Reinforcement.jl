module Reinforcement

include("environment.jl")
include("policy.jl")
include("value.jl")
include("qfunctions.jl")

include("envs/gridworld.jl")
include("envs/bandits_testbed.jl")
include("envs/gambler.jl")
include("algs/dp.jl")
include("algs/bandits.jl")
include("algs/model_free.jl")
# include("algs/tree_search.jl")

end