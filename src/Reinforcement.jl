module Reinforcement

include("environment.jl")
include("policy.jl")
include("value.jl")
include("qfunctions.jl")

include("envs/gridworld.jl")
include("algs/value_iteration.jl")

end
