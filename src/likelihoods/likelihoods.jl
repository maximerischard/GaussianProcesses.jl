#This file contains a list of the currently available likelihood functions

import Base.show

abstract Likelihood

function show(io::IO, lik::Likelihood, depth::Int = 0)
    pad = repeat(" ", 2*depth)
    print(io, "$(pad)Type: $(typeof(lik)), Params: ")
  # show(io, get_params(lik))
    print(io, "\n")
end

include("logistic.jl")          # Logistic function

