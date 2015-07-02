#Probit likelihood function

type Probit <: Likelihood end

function like(probit::Probit, y::Vector{Bool},f::Vector{Float64})
    y[y.==0]=-1
    return log(Φ(y.*f))
end

