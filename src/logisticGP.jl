import Base.show

# Main GaussianProcess type

@doc """
# Description
Fits a Gaussian process to a set of training points. The Gaussian process is defined in terms of its mean and covaiance (kernel) functions, which are user defined. As a default it is assumed that the observations are noise free.
# Arguments:
* `x::Matrix{Float64}`: Training inputs
* `y::Vector{Float64}`: Observations
* `m::Mean`           : Mean function
* `k::kernel`         : Covariance function
* `logNoise::Float64` : Log of the observation noise. The default is -1e8, which is equivalent to assuming no observation noise.
# Returns:
* `gp::GP`            : A Gaussian process fitted to the training data
""" ->
type logisticGP
    x::Matrix{Float64}      # Input observations  - each column is an observation
    y::Vector{Float64}      # Output observations
    dim::Int                # Dimension of inputs
    nobsv::Int              # Number of observations
    m:: Mean                # Mean object
    k::Kernel               # Kernel object
    # Auxiliary data
    cK::AbstractPDMat       # (k + obsNoise)
    alpha::Vector{Float64}  # (k + obsNoise)⁻¹y
    mLL::Float64            # Marginal log-likelihood
    dmLL::Vector{Float64}   # Gradient marginal log-likelihood
     function logisticGP(x::Matrix{Float64}, y::Vector{Float64}, m::Mean, k::Kernel)
        dim, nobsv = size(x)
        length(y) == nobsv || throw(ArgumentError("Input and output observations must have consistent dimensions."))
        gp = new(x, y, dim, nobsv, m, k)
        update_mll!(gp)
        return gp
   end
end

# Creates GP object for 1D case
logisticGP(x::Vector{Float64}, y::Vector{Float64}, meanf::Mean, kernel::Kernel, logNoise::Float64=-1e8) = GP(x', y, meanf, kernel, logNoise)

# Update auxiliarly data in GP object after changes have been made
function update_mll!(gp::logisticGP)
        laplace!(gp)
end

# Update gradient of marginal log likelihood
## function update_mll_and_dmll!(gp::GP; noise::Bool=true, mean::Bool=true, kern::Bool=true)
## end


## @doc """
## # Description
## Calculates the posterior mean and variance of Gaussian Process at specified points
## # Arguments:
## * `gp::GP`: Gaussian Process object
## * `x::Matrix{Float64}`:  matrix of points for which one would would like to predict the value of the process.
##                        (each column of the matrix is a point)
## # Returns:
## * `(mu, Sigma)::(Vector{Float64}, Vector{Float64})`: respectively the posterior mean  and variances of the posterior
##                                                     process at the specified points
## """ ->
## function predict(gp::GP, x::Matrix{Float64}; full_cov::Bool=false)
##     size(x,1) == gp.dim || throw(ArgumentError("Gaussian Process object and input observations do not have consistent dimensions"))
##     if full_cov
##         return _predict(gp, x)
##     else
##         ## calculate prediction for each point independently
##             mu = Array(Float64, size(x,2))
##             Sigma = similar(mu)
##         for k in 1:size(x,2)
##             out = _predict(gp, x[:,k:k])
##             mu[k] = out[1][1]
##             Sigma[k] = out[2][1]
##         end
##         return mu, Sigma
##     end
## end

## # 1D Case for prediction
## predict(gp::GP, x::Vector{Float64};full_cov::Bool=false) = predict(gp, x'; full_cov=full_cov)

## ## compute predictions
## function _predict(gp::GP, x::Array{Float64})
##     cK = crossKern(x,gp.x,gp.k)
##     Lck = whiten(gp.cK, cK')
##     mu = meanf(gp.m,x) + cK*gp.alpha    # Predictive mean
##     Sigma = crossKern(x,gp.k) - Lck'Lck # Predictive covariance
##     Sigma = max(Sigma,0)
##     return (mu, Sigma)
## end


function get_params(gp::logisticGP; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    params = Float64[]
    if noise; push!(params, gp.logNoise); end
    if mean;  append!(params, get_params(gp.m)); end
    if kern; append!(params, get_params(gp.k)); end
    return params
end

function set_params!(gp::logisticGP, hyp::Vector{Float64}; noise::Bool=true, mean::Bool=true, kern::Bool=true)
    # println("mean=$(mean)")
    if noise; gp.logNoise = hyp[1]; end
    if mean; set_params!(gp.m, hyp[1+noise:noise+num_params(gp.m)]); end
    if kern; set_params!(gp.k, hyp[end-num_params(gp.k)+1:end]); end
end

function show(io::IO, gp::logisticGP)
    println(io, "logisticGP object:")
    println(io, "  Dim = $(gp.dim)")
    println(io, "  Number of observations = $(gp.nobsv)")
    println(io, "  Mean function:")
    show(io, gp.m, 2)
    println(io, "  Kernel:")
    show(io, gp.k, 2)
    println(io, "  Input observations = ")
    show(io, gp.x)
    print(io,"\n  Output observations = ")
    show(io, gp.y)
    print(io,"\n  Marginal Log-Likelihood = ")
    show(io, round(gp.mLL,3))
end
