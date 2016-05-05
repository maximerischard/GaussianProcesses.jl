@doc """
# Description
A function for running a variety of MCMC algorithms for estimating the GP hyperparameters. This function uses the MCMC algorithms provided by the Lora package and the user is referred to this package for further details.

# Arguments:
* `gp::GP`: Predefined Gaussian process type
* `start::Vector{Float64}`: Select a starting value, default is taken as current GP parameters
* `sampler::Lora.MCSampler`: MCMC sampler selected from the Lora package
* `mcrange::Lora.BasicMCRange`: Choose number of MCMC iterations and burnin length, default is nsteps=5000, burnin = 1000
""" ->
function mcmc(gp::GP;
              start::Vector{Float64}=get_params(gp),
              sampler::Lora.MCSampler=Lora.MH(ones(length(get_params(gp)))),
              mcrange::Lora.BasicMCRange=BasicMCRange(nsteps=5000, burnin=1000))


    store = get_params(gp) #store original parameters
    npara = length(store)  #number of parameters
    
    prior = Distributions.MvNormal(zeros(npara),100*eye(npara)) #default prior
    
    function mll(hyp::Vector{Float64})  #log-target
        set_params!(gp, hyp)
        update_mll!(gp)
        return gp.mLL + logpdf(prior,hyp)
    end
    
    function dmll(hyp::Vector{Float64}) #gradient of the log-target
        set_params!(gp, hyp)
        update_mll_and_dmll!(gp)
        return gp.dmLL + sum(gradlogpdf(prior,hyp))
    end
    starting = Dict(:p=>start)
    q = BasicContMuvParameter(:p, logtarget=mll,gradlogtarget=dmll) 
    model = likelihood_model(q, false)                               #set-up the model
    tune = VanillaMCTuner(period=mcrange.burnin)                     #set length of tuning (default to burnin length)
    job = BasicMCJob(model, sampler, mcrange, starting,tuner=tune)   #set-up MCMC job
    print(job)                                                       #display MCMC set-up for the user
    run(job)
    chain = Lora.output(job)
    set_params!(gp,store)      #reset the parameters stored in the GP to original values
    return chain.value
end    
    
