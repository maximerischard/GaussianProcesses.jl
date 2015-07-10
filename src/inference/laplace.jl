#Laplace approximation for the Gaussian process posterior.

function laplace!(gp::logisticGP)

    m = meanf(gp.m,gp.x)                      # Evaluate the mean function
    k = crossKern(gp.x,gp.k)                  # Evaluate the kernel
    gp.alpha = zeros(gp.nobsv)                 # Starting value for α, NEEDS TO BE CHECKED

    #Psi = log-posterior as defined in Rasmussen and Williams (2006) p.42
    function psi(alpha::Vector{Float64})             
        f = k*alpha+m;
        lp = like(gp.y,f)
        psi = -dot(alpha, (f-m))/2.0 + sum(lp) #up to a normalising constant
        return -psi           
    end

    #1st derivative of Psi    
    function dPsi!(alpha::Vector{Float64}, grad::Vector{Float64})
        f = k*alpha+m;
        dlp = grad1_likef(gp.y,f) 
      #  dpsi = k*(alpha-dlp)
        dpsi = dlp - alpha
        grad[:] = -dpsi
    end

    #2nd derivative of Psi
    function d2Psi!(alpha::Vector{Float64}, hess::Matrix{Float64})
        f = k*alpha+m;
        d2lp = grad2_likef(gp.y,f)
        dpsi2 = diagm(d2lp) - inv(k)
        hess[:,:] = -dpsi2
    end

#func = TwiceDifferentiableFunction(psi,dPsi!,d2Psi!)
    #res = optimize(psi, dPsi!, gp.alpha,show_trace=true)
    res = optimize(psi, dPsi!, d2Psi!, gp.alpha, show_trace=true, method=:newton)
    #res = optimize(psi, alpha; method = :nelder_mead)   #NEED TO CREATE OPTIMIZATION FUNCTION TO MAXIMISE PHI, REQUIRE 1ST AND 2ND DERIVATIVES OF PHI

        
    gp.alpha = res.minimum  

    #Calculate the posterior    
    f = k*gp.alpha+m;                                  # compute latent function values
    lp = like(gp.y,f); W = -grad2_likef(gp.y,f);
    sW = sqrt(abs(W)).*(W./abs(W));                   #Use W./abs(W) to preserve the sign

    L = PDMat(eye(gp.nobsv)+sW*sW'.*k);                 
    gp.mLL = -dot(gp.alpha,f-m)/2.0 - sum(log(diag(L))+lp);   #marginal log-likelihood
end


function update_laplace_and_dmll!(gp::logisticGP,mean::Bool=true, kern::Bool=true)
    laplace!(gp::logisticGP)

    m = meanf(gp.m,gp.x)                      # Evaluate the mean function
    k = crossKern(gp.x,gp.k)                  # Evaluate the kernel
    
    f = k*gp.alpha+m;                                  # compute latent function values
    lp = like(gp.y,f); W = -grad2_likef(gp.y,f); 
    sW = sqrt(abs(W)).*(W./abs(W));                   #Use W./abs(W) to preserve the sign

    L = PDMat(eye(gp.nobsv)+sW*sW'.*k);
    
#Calculate the derivatives of the marginal log-likelihood wrt to the hyperparameters (see Rasmussen and Williams (2006) p. 125)

    gp.dmLL = Array(Float64, mean*num_params(gp.m) + kern*num_params(gp.k))
    Z = repmat(sW,1,gp.nobsv).*(L\diagm(sW)); #sW*inv(B)*sW=inv(K+inv(W))
    C = L\(sW.*k);                     # deriv. of ln|B| wrt W
    g = (diag(k)-sum(C.^2,1)')/2.0;                    # g = diag(inv(inv(K)+W))/2

  dfhat = g.*grad3_likef(gp.y,f);  # deriv. of mLL wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2

  # Mean function hyperparameters
    if mean
        Mgrads = grad_stack(gp.x, gp.m)                              # [dM/dθᵢ]    
        for i in 1:num_params(gp.m)
            gp.dmLL[i] = dot(gp.alpha,Mgrads[:,i]) + dot(dfhat,(Mgrads[:,i]-k*(Z*Mgrads[:,i])));         
        end
    end

  # Kernel hyperparameters
    if kern
        Kgrads = grad_stack(gp.x, gp.k)                              # [dK/dθᵢ]    
        for i in 1:num_params(gp.k)
            b = Kgrads[:,:,i]*grad1_likef(gp.y,f) ;       # b-K*(Z*b) = inv(eye(nobsv)+K*diag(W))*b
            gp.dmLL[i+mean*num_params(gp.m)] = -sum(sum(Z.*Kgrads[:,:,i]))/2.0 + dot(gp.alpha,Kgrads[:,:,i]*gp.alpha)/2.0 + dot(dfhat,(b-k*(Z*b)));  
        end
   end
end


