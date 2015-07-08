#Laplace approximation for the Gaussian process posterior.

function laplace!(gp::logisticGP)

    m = meanf(gp.m,gp.x)                          #Evaluate the mean function
    k = crossKern(gp.x,gp.k)                      #Evaluate the kernel

  alpha = zeros(gp.nobsv,1)                         #Starting value for α, NEEDS TO BE CHECKED

function psi(alpha::Vector{Float64},m::Vector{Float64},k::Matrix{Float64})             #Psi = negative log-posterior as defined in Rasmussen and Williams (2006) p.42
  f = k*alpha+m;
  lp = like(gp.y,f)
  psi = alpha'*(f-m)/2 - sum(lp) #up to a normalising constant
    return psi           
end

#1st derivative of Psi    
function dPsi!(alpha::Vector{Float64},m::Vector{Float64},k::Matrix{Float64})
      dlp = grad1_likef(gp.y,f) 
      dpsi = k*(alpha-dlp)
      return dpsi
end

#2nd derivative of Psi
function d2Psi!(alpha,m,k)
      d2lp = grad2_likef(gp.y,f)
      dpsi2 = -diag(d2lp) - inv(PDMat(k))
      return dpsi2
end

func = TwiceDifferentiableFunction(psi,dPsi!,d2Psi!)

res = optimize(func, alpha, method = :newton)   #NEED TO CREATE OPTIMIZATION FUNCTION TO MAXIMISE PHI, REQUIRE 1ST AND 2ND DERIVATIVES OF PHI

alpha = res.minimum    

#Calculate the posterior    
f = k*alpha+m;                                  # compute latent function values
lp = like(gp.y,f); W = -grad2_likef(gp.y,f); 
sW = sqrt(abs(W));

L = PDMat(eye(gp.nobsv)+sW*sW'.*k);                 
gp.mLL = alpha'*(f-m)/2 + sum(log(diag(L))-lp);   # ..(f-m)/2 -lp +ln|B|/2 negative marginal log-likelihood

## #Calculate the derivatives of the marginal log-likelihood wrt to the hyperparameters (see Rasmussen and Williams (2006) p. 125)
##     Z = sW.*L\diag(sW); #sW*inv(B)*sW=inv(K+inv(W))
##     C = L\(sW.*k);                     # deriv. of ln|B| wrt W
##     g = (diag(k)-sum(C.^2,1)')/2;                    # g = diag(inv(inv(K)+W))/2

##   dfhat = g.*grad3_likef(gp.y,f);  # deriv. of mLL wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2

##   # Mean function hyperparameters
##     if mean
##         Mgrads = grad_stack(gp.x, gp.m)                              # [dM/dθᵢ]    
##         for i in 1:num_params(gp.m)
##             gp.dmLL[i] = -alpha'*Mgrads[:,i] - dfhat'*(Mgrads[:,i]-k*(Z*Mgrads[:,i]));         
##         end
##     end

##   # Kernel hyperparameters
##     if kern
##         Kgrads = grad_stack(gp.x, gp.k)                              # [dK/dθᵢ]    
##         for i in 1:num_params(gp.k)
##             b = Kgrads[:,:,i]*dlp;       # b-K*(Z*b) = inv(eye(nobsv)+K*diag(W))*b
##             gp.dmLL[i+mean*num_params(gp.m)] = sum(sum(Z.*Kgrads[:,:,i]))/2-alpha'*Kgrads[:,:,i]*alpha/2-dfhat'*(b-k*(Z*b));  
##         end
##    end
end




