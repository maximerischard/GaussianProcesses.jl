#Laplace approximation for the Gaussian process posterior.


function laplace(gp::GP x::Matrix{Float64}, y::Vector{Float64})

dim, nobsv = size(x,1);
    m = meanf(gp.m,gp.x)                          #Evaluate the mean function
    k = crossKern(gp.x,gp.k)                      #Evaluate the kernel
    lik = likf()                                  #Evaluate the likelihood function

  alpha = zeros(nobsv,1);                      #Starting value for α, NEEDS TO BE CHECKED

alpha = irls(alpha, m,K,lik);        #NEED TO CREATE OPTIMIZATION FUNCTION TO MAXIMISE PHI, REQUIRE 1ST AND 2ND DERIVATIVES OF PHI


#Calculate the posterior    
f = K*alpha+m;                                  # compute latent function values
[lp,dlp,d2lp,d3lp] = likf(f); W = -d2lp; 
sW = sqrt(abs(W));

L = PDMat(eye(nobsv)+sW*sW'.*k);                 
nlZ = alpha'*(f-m)/2 + sum(log(diag(L))-lp);   # ..(f-m)/2 -lp +ln|B|/2 negative log-likelihood

#Calculate the derivatives
  dnlZ = hyp;                                   # allocate space for derivatives
    Z = repmat(sW,1,nobsv).*solve_chol(post.L,diag(sW)); #sW*inv(B)*sW=inv(K+inv(W))
    C = post.L'\(repmat(sW,1,nobsv).*K);                     # deriv. of ln|B| wrt W
    g = (diag(K)-sum(C.^2,1)')/2;                    # g = diag(inv(inv(K)+W))/2

  dfhat = g.*d3lp;  # deriv. of nlZ wrt. fhat: dfhat=diag(inv(inv(K)+W)).*d3lp/2
    
  for i=1:length(hyp.cov)                                    # covariance hypers
    dK = feval(cov{:}, hyp.cov, x, [], i);
    dnlZ.cov(i) = sum(sum(Z.*dK))/2 - alpha'*dK*alpha/2;         # explicit part
    b = dK*dlp;                            # b-K*(Z*b) = inv(eye(nobsv)+K*diag(W))*b
    dnlZ.cov(i) = dnlZ.cov(i) - dfhat'*( b-K*(Z*b) );            # implicit part
  end
    
  for i=1:length(hyp.lik)                                    # likelihood hypers
    [lp_dhyp,dlp_dhyp,d2lp_dhyp] = feval(lik{:},hyp.lik,y,f,[],inf,i);
    dnlZ.lik(i) = -g'*d2lp_dhyp - sum(lp_dhyp);                  # explicit part
    b = K*dlp_dhyp;                        # b-K*(Z*b) = inv(eye(nobsv)+K*diag(W))*b
    dnlZ.lik(i) = dnlZ.lik(i) - dfhat'*( b-K*(Z*b) );            # implicit part
  end
    
  for i=1:length(hyp.mean)                                         # mean hypers
    dm = feval(mean{:}, hyp.mean, x, i);
    dnlZ.mean(i) = -alpha'*dm;                                   # explicit part
    dnlZ.mean(i) = dnlZ.mean(i) - dfhat'*(dm-K*(Z*dm));          # implicit part
  end


# Evaluate criterion Psi(alpha) = alpha'*K*alpha + likfun(f), where 
# f = K*alpha+m, and likfun(f) = feval(lik{:},hyp.lik,y,  f,  [],inf).
    
function Psi(alpha,m,K,likfun)
  f = K*alpha+m;
  [lp,dlp,d2lp] = likfun(f); W = -d2lp;
  psi = alpha'*(f-m)/2 - sum(lp);
  dpsi = K*(alpha-dlp);
    out = [psi,dpsi,f,alpha,dlp,W]
    return out
end



end
