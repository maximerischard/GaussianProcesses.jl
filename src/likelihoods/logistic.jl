#Logistic likelihood function

type Logistic <: Likelihood end


function like(logistic::Logistic, y::Vector{Float64},f::Vector{Float64})
    y[y.==0]=-1
    return -log(1+exp(-y.*f))
end

#1st derivative with respect to f
function grad1_likef(logistic::Logistic, y::Vector{Float64},f::Vector{Float64})
        s   = min(0,f); 
        p   = exp(s)./(exp(s)+exp(s-f));                    
        dlp_df = (y+1)/2-p;                           # 1st derivative of log likelihood
        return dlp_df
end

#2nd derivative with respect to f
function grad2_likef(logistic::Logistic, y::Vector{Float64},f::Vector{Float64})
        s   = min(0,f); 
        d2lp_df2 = -exp(2*s-f)./(exp(s)+exp(s-f)).^2; # 2nd derivative of log likelihood
        return d2lp_df2
end

#3rd derivative with respect to f
function grad3_likef(logistic::Logistic, y::Vector{Float64},f::Vector{Float64})
        s   = min(0,f); 
        p   = exp(s)./(exp(s)+exp(s-f));                    
        d2lp_df2 = -exp(2*s-f)./(exp(s)+exp(s-f)).^2; # 2nd derivative of log likelihood
        d3lp_df3 = 2*d2lp.*(0.5-p);                   # 3rd derivative of log likelihood
        return d3lp_df3
end
