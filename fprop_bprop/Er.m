function [rec_error] = Er(x,W,z)  
    err = x - W*z;   
    rec_error = 0.5*sum(err(:).^2);
end
