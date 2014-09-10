function [rec_error] = Er(x,W,z)  
    rec_error = 0; 
    for n = 1:3 
        err = x(:,:,n) - W*z(:,:,n);   
        rec_error = rec_error + 0.5*sum(err(:).^2);
    end
end
