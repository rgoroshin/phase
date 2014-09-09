function [dEr_dW] = diff_Er_dW(x,W,z)
    dEr_dW = zeros(size(W)); 
    for ii = 1:3 
        dEr_dW = dEr_dW + (x(:,:,ii) - W*z(:,:,ii))*(-z(:,:,ii)');
    end
end

