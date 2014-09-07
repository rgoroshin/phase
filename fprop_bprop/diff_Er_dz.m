function [dEr_dz] = diff_Er_dz(x,W,z)
    dEr_dz = zeros(size(z)); 
    for ii = 1:3 
        dEr_dz(:,:,ii) = -W'*(x(:,:,ii) - W*z(:,:,ii));
    end
end

