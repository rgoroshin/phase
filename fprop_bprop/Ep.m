function [pred_error] = Ep(z,P,M1)  
    
    ph = zeros(numel(z(:,:,1)),3); 

    for n = 1:3  
        zi = z(:,:,n); 
        zi = zi(:); 
        ph(:,n) = zi./(P*zi + eps*ones(size(zi))); 
    end

    moments = M1*ph; 
    moments_error = (moments(:,3) - 2*moments(:,2) + moments(:,1))';  
    pred_error = 0.5*sum(moments_error.^2);

end

