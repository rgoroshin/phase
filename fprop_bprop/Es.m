function [slow_error] = Es(z,P)
    
    mag = zeros(size(P,1),3); 
    for ii = 1:3 
        zi = z(:,:,ii); 
        zi = zi(:); 
        mag(:,ii) = P*zi; 
    end
    
    err1 = mag(:,2)-mag(:,1); 
    err2 = mag(:,2)-mag(:,1);
    
    slow_error = 0.5*(err1.^2 + err2.^2); 
    
end