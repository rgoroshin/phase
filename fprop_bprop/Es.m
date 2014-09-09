function [slow_error] = Es(z,P1)
    
    mag = zeros(size(P1,1),3); 
    for ii = 1:3 
        zi = z(:,:,ii); 
        zi = zi(:); 
        mag(:,ii) = P1*zi; 
    end
    
    err1 = mag(:,2)-mag(:,1); 
    err2 = mag(:,3)-mag(:,2);
    
    slow_error = 0.5*sum(err1.^2 + err2.^2); 
    
end