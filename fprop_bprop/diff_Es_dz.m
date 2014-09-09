function [dEs_dz] = diff_Es_dz(z,P)

    dEs_dz = zeros(size(z));  
    z1 = z(:,:,1); 
    z2 = z(:,:,2); 
    z3 = z(:,:,3); 
    z1 = z1(:); 
    z2 = z2(:); 
    z3 = z3(:); 
    dEs_dz(:,:,1) = reshape(P*(z1-z2),size(z(:,:,1))); 
    dEs_dz(:,:,2) = reshape(P*(2*z2-z1-z3),size(z(:,:,1))); 
    dEs_dz(:,:,3) = reshape(P*(z3-z2),size(z(:,:,1))); 
    
end

