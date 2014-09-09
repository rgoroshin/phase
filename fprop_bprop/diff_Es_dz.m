function [grad] = diff_Es_dz(P,z,bsz,grad)

    grad(:,1:bsz) = P*P*(z(:,1:bsz)-z(:,bsz+1:2*bsz)); 
    grad(:,bsz+1:2*bsz) = P*P*(2*z(:,bsz+1:2*bsz)-z(:,1:bsz)-z(:,2*bsz+1:3*bsz)); 
    grad(:,2*bsz+1:3*bsz) = P*P*(z(:,2*bsz+1:3*bsz)-z(:,bsz+1:2*bsz)); 

end

