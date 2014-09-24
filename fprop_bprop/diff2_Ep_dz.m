function [ dEp_dz ] = diff2_Ep_dz(z,P,M1,poolsz)
%computes gradient of linear prediction of first moment of phase w.r.t. activations 

    z1 = z(:,:,1); z1 = z1(:); 
    z2 = z(:,:,2); z2 = z2(:); 
    z3 = z(:,:,3); z3 = z3(:); 

    m1 = P*z1 + eps*ones(size(z1));
    m2 = P*z2 + eps*ones(size(z1));
    m3 = P*z3 + eps*ones(size(z1));
    
    p1 = z1./m1; 
    p2 = z2./m2; 
    p3 = z3./m3; 
    
    mom1 = reshape(repmat(M1*p1,[1 poolsz])',[size(z1,1) 1]); 
    mom2 = reshape(repmat(M1*p2,[1 poolsz])',[size(z1,1) 1]);
    mom3 = reshape(repmat(M1*p3,[1 poolsz])',[size(z1,1) 1]);  
    
    G = sum(M1)'; 
    
    moments_error = (mom3 - 2*mom2 + mom1);
    
    dEp_dz = [moments_error./m1.*(G.*(abs(mom1)>eps) - mom1), ...
           -2*moments_error./m2.*(G.*(abs(mom2)>eps) - mom2), ...
              moments_error./m3.*(G.*(abs(mom3)>eps) - mom3)];
% keyboard
    dEp_dz = reshape(dEp_dz,size(z));       

 
end
