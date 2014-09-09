function [ dEp_dz ] = diff_Ep_dz(dphase_dz,moments_error,M1,zsz)
%computes gradient of linear prediction of first moment of phase w.r.t. activations 

 dEp_dz = [moments_error*M1*dphase_dz(:,:,1); -2*moments_error*M1*dphase_dz(:,:,2); moments_error*M1*dphase_dz(:,:,3)]'; 
 dEp_dz = reshape(dEp_dz,zsz);    
 
end

