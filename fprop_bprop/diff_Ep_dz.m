function [ dEp_dz ] = diff_Ep_dz(P,M1,z)
%computes gradient of linear prediction of first moment of phase w.r.t. activations 

diff_phase_dz = @(P,zi)(P*diag((P*zi + eps*ones(size(zi))).^-2)).*(diag(P*zi) - diag(zi)*P); 
dphase_dz = zeros(size(z,1),size(z,1),3); 
dphase_dz(:,:,1) = diff_phase_dz(P,z(:,1)); 
dphase_dz(:,:,2) = diff_phase_dz(P,z(:,2)); 
dphase_dz(:,:,3) = diff_phase_dz(P,z(:,3)); 

%dEp/dz 
dEp_dz = zeros(size(z)); 
moments = M1*(z./(P*z + eps*ones(size(z))));
moments_err = (moments(:,3) - 2*moments(:,2) + moments(:,1))'; 

dEp_dz(:,1) = moments_err*M1*dphase_dz(:,:,1); 
dEp_dz(:,2) = -2*moments_err*M1*dphase_dz(:,:,2); 
dEp_dz(:,3) = moments_err*M1*dphase_dz(:,:,3); 

end

