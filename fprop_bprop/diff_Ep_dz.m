function [ dEp_dz ] = diff_Ep_dz(P,M1,z,bsz)
%computes gradient of linear prediction of first moment of phase w.r.t. activations 

diff_phase_dz = @(P,zi,Pzi)(P*diag((Pzi + eps*ones(size(zi))).^-2)).*(diag(Pzi) - diag(zi)*P); 
dphase_dz = zeros(size(z,1)*bsz,size(z,1)*bsz,3); 
% Pbz = zeros(size(z,1)*bsz,3); 


for i = 1:3 
 zi = z(:,1+(i-1)*bsz:i*bsz);
 zi = zi(:); 
 Pzi = P*zi;
%  Pz(:,i) = Pzi; 
 dphase_dz(:,:,i) = diff_phase_dz(P,zi,Pzi); 
end

dEp_dz = dphase_dz; 
%dEp/dz 
% dEp_dz = zeros(size(z)); 
% moments = M1*(z./(P*z + eps*ones(size(z))));
% moments_err = (moments(:,3) - 2*moments(:,2) + moments(:,1))'; 
% 
% dEp_dz(:,1) = moments_err*M1*dphase_dz(:,:,1); 
% dEp_dz(:,2) = -2*moments_err*M1*dphase_dz(:,:,2); 
% dEp_dz(:,3) = moments_err*M1*dphase_dz(:,:,3); 

end

