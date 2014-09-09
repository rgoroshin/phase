function [ dphase_dz ] = diff_phase_dz(z,P)
%computes gradient of the phase w.r.t. activations 

diff_phase_dzi = @(P,zi,Pzi)(P*diag((Pzi + eps*ones(size(zi))).^-2)).*(diag(Pzi) - diag(zi)*P); 
dphase_dz = zeros(numel(z(:,:,1)),numel(z(:,:,1)),3); 

for i = 1:3 
 zi = z(:,:,i); 
 zi = zi(:); 
 Pzi = P*zi; 
 dphase_dz(:,:,i) = diff_phase_dzi(P,zi,Pzi); 
end


end