clc; 
%pooling matrix 

Pb = P; 
M1b = M1; 
for i = 1:bsz-1
    Pb = blkdiag(Pb,P);
    M1b = blkdiag(M1b,M1);
end

diff_phase_dz = @(P,zi)(P*diag((P*zi + eps*ones(size(zi))).^-2)).*(diag(P*zi) - diag(zi)*P); 
dphase_dzi = zeros(codesz*bsz,codesz*bsz,3); 

for i = 1:3 
 zi = z(:,1+(i-1)*bsz:i*bsz);
 zi = zi(:); 
 Pbzi = Pb*zi + eps*ones(size(zi)); 

 dphase_dzi(:,:,i) = diff_phase_dz(Pb,zi);
end

dphase_dz2 = diff_Ep_dz(P,zeros(1),z,bsz);
err = dphase_dz - dphase_dz2;
err = norm(err(:)); 

%dEp/dz 
dEp_dz = zeros(size(z)); 
moments = M1*(z./(P*z + eps*ones(size(z))));
moments_err = (moments(:,2*bsz+1:3*bsz) - 2*moments(:,bsz+1:2*bsz) + moments(:,1:bsz))'; 
%  
% dEp_dz(:,1) = moments_err*M1b*dphase_dz(:,:,1); 
tmp = M1b*dphase_dz(:,:,1); 
% dEp_dz(:,2) = -2*moments_err*M1*dphase_dz(:,:,2); 
% dEp_dz(:,3) = moments_err*M1*dphase_dz(:,:,3); 