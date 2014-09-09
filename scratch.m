clc; 
%pooling matrix 
% 
% Pb = P; 
% % M1b = M1; 
% for i = 1:bsz-1
%     Pb = blkdiag(Pb,P);
% %     M1b = blkdiag(M1b,M1);
% end
% 
% diff_phase_dz = @(P,zi)(P*diag((P*zi + eps*ones(size(zi))).^-2)).*(diag(P*zi) - diag(zi)*P); 
% dphase_dz2 = zeros(codesz*bsz,codesz*bsz,3); 
% 
% for i = 1:3 
%  zi = z(:,1+(i-1)*bsz:i*bsz);
%  zi = zi(:); 
%  Pbzi = Pb*zi + eps*ones(size(zi)); 
%  dphase_dz2(:,:,i) = diff_phase_dz(Pb,zi);
% end
% 
% dphase_dz = diff_Ep_dz(P,zeros(1),z,bsz);
% err = dphase_dz - dphase_dz2;
% err = norm(err(:)); 

dphase_dz = diff_Ep_dz(P,zeros(1),z);

p = @(z,P) z./(P*z + eps*ones(size(z))); 

i = 2; 
z = reshape(z,[16 8 3]); 
zi = z(:,:,i); 
zi = zi(:); 
% 
% zr = repmat(zi,[1 128]) + dx*ones(size(diag(zi))); 
% zl = repmat(zi,[1 128]) - dx*ones(size(diag(zi))); 
% D = (p(zr,Pb) - p(zl,Pb))/(2*dx); 
% err = dphase_dz(:,:,1) - D;
% err = norm(err(:)); 

 
% numerical check 1
    dphase_dz_num = zeros(128,128); 
    for ii = 1:128
        for jj = 1:128
            zr = zi; 
            zl = zi;
            zr(ii) = zr(ii) + dx; 
            zl(ii) = zl(ii) - dx;
            pr = p(zr,P); 
            pl = p(zl,P); 
            dphase_dz_num(jj,ii) = (pr(jj) - pl(jj))/(2*dx); 
        end 
    end

err = dphase_dz_num - dphase_dz(:,:,i); 
err_val = norm(err(:));

%dEp/dz 
% dEp_dz = zeros(size(z)); 
% moments = M1*(z./(P*z + eps*ones(size(z))));
% moments_err = (moments(:,2*bsz+1:3*bsz) - 2*moments(:,bsz+1:2*bsz) + moments(:,1:bsz))'; 
% %  
% % dEp_dz(:,1) = moments_err*M1b*dphase_dz(:,:,1); 
% tmp = M1b*dphase_dz(:,:,1); 
% % dEp_dz(:,2) = -2*moments_err*M1*dphase_dz(:,:,2); 
% dEp_dz(:,3) = moments_err*M1*dphase_dz(:,:,3); 
