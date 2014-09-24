% 
%     z1 = z(:,:,1); z1 = z1(:); 
%     z2 = z(:,:,2); z2 = z2(:); 
%     z3 = z(:,:,3); z3 = z3(:); 
%     
%     m1 = P*z1 + eps*ones(size(z1));
%     m2 = P*z2 + eps*ones(size(z1));
%     m3 = P*z3 + eps*ones(size(z1));
%     
%     p1 = z1./m1; 
%     p2 = z2./m2; 
%     p3 = z3./m3; 
%     
%     mom1 = reshape(repmat(M1*p1,[1 poolsz])',[size(z1,1) 1]); 
%     mom2 = reshape(repmat(M1*p2,[1 poolsz])',[size(z1,1) 1]);
%     mom3 = reshape(repmat(M1*p3,[1 poolsz])',[size(z1,1) 1]);  
%    
%     G = sum(M1)'; 
%     
%     moments_error = (mom3 - 2*mom2 + mom1);
%     
%     tmp = moments_error./m1;


% 
% dEp_dz2 = diff2_Ep_dz(z,P,M1,poolsz); 
% dEp_dz1 = diff_Ep_dz(z,P,M1); 

%  z = z + 0.01*ones(size(z)); 


 
 
%     z = ReLU(rand(codesz,bsz,3),0.5); 
    
%     z1 = z(:,:,1); z1 = z1(:); 
%     z2 = z(:,:,2); z2 = z2(:); 
%     z3 = z(:,:,3); z3 = z3(:); 
%     
%     m1 = P*z1 + eps*ones(size(z1));
%     m2 = P*z2 + eps*ones(size(z1));
%     m3 = P*z3 + eps*ones(size(z1));
%     
%     p1 = z1./m1; 
%     p2 = z2./m2; 
%     p3 = z3./m3; 
%     
%     mom1 = reshape(repmat(M1*p1,[1 poolsz])',[size(z1,1) 1]); 
%     mom2 = reshape(repmat(M1*p2,[1 poolsz])',[size(z1,1) 1]);
%     mom3 = reshape(repmat(M1*p3,[1 poolsz])',[size(z1,1) 1]);  
%     
%     G = sum(M1)'; 
%     
%     moments_error = (mom3 - 2*mom2 + mom1);
%     
%     dEp_dz2 = [moments_error./m1.*(G - z1.*mom1); 
%             -2*moments_error./m2.*(G - z2.*mom2);
%                moments_error./m3.*(G - z3.*mom3)];
%         
%     dEp_dz2 = reshape(dEp_dz2,size(z));       
    
 dEp_dz = diff_Ep_dz(z,P,M1); 
 dEp_dz2 = diff2_Ep_dz(z,P,M1,poolsz); 
 dEp_dz_num = check_gradients(x,z,W,P,P1,M1,dEr_dz,dEr_dW,dEs_dz,dEp_dz,dx);
 
 figure(2); 
 hold on; 
%  z1 = z(:,:,1); z1 = z1(:); 
%  plot(z1(:),'c');
 t = 2; 
 dEp_dz_num = dEp_dz_num(:,:,t);
 dEp_dz_num = dEp_dz_num(:); 
 dEp_dz = dEp_dz(:,:,t);
 dEp_dz = dEp_dz(:);
 dEp_dz2 = dEp_dz2(:,:,t);
 dEp_dz2 = dEp_dz2(:);
 dEp_dz2(abs(dEp_dz2)>100) = 10; 
 plot(dEp_dz_num,'go'); 
 plot(dEp_dz,'r'); 
 plot(dEp_dz2,'b*'); 
 
 
 














% insz = 16; %patch size
% codesz = 9;%2*insz; %code size 
% poolsz = 3; %pool group size
% poolst = 3; %pool group stride 
% bsz = 8; %batch size 
% L = 100; 
% outsz = (codesz-poolsz)/poolst + 1; %output size 
% 
% z1 = sym('z1','real'); z2 = sym('z2','real'); z3 = sym('z3','real'); 
% z4 = sym('z4','real'); z5 = sym('z5','real'); z6 = sym('z6','real'); 
% z7 = sym('z7','real'); z8 = sym('z8','real'); z9 = sym('z9','real'); 
% z = [z1 z2 z3 z4 z5 z6 z7 z8 z9]'; 
% 
% %pooling matrix 
% r = zeros(1,codesz); 
% r(1,1:poolsz) = 1; 
% P1 = zeros(outsz,codesz); 
% for ii = 1:outsz
%     P1(ii,:) = r; 
%     r = circshift(r,[1 poolst]); 
% end
% P = P1'*P1; 
% 
% 
% diag(P*z) - diag(z)*P









% 
% 
% 
% tic(); 
% dEp_dz = diff_Ep_dz(z,P,M1);
% disp('===')
% disp(toc());
% 
% tic(); 
% dEs_dz = diff_Es_dz(z,P);
% disp('===')
% disp(toc());
% 





% 
% xrec = zeros(size(x)); 
% 
% for ii = 1:3 
%     
%     xrec(:,:,ii) = W*z(:,:,ii); 
%     
% end
% 
% Ixrec = reshape(xrec, [16 16 1 24]);
% figure; 
% imdisp(Ixrec,'Border',[0.1 0.1])
% 
% 



% clc; 
% %pooling matrix 
% % 
% % Pb = P; 
% % % M1b = M1; 
% % for i = 1:bsz-1
% %     Pb = blkdiag(Pb,P);
% % %     M1b = blkdiag(M1b,M1);
% % end
% % 
% % diff_phase_dz = @(P,zi)(P*diag((P*zi + eps*ones(size(zi))).^-2)).*(diag(P*zi) - diag(zi)*P); 
% % dphase_dz2 = zeros(codesz*bsz,codesz*bsz,3); 
% % 
% % for i = 1:3 
% %  zi = z(:,1+(i-1)*bsz:i*bsz);
% %  zi = zi(:); 
% %  Pbzi = Pb*zi + eps*ones(size(zi)); 
% %  dphase_dz2(:,:,i) = diff_phase_dz(Pb,zi);
% % end
% % 
% % dphase_dz = diff_Ep_dz(P,zeros(1),z,bsz);
% % err = dphase_dz - dphase_dz2;
% % err = norm(err(:)); 
% 
% dphase_dz = diff_Ep_dz(P,zeros(1),z);
% 
% p = @(z,P) z./(P*z + eps*ones(size(z))); 
% 
% z = reshape(z,[16 8 3]); 
% 
% % 
% % zr = repmat(zi,[1 128]) + dx*ones(size(diag(zi))); 
% % zl = repmat(zi,[1 128]) - dx*ones(size(diag(zi))); 
% % D = (p(zr,Pb) - p(zl,Pb))/(2*dx); 
% % err = dphase_dz(:,:,1) - D;
% % err = norm(err(:)); 
% 
%  
% % numerical check 1
%     dphase_dz_num = zeros(128,128,3);
%     for n = 1:3
%         zi = z(:,:,n); 
%         zi = zi(:); 
%         for ii = 1:128
%             for jj = 1:128
%                 zr = zi; 
%                 zl = zi;
%                 zr(ii) = zr(ii) + dx; 
%                 zl(ii) = zl(ii) - dx;
%                 pr = p(zr,P); 
%                 pl = p(zl,P); 
%                 dphase_dz_num(jj,ii,n) = (pr(jj) - pl(jj))/(2*dx); 
%             end 
%         end
%     end
% 
% err = dphase_dz_num - dphase_dz; 
% err_val = norm(err(:));
% 
% %dEp/dz 
% % dEp_dz = zeros(size(z)); 
% % moments = M1*(z./(P*z + eps*ones(size(z))));
% % moments_err = (moments(:,2*bsz+1:3*bsz) - 2*moments(:,bsz+1:2*bsz) + moments(:,1:bsz))'; 
% % %  
% % % dEp_dz(:,1) = moments_err*M1b*dphase_dz(:,:,1); 
% % tmp = M1b*dphase_dz(:,:,1); 
% % % dEp_dz(:,2) = -2*moments_err*M1*dphase_dz(:,:,2); 
% % dEp_dz(:,3) = moments_err*M1*dphase_dz(:,:,3); 
