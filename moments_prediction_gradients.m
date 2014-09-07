% Checks the gradients of the various terms in the loss using central difference 
clc; 
clearvars -except 'X'; 
close all; 
addpath ./fprop_bprop/

% Constants 
%dimensions 
insz = 16*16; %patch size
codesz = 16; %code size 
poolsz = 4; %pool group size
bsz = 8; %batch size 
outsz = codesz/poolsz; %output size 
assert(rem(outsz,1)==0,'outsz is not an integer'); 
numerical_check = true; 
display_sample = false; 
dx = 1e-6; %step size for numerical check
%loss multipliers 
wl1 = 0.5; 
ws = 0.5; 
wp = 0.5; 
% Generate toy dataset 
if ~exist('X')  %#ok<EXIST>
    disp('generating dataset...'); 
    options = struct('ntemplates',2); 
    X=generate_jitter_data2d(options);
    disp('done!'); 
end
batch = zeros(insz,bsz,3);
order = im2col(randperm(size(X,2)-2),[1,bsz]); 
batch = get_batch(X,order(:,1),batch); 
if display_sample == true
   n = 1; 
   Ibatch = reshape(batch, [16 16 1 bsz 3]); 
   imdisp(Ibatch(:,:,:,1,:),'Border',[0.1 0.1]); 
end
% Loss Definition and Initialization 
% ReLU 
ReLU = @(x) x.*(x>0); 

%data sample 
x = batch; 

%codes for three temporal samples 
z = ReLU(rand(codesz,bsz,3) - 0.5); 

%normalized decoder dictionary 
W = rand(insz,codesz) - 0.5;
normalize_cols = @(W,insz) W./repmat(sqrt(sum(W.^2,1)),[insz,1]); 
W = normalize_cols(W,insz); 

%pooling matrix 
r = zeros(1,codesz); 
r(1,1:poolsz) = 1; 
P = zeros(outsz,codesz); 
for ii = 1:outsz
    P(ii,:) = r; 
    r = circshift(r,[1 poolsz]); 
end
P = P'*P; 

%mini-batches
Pb = P; 
for i = 1:bsz-1
    Pb = blkdiag(Pb,P);
end
P = Pb; 

%check that: z = ((P'*m).*s)
% assert(norm(z-(P*z + eps*ones(size(z))).*(z./(P*z + eps*ones(size(z))))) < 1e-10); 

%first moment matrix (define coordinates from -1 to +1)  
% r = zeros(1,codesz); 
% r(1:poolsz) = linspace(-1,1,poolsz); 
% M1 = zeros(outsz,codesz);
% for ii = 1:outsz
%     M1(ii,:) = r; 
%     r = circshift(r,[1 poolsz]); 
% end
% 

%L2-slowness-error 
Es = @(mag,bsz) 0.5*sum(sum((mag(:,bsz+1:2*bsz)-mag(:,1:bsz)).^2+(mag(:,2*bsz+1:3*bsz)-mag(:,bsz+1:2*bsz)).^2)); 
% 
% %L2-Prediction-error 
% Ep = @(moments,bsz) 0.5*sum(sum((moments(:,2*bsz+1:3*bsz) - 2*moments(:,bsz+1:2*bsz) + moments(:,1:bsz)).^2));  
%  
% %L2-reconstruction-error 
% Er = @(x,W,z) 0.5*sum((sum((x - W*z).^2,1))); 
% 
% %Loss 
L = Er(x(:,:,1),W,z(:,:,1)); %+ wp*Ep(M1*(z./(P*z + eps*ones(size(z)))),bsz) + ws*Es((P*z + eps*ones(size(z))),bsz) + wl1*sum(abs(z(:)));
% 
% % Define/check gradients 
% 
% %dEr/dz 
dErdz = diff_Er_dz(x,W,z); 
% 
%dEr/dW
dErdW = diff_Er_dW(x,W,z); 
% 
% %dEs/dz = (dEs/dmag)(dmag/dz)
% dEs_dz = zeros(size(z));    
% dEs_dz = diff_Es_dz(P,z,bsz,dEs_dz);
% 
% % % %dEp/dz 
% % dEp_dz = diff_Ep_dz(P,M1,z,bsz); 
% % zi = reshape(numel(z(:,1:bsz);
% % Pzi = reshape((P*zi + eps*ones(size(zi))).^-2,[numel(zi),1]);
% % f = (P*diag(Pzi.^-2).*(diag(Pzi) - diag(reshape(zi,[numel(zi),1])*P); 
% 
% if numerical_check == true 
%     
%     disp('checking gradients via central difference...') 
% 
    %numerical check 1
    dErdz_num = zeros(size(z)); 
    for n = 1:3 
        for ii = 1:size(z,1)
            for jj = 1:size(z,2) 
                zr = z(:,:,n); 
                zl = z(:,:,n);
                zr(ii,jj) = zr(ii,jj) + dx; 
                zl(ii,jj) = zl(ii,jj) - dx; 
                dErdz_num(ii,jj,n) = (Er(x(:,:,n),W,zr) - Er(x(:,:,n),W,zl))/(2*dx); 
            end 
        end
    end
    err = dErdz - dErdz_num; 
    assert(sum(abs(err(:)))/sum(abs(dErdz(:))) < 1e-6); 

    %numerical check 2
    dErdW_num = zeros(size(W));  
    for n = 1:3 
        for ii = 1:size(W,1) 
            for jj = 1:size(W,2) 
                Wr = W; 
                Wl = W;
                Wr(ii,jj) = Wr(ii,jj) + dx; 
                Wl(ii,jj) = Wl(ii,jj) - dx; 
                dErdW_num(ii,jj) = dErdW_num(ii,jj) + ((Er(x(:,:,n),Wr,z(:,:,n)) - Er(x(:,:,n),Wl,z(:,:,n)))/(2*dx)); 
            end
        end
    end
    err = dErdW - dErdW_num; 
    assert(sum(abs(err(:)))/sum(abs(dErdW(:))) < 1e-6); 
% 
%  
%     %numerical check 3
%     dEsdz_num = zeros(size(z)); 
%     for ii = 1:size(z,1)
%         for jj = 1:size(z,2) 
%             zr = z; 
%             zl = z;
%             zr(ii,jj) = zr(ii,jj) + dx; 
%             zl(ii,jj) = zl(ii,jj) - dx; 
%             dEsdz_num(ii,jj) = (Es((P*zr + eps*ones(size(zr))),bsz) - Es((P*zl + eps*ones(size(zl))),bsz))/(2*dx); 
%         end 
%     end
%     assert(norm(dEs_dz - dEsdz_num, 1)/norm(dEsdz_num,1) < 1e-6); 
% % % 
% % 
%     % numerical check 4 
% %     dEp_dz_num = zeros(size(z)); 
% %     for ii = 1:size(z,1)
% %         for jj = 1:size(z,2) 
% %             zr = z; 
% %             zl = z;
% %             zr(ii,jj) = zr(ii,jj) + dx; 
% %             zl(ii,jj) = zl(ii,jj) - dx; 
% %             dEp_dz_num(ii,jj) = (Ep(M1*(zr./(P*zr + eps*ones(size(zr))))) - Ep(M1*(zl./(P*zl + eps*ones(size(zl))))))/(2*dx); 
% %         end 
% %     end
% %     
% %     unstable when group is off, so relax the accuracy (?) 
% %     assert(norm(dEp_dz - dEp_dz_num, 1)/norm(dEp_dz_num,1) < 1e-3); 
% 
%     disp('passed!')
% 
% end 
% 
% % Training 













