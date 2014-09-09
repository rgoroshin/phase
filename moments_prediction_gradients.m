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
P1 = zeros(outsz,codesz); 
for ii = 1:outsz
    P1(ii,:) = r; 
    r = circshift(r,[1 poolsz]); 
end
P = P1'*P1; 

%first moment matrix (define coordinates from -1 to +1)  
r = zeros(1,codesz); 
r(1:poolsz) = linspace(-1,1,poolsz); 
M1 = zeros(outsz,codesz);
for ii = 1:outsz
    M1(ii,:) = r; 
    r = circshift(r,[1 poolsz]); 
end

%mini-batches
Pb = P;  
P1b = P1; 
M1b = M1; 
for i = 1:bsz-1
    Pb = blkdiag(Pb,P);
    P1b = blkdiag(P1b,P1);
    M1b = blkdiag(M1b,M1);
end
P = Pb; 
P1 = P1b; 
M1 = M1b; 

%check that: z = ((P'*m).*s)
% assert(norm(z-(P*z + eps*ones(size(z))).*(z./(P*z + eps*ones(size(z))))) < 1e-10); 



%L2-slowness-error 
% Es = @(mag,bsz) 0.5*sum(sum((mag(:,bsz+1:2*bsz)-mag(:,1:bsz)).^2+(mag(:,2*bsz+1:3*bsz)-mag(:,bsz+1:2*bsz)).^2)); 
% 
% %L2-Prediction-error 
% Ep = @(moments,bsz) 0.5*sum(sum((moments(:,2*bsz+1:3*bsz) - 2*moments(:,bsz+1:2*bsz) + moments(:,1:bsz)).^2));  
%  
% %L2-reconstruction-error 
% Er = @(x,W,z) 0.5*sum((sum((x - W*z).^2,1))); 
% 
% %Loss 
% L = Er(x(:,:,1),W,z(:,:,1)); %+ wp*Ep(M1*(z./(P*z + eps*ones(size(z)))),bsz) + ws*Es((P*z + eps*ones(size(z))),bsz) + wl1*sum(abs(z(:)));
% 
% % Define/check gradients 
% 
% %dEr/dz 
dEr_dz = diff_Er_dz(x,W,z); 
% 
%dEr/dW
dEr_dW = diff_Er_dW(x,W,z); 
%dEs/dz 
dEs_dz = diff_Es_dz(z,P); 
%dphase/dz
dphase_dz = diff_phase_dz(z,P);
%dEp/dz 
[pred_loss, moments_error] = Ep(z,P,M1);
dEp_dz = diff_Ep_dz(dphase_dz,moments_error',M1,size(z)); 

if numerical_check == true 
    
    disp('checking gradients via central difference...') 

    %numerical check 1
    dEr_dz_num = zeros(size(z)); 
    for n = 1:3 
        for ii = 1:size(z,1)
            for jj = 1:size(z,2) 
                zr = z(:,:,n); 
                zl = z(:,:,n);
                zr(ii,jj) = zr(ii,jj) + dx; 
                zl(ii,jj) = zl(ii,jj) - dx; 
                dEr_dz_num(ii,jj,n) = (Er(x(:,:,n),W,zr) - Er(x(:,:,n),W,zl))/(2*dx); 
            end 
        end
    end
    err = dEr_dz - dEr_dz_num; 
    assert(sum(abs(err(:)))/sum(abs(dEr_dz(:))) < 1e-6); 

    %numerical check 2
    dEr_dW_num = zeros(size(W));  
    for n = 1:3 
        for ii = 1:size(W,1) 
            for jj = 1:size(W,2) 
                Wr = W; 
                Wl = W;
                Wr(ii,jj) = Wr(ii,jj) + dx; 
                Wl(ii,jj) = Wl(ii,jj) - dx; 
                dEr_dW_num(ii,jj) = dEr_dW_num(ii,jj) + ((Er(x(:,:,n),Wr,z(:,:,n)) - Er(x(:,:,n),Wl,z(:,:,n)))/(2*dx)); 
            end
        end
    end
    err = dEr_dW - dEr_dW_num; 
    assert(sum(abs(err(:)))/sum(abs(dEr_dW(:))) < 1e-6); 
 
    %numerical check 3
    dEs_dz_num = zeros(size(z)); 
    for n = 1:3 
        for ii = 1:size(z,1)
            for jj = 1:size(z,2) 
                zr = z; 
                zl = z;
                zr(ii,jj,n) = zr(ii,jj,n) + dx; 
                zl(ii,jj,n) = zl(ii,jj,n) - dx; 
                dEs_dz_num(ii,jj,n) = (Es(zr,P1) - Es(zl,P1))/(2*dx); 
            end 
        end
    end
    err = dEs_dz - dEs_dz_num; 
    assert(sum(abs(err(:)))/sum(abs(dEs_dz_num(:))) < 1e-6); 
    
    %numerical check 4
    dphase_dz_num = zeros(numel(z(:,:,1)),numel(z(:,:,1)),3);
    for n = 1:3
        zi = z(:,:,n); 
        zi = zi(:); 
        for ii = 1:numel(z(:,:,1))
            for jj = 1:numel(z(:,:,1))
                zr = zi; 
                zl = zi;
                zr(ii) = zr(ii) + dx; 
                zl(ii) = zl(ii) - dx;
                pr = zr./(P*zr + eps*ones(size(zr))); 
                pl = zl./(P*zl + eps*ones(size(zl)));
                dphase_dz_num(jj,ii,n) = (pr(jj) - pl(jj))/(2*dx); 
            end 
        end
    end

    err = dphase_dz - dphase_dz_num; 
    assert(sum(abs(err(:)))/sum(abs(dphase_dz(:))) < 5e-5); %unstable deravative for innactive groups 

    %numerical check 5
        dEp_dz_num = zeros(size(z));
        for n = 1:3
            for ii = 1:size(z,1)
                for jj = 1:size(z,2)
                    zr = z; 
                    zl = z;
                    zr(ii,jj,n) = zr(ii,jj,n) + dx; 
                    zl(ii,jj,n) = zl(ii,jj,n) - dx;
                    dEp_dz_num(ii,jj,n) = (Ep(zr,P,M1) - Ep(zl,P,M1))/(2*dx); 
                end 
            end
        end
    
    err = dEp_dz - dEp_dz_num;
    assert(sum(abs(err(:)))/sum(abs(dEp_dz(:))) < 5e-5)
    
    
    disp('passed!')

end 

% % Training 













