%% Checks the gradients of the various terms in the loss using central difference 
clc; 
clear all; 
close all; 

%% Constants 
%dimensions 
insz = 16*16; %patch size
codesz = 16; %code size 
poolsz = 4; %pool group size
outsz = codesz/poolsz; %output size 
assert(rem(outsz,1)==0,'outsz is not an integer'); 
dx = 1e-6; %step size for numerical check

%loss multipliers 
wl1 = 0.5; 
ws = 0.5; 
wp = 0.5; 
%% 

%% Definitions and Initialization 
%ReLU 
ReLU = @(x) x.*(x>0); 

%input data 
x = rand(insz,3) - 0.5; 

%codes for three temporal samples 
z = ReLU(rand(codesz,3) - 0.5); 

%normalized decoder dictionary 
W = rand(insz,codesz) - 0.5;
W = W./repmat(sqrt(sum(W.^2,1)),[insz,1]); 

%pooling matrix 
r = zeros(1,codesz); 
r(1,1:poolsz) = 1; 
P = zeros(outsz,codesz); 
for ii = 1:outsz
    P(ii,:) = r; 
    r = circshift(r,[1 poolsz]); 
end
P = P'*P; 

%check that: z = ((P'*m).*s)
assert(norm(z-(P*z + eps*ones(size(z))).*(z./(P*z + eps*ones(size(z))))) < 1e-10); 

%first moment matrix (define coordinates from -1 to +1)  
r = zeros(1,codesz); 
r(1:poolsz) = linspace(-1,1,poolsz); 
M1 = zeros(outsz,codesz);
for ii = 1:outsz
    M1(ii,:) = r; 
    r = circshift(r,[1 poolsz]); 
end

%L1-sparsity-error 
El1 = @(z) sum(z(:)); 

%L2-slowness-error 
Es = @(mag) 0.5*sum((mag(:,2)-mag(:,1)).^2+(mag(:,3)-mag(:,2)).^2); 

%L2-Prediction-error 
Ep = @(moments) 0.5*sum((moments(:,3) - 2*moments(:,2) + moments(:,1)).^2);  
 
%L2-reconstruction-error 
Er = @(x,W,z) sum(0.5*(sum((x - W*z).^2,1))); 

%Loss 
L = Er(x,W,z) + wp*Ep(M1*(z./(P*z + eps*ones(size(z))))) + ws*Es((P*z + eps*ones(size(z)))) + wl1*El1(z);
% L = Er(x,W,z) + wp*Ep(M1*(z./(P*z + eps*ones(size(z))) + ws*Es((P*z + eps*ones(size(z)))) + wl1*El1(z);


%% Learning

%dEr/dz
dErdz = @(x,W,z) -W'*(x - W*z); 

%numerical check 
dErdz_num = zeros(size(z)); 
for ii = 1:size(z,1)
    for jj = 1:size(z,2) 
        zr = z; 
        zl = z;
        zr(ii,jj) = zr(ii,jj) + dx; 
        zl(ii,jj) = zl(ii,jj) - dx; 
        dErdz_num(ii,jj) = (Er(x,W,zr) - Er(x,W,zl))/(2*dx); 
    end 
end
assert(norm(dErdz(x,W,z) - dErdz_num) < 1e-6); 

%dEr/dW
dErdW = @(x,W,z) (x - W*z)*(-z'); 

%numerical check 
dErdW_num = zeros(size(W));  
for ii = 1:size(W,1) 
    for jj = 1:size(W,2) 
        Wr = W; 
        Wl = W;
        Wr(ii,jj) = Wr(ii,jj) + dx; 
        Wl(ii,jj) = Wl(ii,jj) - dx; 
        dErdW_num(ii,jj) = (Er(x,Wr,z) - Er(x,Wl,z))/(2*dx); 
    end
end
assert(norm(dErdW(x,W,z) - dErdW_num) < 1e-6); 

%dEs/dmag = (dEs/dmag)(dmag/dz)
dEsdz = @(P,z) P * (P*z + eps*ones(size(z))) * [1 -1  0; -1  2 -1; 0 -1  1];

%numerical check 
dEsdz_num = zeros(size(z)); 
for ii = 1:size(z,1)
    for jj = 1:size(z,2) 
        zr = z; 
        zl = z;
        zr(ii,jj) = zr(ii,jj) + dx; 
        zl(ii,jj) = zl(ii,jj) - dx; 
        dEsdz_num(ii,jj) = (Es((P*zr + eps*ones(size(zr)))) - Es((P*zl + eps*ones(size(zl)))))/(2*dx); 
    end 
end
assert(norm(dEsdz(P,z) - dEsdz_num) < 1e-6); 

%d(z./(P*z))/dz
% diff_phase_dz = @(P,zi)(P*diag((mag(P,zi)).^-2)).*(diag(P*zi) - diag(zi)*P); 
% dphase_dz = zeros(codesz,codesz,3); 
% dphase_dz(:,:,1) = diff_phase_dz(P,z(:,1)); 
% dphase_dz(:,:,2) = diff_phase_dz(P,z(:,2)); 
% dphase_dz(:,:,3) = diff_phase_dz(P,z(:,3)); 
% 
% %dEp/dz 
% dEp_dz = zeros(size(z)); 
% moments = M1*(z./(P*z + eps*ones(size(z))));
% moments_err = (moments(:,3) - 2*moments(:,2) + moments(:,1))'; 
% 
% dEp_dz(:,1) = moments_err*M1*dphase_dz(:,:,1); 
% dEp_dz(:,2) = -2*moments_err*M1*dphase_dz(:,:,2); 
% dEp_dz(:,3) = moments_err*M1*dphase_dz(:,:,3); 

dEp_dz = diff_Ep_dz(P,M1,z); 

% numerical check 
dEp_dz_num = zeros(size(z)); 
for ii = 1:size(z,1)
    for jj = 1:size(z,2) 
        zr = z; 
        zl = z;
        zr(ii,jj) = zr(ii,jj) + dx; 
        zl(ii,jj) = zl(ii,jj) - dx; 
        dEp_dz_num(ii,jj) = (Ep(M1*(zr./(P*zr + eps*ones(size(zr))))) - Ep(M1*(zl./(P*zl + eps*ones(size(zl))))))/(2*dx); 
    end 
end
assert(norm(dEp_dz - dEp_dz_num) < 1e-6); 















