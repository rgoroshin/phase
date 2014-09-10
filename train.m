% Checks the gradients of the various terms in the loss using central difference 
clc; 
clearvars -except 'X'; 
close all; 
addpath ./fprop_bprop/

% Constants 
%dimensions 
insz = 16*16; %patch size
codesz = 16;%2*insz; %code size 
poolsz = 4; %pool group size
poolst = 4; %pool group stride 
bsz = 8; %batch size 
L = 100; 
outsz = codesz/poolsz; %output size 
assert(rem(outsz,1)==0,'outsz is not an integer'); 
numerically_check_gradients = true; 
display_sample = true; 
dx = 1e-6; %step size for numerical check
%loss multipliers 
wL1 = 0.0; 
ws = 0.0; 
wp = 0.0; 
% Generate toy dataset 
if ~exist('X')  %#ok<EXIST>
    disp('generating dataset...'); 
    options = struct('ntemplates',2,'L',L); 
    X=generate_jitter_data2d(options);
    X = X - repmat(mean(X,2),[1 size(X,2)]); 
    disp('done!'); 
end
batch = zeros(insz,bsz,3);
order = im2col(randperm(size(X,2)-2),[1,bsz]); 
batch = get_batch(X,order(:,1),batch); 
if display_sample == true
   n = 1; 
   Ibatch = reshape(batch, [16 16 1 bsz 3]); 
   imdisp(Ibatch(:,:,:,2,:),'Border',[0.1 0.1]); 
end
% Loss Definition and Initialization 
% ReLU 
ReLU = @(x,lambda) (x-lambda).*((x-lambda)>0); 

%data sample 
x = batch; 

%codes for three temporal samples 
z = ReLU(rand(codesz,bsz,3),0.5); 

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
    r = circshift(r,[1 poolst]); 
end
P = P1'*P1; 

%first moment matrix (define coordinates from -1 to +1)  
r = zeros(1,codesz); 
r(1:poolsz) = linspace(-1,1,poolsz); 
M1 = zeros(outsz,codesz);
for ii = 1:outsz
    M1(ii,:) = r; 
    r = circshift(r,[1 poolst]); 
end

%mini-batches
Pb = P;  
P1b = P1; 
M1b = M1; 
for k = 1:bsz-1
    Pb = blkdiag(Pb,P);
    P1b = blkdiag(P1b,P1);
    M1b = blkdiag(M1b,M1);
end
P = Pb; 
P1 = P1b; 
M1 = M1b; 

if numerically_check_gradients == true 
      
    %dEr/dW
    dEr_dz = diff_Er_dz(x,W,z);  
    %dEr/dW
    dEr_dW = diff_Er_dW(x,W,z); 
    %dEs/dz 
    dEs_dz = diff_Es_dz(z,P); 
    %dEp/dz 
    dEp_dz = diff_Ep_dz(z,P,M1); 
    
    disp('checking gradients via central difference...') 
    check_gradients(x,z,W,P,P1,M1,dEr_dz,dEr_dW,dEs_dz,dEp_dz,dx)
    disp('passed!')

end 

% Training 
epochs = 100; 
infer_steps = 20; 
zstep = 0.01; 
Wstep = 0.05; 

for iter = 1:epochs 
    
    Loss = 0; 
    
    for n = 1:size(order,2)
        
         waitbar(n/size(order,2)); 
       
         x = get_batch(X,order(:,n),x); 
         
         RecCost = 2*Er(x,W,z)/(norm(x(:))^2); 
%          L1Cost = sum(abs(z(:))); 
%          SlowCost = Es(z,P1);
%          PredCost = Ep(z,P,M1);  
         
         Loss_batch = RecCost; %+ wL1*L1Cost + ws*SlowCost + wp*PredCost; 
         Loss = Loss + Loss_batch; 
         
         %inference (z-update) 
         z = zeros(codesz,bsz,3); 
         
         for ii = 1:infer_steps 
            
%              dEr_dz = diff_Er_dz(x,W,z);
%              dEs_dz = diff_Es_dz(z,P);
%              dEp_dz = diff_Ep_dz(z,P,M1);            
             dz = diff_Er_dz(x,W,z);%+ ws*dEs_dz + wp*dEp_dz;  
%              z = ReLU(z - zstep*dz, zstep*wL1); 
             z = z - zstep*dz; 
             
         end
         
         %dictionary update 
         dW = diff_Er_dW(x,W,z);
         W = W - Wstep*dW; 
         W = normalize_cols(W,insz); 
        
    end
    
    Iw = reshape(W,[sqrt(insz),sqrt(insz),1,codesz]); 
    figure(1); 
    imdisp(Iw,'Border',[0.1 0.1])
    disp(['Loss = ' num2str(Loss/L)])
    
end


% imdisp(Ibatch(:,:,:,2,:),'Border',[0.1 0.1]); 









