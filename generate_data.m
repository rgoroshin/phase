clc; 
clearvars -except 'X' 'IX'; 
close all; 

options = struct('ntemplates',2); 
tic() 
X=generate_jitter_data2d(options);

IX = reshape(X,[16,16,1,2^17]); 

imdisp(IX(:,:,:,3000:3100),'Border',[0.1 0.1]); 