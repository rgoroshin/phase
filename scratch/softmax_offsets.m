clc; 
clear all; 
close all; 

SoftMax = @(x,beta) exp(beta*x)/sum(exp(beta*x)); 
ReLU = @(x) x.*(x>0); 

%target x y (xi = -1, yi = -1)
xc = 1; 
yc = 1; 

n = 11; 
beta = 10.0;
dt = 0.005;
ticklabels = round((-1:2/(n-1):1)*10)/10;
ticks = linspace(1, n, numel(ticklabels));

gradX = repmat(-1:2/(n-1):1,[n,1]); 
gradY = gradX'; 
gradX = gradX(:); 
gradY = gradY(:); 

SMZave = zeros(n*n,1); 

for k = 1:100 

disp(k) 
    
Z = ReLU(randn(n*n,1));
% Z = Z - min(Z(:)); 


    for i = 1:100

        SMZ = SoftMax(Z(:),beta); 

        %current x,y
        x = SMZ'*gradX;
        y = SMZ'*gradY;

        %gradient w.r.t. Z (input) 
        dSMZ = beta*(diag(SMZ) - repmat((SMZ.^2),[1,size(SMZ)])); 
        dZ = (x-xc)*dSMZ*gradX + (y-yc)*dSMZ*gradY; 
        Z = Z-dt*dZ; 

        subplot(1,2,1);  
        imagesc(reshape(Z,[n,n]));
        title('Z');
        set(gca, 'XTick', ticks, 'XTickLabel', ticklabels)
        set(gca, 'YTick', ticks, 'YTickLabel', ticklabels)
    
        subplot(1,2,2); 
        imagesc(reshape(SMZ,[n,n])); 
        title('SoftMax(Z)'); 
        set(gca, 'XTick', ticks, 'XTickLabel', ticklabels)
        set(gca, 'YTick', ticks, 'YTickLabel', ticklabels)

        err = sqrt((x-xc)^2 + (y-yc)^2); 
        disp(err); 
        pause(0.1); 

    end
    
    SMZave = SMZave + SMZ;

end

imagesc(reshape(SMZave,[n,n])); 

