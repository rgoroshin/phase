function [ dEp_dz ] = diff_Ep_dz(z,P,M1)
%computes gradient of linear prediction of first moment of phase w.r.t. activations 

    function [ ph dph_dz ] = diff_phase_dz(z,P)
    %computes gradient of the phase w.r.t. activations 

        diff_phase_dzi = @(P,zi,Pzi)(P*diag((Pzi + eps*ones(size(zi))).^-2)).*(diag(Pzi) - diag(zi)*P); 
     
        dph_dz = zeros(numel(z(:,:,1)),numel(z(:,:,1)),3); 
        ph = zeros(numel(z(:,:,1)),3); 
        
        for ii = 1:3 
            zi = z(:,:,ii); 
            zi = zi(:); 
            Pzi = P*zi; 
            ph(:,ii) = zi./(P*zi + eps*ones(size(zi))); 
            dph_dz(:,:,ii) = diff_phase_dzi(P,zi,Pzi); 
        end

    end

    [ph dph_dz] = diff_phase_dz(z,P); 
    
    moments = M1*ph; 
    moments_error = (moments(:,3) - 2*moments(:,2) + moments(:,1))'; 

    dEp_dz = [moments_error*M1*dph_dz(:,:,1); -2*moments_error*M1*dph_dz(:,:,2); moments_error*M1*dph_dz(:,:,3)]'; 
    dEp_dz = reshape(dEp_dz,size(z));    
 
end

