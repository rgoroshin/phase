function [ dEp_dz ] = diff_Ep_dz(z,moments_error,P,M1)
%computes gradient of linear prediction of first moment of phase w.r.t. activations 

    function [ dphase_dz ] = diff_phase_dz(z,P)
    %computes gradient of the phase w.r.t. activations 

        diff_phase_dzi = @(P,zi,Pzi)(P*diag((Pzi + eps*ones(size(zi))).^-2)).*(diag(Pzi) - diag(zi)*P); 
        dphase_dz = zeros(numel(z(:,:,1)),numel(z(:,:,1)),3); 

        for i = 1:3 
            zi = z(:,:,i); 
            zi = zi(:); 
            Pzi = P*zi; 
            dphase_dz(:,:,i) = diff_phase_dzi(P,zi,Pzi); 
        end

    end

    dphase_dz = diff_phase_dz(z,P); 

    dEp_dz = [moments_error*M1*dphase_dz(:,:,1); -2*moments_error*M1*dphase_dz(:,:,2); moments_error*M1*dphase_dz(:,:,3)]'; 
    dEp_dz = reshape(dEp_dz,size(z));    
 
end

