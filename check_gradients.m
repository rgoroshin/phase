function [] = check_gradients(x,z,W,P,P1,M1,dEr_dz,dEr_dW,dEs_dz,dEp_dz,dx)
%check gradients using central difference 

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

end

