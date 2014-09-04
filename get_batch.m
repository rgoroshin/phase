function [sample] = get_batch(X,idx,sample)

bsz = size(idx,1); 
sample(:,1:bsz) = X(:,idx); 
sample(:,bsz+1:2*bsz) = X(:,idx+1); 
sample(:,2*bsz+1:3*bsz) = X(:,idx+2); 

end

