function mps_out = measure_prod(mps, op)
% measure_prod: Perform a measurement of op at each site of mps and return
% the state of the system after the measurements as a rank one MPS

tol = 1e-12;

[evecs, ~] = eig(op);

n = mps.num_sites();
pdim = size(op, 1);

% Iterate over the sites, performing the measurement for the operator at
% that site
ms_out = cell(1,n);

% Make a copy since we will be modifying the MPS in place
mps = mps.substate(1:n);
mps2 = mps.substate(1:n);
for ii=1:n
    if mps.tensors{ii}.dim(3) ~= pdim
        error(['Physical dimension of operator: ', num2str(pdim), ' does not match MPS tensors: ', num2str(mps.tensors{ii}.dim(3))]);
    end
    
    % Compute the probability of each eigenvector
    % TODO - By using the inner method, this computes a full contraction
    % across all sites every time. This would be much faster if you assume
    % a left/right normal form and then just build the contraction one site
    % at a time like in DMRG or TDVP
    pdf = zeros(1,pdim);
    cdf = zeros(1,pdim);
    for jj=1:pdim
        T = Tensor(evecs(:,jj)*evecs(:,jj)');
        mps.set_tensor(ii,mps.tensors{ii}.contract(T,[3,2]));
        prob = mps2.inner(mps);
        if jj == 1
            cdf(jj) = prob;
        else
            cdf(jj) = cdf(jj-1) + prob;
        end
        pdf(jj) = prob;
        
        % Reset the mps tensor for this site before moving on to the next
        mps.set_tensor(ii,Tensor(mps2.tensors{ii}.matrix()));
    end
    
    if abs(cdf(end) - 1) > tol
        error('Measurement CDF does not end with 1');
    end
    
    % Make the measurement for this site
    eps = rand();
    for jj=1:pdim
        if eps < cdf(jj)
            idx = jj;
            break;
        end
    end
    
    % Create the tensor for the new product state
    ms_out{ii} = Tensor(reshape(evecs(:,idx), 1, 1, []));
    
    % Project the MPS based on the outcome of the measurement
    T = Tensor(evecs(:,idx)*evecs(:,idx)');
    T2 = mps.tensors{ii}.contract(T,[3,2]);
    T2.mult_eq(pdf(idx)^(-1/2));
    mps.set_tensor(ii,T2);
    mps2.set_tensor(ii,T2);
    
    % TODO - Temporary
    if abs(mps.inner(mps) - 1) > tol
        error('Failed to normalize the state');
    end
end

mps_out = MPS(ms_out);