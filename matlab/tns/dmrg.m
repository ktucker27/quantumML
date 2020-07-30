function mps_out = dmrg(mpo, mps, tol, maxit)

n = mps.num_sites();

ms = mps.substate(1:n);
msd = mps.dagger();

% Initialize the R list
R = cell(1,n);
R{n} = Tensor(1,3);
T = mpo.tensors{n}.contract(ms.tensors{n}, [3,3]);
T = T.contract(msd.tensors{n}, [3,3]);
R{n-1} = T.split({2,5,1,4,3,6});
for ii=n-2:-1:1
    T = R{ii+1}.contract(ms.tensors{ii+1}, [1,2]);
    T = T.contract(mpo.tensors{ii+1}, [1,2;7,3]);
    T = T.contract(msd.tensors{ii+1}, [1,2;7,3]);
    R{ii} = T.split({4,5,6,1,2,3});
end

% Initialize the L list
L = cell(1,n);
L{1} = Tensor(1,3);

startidx = 1;
idxinc = 1;
endidx = n;

for itidx=1:2*maxit
    % Sweep to the right/left updating tensors
    for ii=startidx:idxinc:endidx
        nextidx = ii + idxinc;
        
        % Contract the MPO state with R
        A = mpo.tensors{ii}.contract(R{ii}.squeeze(), [2,2]);
        
        % Contract the result with L
        A = A.contract(L{ii}.squeeze(), [1,2]);
        
        % Group the tensor into a matrix and get the eigenvector
        mdims = A.dim([5,3,1]);
        A = A.group({[6,4,2],[5,3,1]});
        [evec, ~] = eigs(A.matrix(),1);
        M = Tensor(evec);
        if M.rank() ~= 1
            error(['Expected rank one tensor from eigenvector, got ', num2str(M.rank())]);
        end
        M2 = M.split({[1,2,3;mdims]});
        
        % Left normalize the tensor and update the tensor list
        M2 = M2.group({[1,3],2});
        [TU, TS, TV] = M2.svd();
        new_m = TU.split({[1,3;mdims([1,3])],2});
        
        ms.tensors{ii} = new_m;
        msd.tensors{ii} = new_m.conjugate();
        
        if ii < n
            % Update the next tensor
            new_m = TS.contract(TV.dagger(), [2,1]);
            new_m = new_m.contract(ms{ii+1},[2,1]);
            ms.tensors{nextidx} = new_m;
            msd.tensors{nextidx} = new_m.conjugate();
        end
        
        if idxinc > 0
            if ii == 2
                % Initialize the L list
                T = mpo.tensors{1}.contract(ms.tensors{1}, [3,3]);
                T = T.contract(msd.tensors{1}, [3,3]);
                L{ii} = T.split({5,2,4,1,6,3});
            elseif ii > 1
                % Update the L list
                T = L{ii-1}.contract(ms.tensors{ii-1}, [1,1]);
                T = T.contract(mpo.tensors{ii-1}, [1,1;7,3]);
                T = T.contract(msd.tensors{ii-1}, [1,1;7,3]);
                L{ii} = T.split({4,5,6,1,2,3});
            end
        else
            if ii == n-1
                T = mpo.tensors{n}.contract(ms.tensors{n}, [3,3]);
                T = T.contract(msd.tensors{n}, [3,3]);
                R{ii} = T.split({2,5,1,4,3,6});
            elseif ii < n
                T = R{ii+1}.contract(ms.tensors{ii+1}, [1,2]);
                T = T.contract(mpo.tensors{ii+1}, [1,2;7,3]);
                T = T.contract(msd.tensors{ii+1}, [1,2;7,3]);
                R{ii} = T.split({4,5,6,1,2,3});
            end
        end
    end
    
    % Compute energy variance to see if we've converged
    mpo_ms = apply_mpo(mpo, ms);
    mpo_mpo_ms = apply_mpo(mpo, mpo_ms);
    var = ms.inner(mpo_mpo_ms) - (ms.inner(mpo_ms))^2;
    
    if var < 0
        error('Found negative energy variance');
    end
    
    if var < tol
        disp(['Converged with energy variance ', num2str(var)]);
        break;
    end

    % Flip the sweep direction
    if idxinc > 0
        startidx = n;
        idxinc = -1;
        endidx = 1;
    else
        startidx = 1;
        idxinc = 1;
        endidx = n;
    end
        
    if itidx >= 2*maxit
        disp('WARNING - Max iterations reached');
    end
end

% Build the MPS from the tensor list
mps_out = MPS(ms.tensors);
