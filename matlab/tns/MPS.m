classdef MPS < handle
    properties
        tensors
    end
    methods
        function obj = MPS(tensors)
            % MPS: Class for representing a matrix product state
            % tensors: A cell row vector of rank 3 tensors indexed as follows
            %     ___
            % 1__|   |__2
            %    |___|
            %      |
            %      3
            if size(tensors,1) ~= 1
                error('Expected row cell array of tensors');
            end
            
            obj.tensors = tensors;
            
            % Validate the incoming tensors
            obj.validate();
        end
        function validate(obj)
            n = size(obj.tensors,2);
            for ii=1:n
                T = obj.tensors{ii};
                
                if T.rank() ~= 3
                    error(['Expected rank 3 tensor at site ', num2str(ii)]);
                end
                
                iim1 = ii-1;
                if ii == 1
                    iim1 = n;
                end
                
                if obj.tensors{iim1}.dim(2) ~= T.dim(1)
                    error(['Bond dimension mismatch between sites ', num2str(iim1), ' and ', num2str(ii)]);
                end
            end
        end
        function set_tensor(obj, ii, T, val)
            if nargin < 4
                val = true;
            end
            
            if T.rank() ~= 3
                error(['Expected rank 3 tensor at site ', num2str(ii)]);
            end
            
            n = obj.num_sites();
            
            iim1 = ii-1;
            if ii == 1
                iim1 = n;
            end
            
            if val && obj.tensors{iim1}.dim(2) ~= T.dim(1)
                error(['Bond dimension mismatch between sites ', num2str(iim1), ' and ', num2str(ii)]);
            end
            
            iip1 = ii+1;
            if ii == n
                iip1 = 1;
            end
            
            if val && obj.tensors{iip1}.dim(1) ~= T.dim(2)
                error(['Bond dimension mismatch between sites ', num2str(ii), ' and ', num2str(iip1)]);
            end
            
            obj.tensors{ii} = T;
        end
        function mps = substate(obj, indices)
            ms = {};
            for ii=1:size(indices,2)
                ms = cat(2, ms, {Tensor(obj.tensors{indices(ii)}.matrix())});
            end
            mps = MPS(ms);
        end
        function mps = dagger(obj)
            ms = {};
            for ii=1:obj.num_sites()
                ms = cat(2, ms, {obj.tensors{ii}.conjugate()});
            end
            mps = MPS(ms);
        end
        function val = inner(obj, mps)
            if mps.num_sites() ~= obj.num_sites()
                error('Inner product attempted between states of different size');
            end
            
            mps = mps.dagger();
            
            n = obj.num_sites();
            T = obj.tensors{1}.contract(mps.tensors{1}, [3,3]);
            for ii=2:n
                T = T.contract(obj.tensors{ii}, [2,1]);
                T = T.contract(mps.tensors{ii}, [3,1;5,3]);
                
                if ii ~= n || T.rank() > 0
                    T = T.split({1,3,2,4});
                end
            end
            
            if T.rank() ~= 0
                T = T.trace([1,2;3,4]);
                
                if T.rank() ~= 0
                    error('Expected a scalar at the end of an inner product');
                end
            end
            
            val = T.matrix();
        end
        function psi = eval(obj,sigma)
            if ~isequal(size(sigma), size(obj.tensors))
                error('Index vector has incorrect rank');
            end
            
            psi = obj.tensors{1}.A(:,:,sigma(1));
            for ii=2:size(obj.tensors,2)
                psi = psi*obj.tensors{ii}.A(:,:,sigma(ii));
            end
            
            psi = trace(psi);
        end
        function n = num_sites(obj)
            n = size(obj.tensors,2);
        end
        function psi = state_vector(obj)
            pdim = obj.pdim();
            
            iter = IndexIter(pdim);
            psi = zeros(prod(pdim),1);
            ii = 1;
            while ~iter.end()
                psi(ii,1) = obj.eval(iter.curridx);
                ii = ii + 1;
                iter.reverse_next();
            end
        end
        function d = pdim(obj)
            d = zeros(1, obj.num_sites());
            for ii=1:size(d,2)
                d(ii) = obj.tensors{ii}.dim(3);
            end
        end
        function e = equals(obj, T, tol)
            n = obj.num_sites();
            e = (n == T.num_sites());
            if ~e 
                return
            end
            
            for ii=1:n
                e = obj.tensors{ii}.equals(T.tensors{ii}, tol);
                if ~e
                    return
                end
            end
            
            e = true;
        end
        function e = is_left_normal(obj, tol)
            e = true;
            for ii=1:obj.num_sites()
                A = zeros(obj.tensors{ii}.dim(2), obj.tensors{ii}.dim(2));
                for jj=1:obj.tensors{ii}.dim(3)
                    A = A + obj.tensors{ii}.A(:,:,jj)'*obj.tensors{ii}.A(:,:,jj);
                end
                
                D = A - eye(obj.tensors{ii}.dim(2));
                if max(abs(D(:))) > tol
                    e = false;
                    break
                end
            end
        end
        function e = is_right_normal(obj, tol)
            e = true;
            for ii=obj.num_sites():-1:1
                A = zeros(obj.tensors{ii}.dim(1), obj.tensors{ii}.dim(1));
                for jj=1:obj.tensors{ii}.dim(3)
                    A = A + obj.tensors{ii}.A(:,:,jj)*obj.tensors{ii}.A(:,:,jj)';
                end
                
                D = A - eye(obj.tensors{ii}.dim(1));
                if max(abs(D(:))) > tol
                    e = false;
                    break
                end
            end
        end
        function left_normalize(obj, tol)
            if nargin < 2
                tol = 0;
            end
            
            for ii=1:obj.num_sites()-1
                mdims = obj.tensors{ii}.dims();
                M = obj.tensors{ii}.group({[1,3],2});
                if tol > 0
                    [TU, TS, TV] = M.svd_trunc(tol);
                else
                    [TU, TS, TV] = M.svd();
                end
                obj.tensors{ii} = TU.split({[1,3;mdims([1,3])],2});
                
                % Update the next tensor
                next_m = TS.contract(TV.conjugate(), [2,2]);
                obj.tensors{ii+1} = next_m.contract(obj.tensors{ii+1},[2,1]);
            end
        end
        function right_normalize(obj, tol)
            if nargin < 2
                tol = 0;
            end
            
            for ii=obj.num_sites():-1:2
                mdims = obj.tensors{ii}.dims();
                M = obj.tensors{ii}.group({1,[2,3]});
                if tol > 0
                    [TU, TS, TV] = M.svd_trunc(tol);
                else
                    [TU, TS, TV] = M.svd();
                end
                obj.tensors{ii} = TV.conjugate().split({[2,3;mdims([2,3])],1});
                
                % Update the next tensor
                next_m = TU.contract(TS, [2,1]);
                next_m = obj.tensors{ii-1}.contract(next_m,[2,1]);
                obj.tensors{ii-1} = next_m.split({1,3,2});
            end
        end
    end
    methods(Static)
        function obj = mps_zeros(n,bdim,pdim,obc)
            % mps_zeros: Build zero Matrix Product State
            %
            % Parameters:
            % n    = Number of sites
            % bdim = Bond dimension. If a scalar, will be the bond
            %        dimension between each pair of consecutive sites.
            %        Otherwise, bdim(i) is the bond dimension between sites
            %        i and i + 1
            % pdim = Physical dimension. A single scalar to use at each
            %        site
            % obc  = If 1, open boundary conditions will be assumed,
            %        otherwise they will be periodic
            
            if n < 2
                error('MPS requires at least 2 sites');
            end
            
            if norm(size(bdim) - ones(1,2)) == 0
                bdim = bdim*ones(1,n-1);
            elseif norm(size(bdim) - [1,n-1]) ~= 0
                error('bdim must be a scalar or 1xn-1 vector');
            end
            
            if norm(size(pdim) - ones(1,2)) == 0
                pdim = pdim*ones(1,n);
            elseif norm(size(pdim) - [1,n]) ~= 0
                error('pdim must be a scalar or 1xn vector');
            end
            
            tensors = {};
            for ii=1:n
                if ii == 1
                    if obc == 1
                        t = Tensor(zeros(1,bdim(1),pdim(1)));
                    else
                        t = Tensor(zeros(bdim(n-1),bdim(1),pdim(1)));
                    end
                elseif ii == n
                    if obc == 1
                        t = Tensor(zeros(bdim(n-1),1,pdim(n)));
                    else
                        t = Tensor(zeros(bdim(n-1),bdim(1),pdim(n)));
                    end
                else
                    t = Tensor(zeros(bdim(ii-1),bdim(ii),pdim(ii)));
                end
                tensors = cat(2,tensors,{t});
            end
            
            obj = MPS(tensors);
        end
    end
end