classdef Tensor < handle
    properties
        A
    end
    methods
        function obj = Tensor(T)
            if nargin == 1
                obj.A = T;
            end
        end
        function set(obj, idx, val)
            if numel(idx) ~= obj.rank()
                error(['Received ', num2str(numel(idx)), ' indices for tensor of rank ', num2str(obj.rank())]);
            end
            
            while numel(idx) < numel(size(obj.A))
                idx = cat(2, idx, 1);
            end
            
            if min(idx <= size(obj.A)) == 0
                error('Tensor.set received index out of range');
            end
            
            idx = num2cell(idx);
            obj.A(idx{:}) = val;
        end
        function val = get(obj, idx)
            if numel(idx) ~= obj.rank()
                error(['Received ', num2str(numel(idx)), ' indices for tensor of rank ', num2str(obj.rank())]);
            end
            
            while numel(idx) < numel(size(obj.A))
                idx = cat(2, idx, 1);
            end
            
            if min(idx <= size(obj.A)) == 0
                error('Tensor.set received index out of range');
            end
            
            idx = num2cell(idx);
            val = obj.A(idx{:});
        end
        function T = group(obj, idxlist)
            if size(idxlist,1) ~= 1
                error('Expected idxlist to be a row vector of cells');
            end
            
            % Determine the dimesnions of the new tensor
            dims = size(obj.A);
            
            tdims = ones(size(idxlist));
            for ii=1:size(tdims,2)
                group_indices = idxlist{ii};
                for jj=1:size(group_indices,2)
                    tdims(ii) = tdims(ii)*dims(group_indices(jj));
                end
            end
            
            if isscalar(tdims)
                tdims = [tdims,1];
            end
            
            % Create and populate the values of the new tensor
            T = Tensor(zeros(tdims));
            
            iter = IndexIter(tdims);
            while ~iter.end()
                origidx = zeros(size(dims));
                for ii=1:size(idxlist,2)
                    group_idx = grouped_to_split(iter.curridx(ii), dims(idxlist{ii}));
                    origidx(idxlist{ii}) = group_idx;
                end
                origidx = num2cell(origidx);
                newidx = num2cell(iter.curridx);
                T.A(newidx{:}) = obj.A(origidx{:});
                iter.next();
            end
        end
        function T = split(obj, idxlist)
            % SPLIT: Create a new tensor splitting this tensor's indices
            % according to idxlist
            %
            % idxlist{i} = Destination for the ith index in the new tensor.
            % If a scalar is provided, it will indicate the destination
            % index. Otherwise, a matrix with two rows is expected. The
            % first row is the destination indices for the split, while the
            % second is the dimensions
            
            if size(idxlist,1) ~= 1
                error('Expected idxlist to be a row vector of cells');
            end
            
            % Determine the dimesnions of the new tensor
            dims = size(obj.A);
            
            tdims = [];
            for ii=1:size(idxlist,2)
                split_indices = idxlist{ii};
                split_size = size(split_indices);
                
                if ~isequal(split_size, [1,1]) && split_size(1) ~= 2
                    error('Split idxlist value must be a scalar or matrix with two rows');
                end
                
                if ~isscalar(split_indices) && prod(split_indices(2,:)) ~= dims(ii)
                    error('Product of split dimensions does not equal original dimension');
                end
                
                if isscalar(split_indices)
                    tdims(split_indices) = dims(ii); %#ok<AGROW>
                else
                    tdims(split_indices(1,:)) = split_indices(2,:); %#ok<AGROW>
                end
            end
            
            if isscalar(tdims)
                tdims = [tdims,1];
            end
            
            % Create and populate the values of the new tensor
            T = Tensor(zeros(tdims));
            
            iter = IndexIter(dims);
            while ~iter.end()
                newidx = zeros(size(tdims));
                for ii=1:size(idxlist,2)
                    if isscalar(idxlist{ii})
                        newidx(idxlist{ii}) = iter.curridx(ii);
                    else
                        newidx(idxlist{ii}(1,:)) = grouped_to_split(iter.curridx(ii), idxlist{ii}(2,:));
                    end
                end
                newidx = num2cell(newidx);
                origidx = num2cell(iter.curridx);
                
                T.A(newidx{:}) = obj.A(origidx{:});
                
                iter.next();
            end
        end
        function [TU,TS,TV] = svd(obj)
            if obj.rank() ~= 2
                error('SVD can only be performed on a rank 2 tensor');
            end
            
            [u,s,v] = svd(obj.A, 'econ');
            
            TU = Tensor(u);
            TS = Tensor(s);
            TV = Tensor(v);
        end
        function T = conjugate(obj)
            T = Tensor(conj(obj.A));
        end
        function C = contract(obj, T, indices)
            r1 = ndims(obj.A);
            r2 = ndims(T.A);
            
            if min(r1 >= indices(:,1)) == 0 || min(r2 >= indices(:,2)) == 0
                error('Index exceeds tensor rank');
            end
            
            s1 = size(obj.A);
            s2 = size(T.A);
            if norm(s1(indices(:,1)') - s2(indices(:,2)')) > 0
                error('Contracted index dimension mismatch');
            end
            cdim = s1(indices(:,1)');
            
            s1(indices(:,1)) = [];
            s2(indices(:,2)) = [];
            csize = cat(2,s1,s2);
            if min(size(csize)) == 0
                csize = 1;
            end
            C = Tensor(zeros(csize));
            
            iter = IndexIter(csize);
            
            while ~iter.end()
                citer = IndexIter(cdim);
                while ~citer.end()
                    iteridx = 1;
                    citeridx = 1;
                    idx1 = [];
                    for ii=1:ndims(obj.A)
                        if min(size(find(indices(:,1) == ii))) == 0
                            idx1 = cat(2, idx1, iter.curridx(iteridx));
                            iteridx = iteridx + 1;
                        else
                            idx1 = cat(2, idx1, citer.curridx(citeridx));
                            citeridx = citeridx + 1;
                        end
                    end
                    idx1 = num2cell(idx1);
                    
                    citeridx = 1;
                    idx2 = [];
                    for ii=1:ndims(T.A)
                        if min(size(find(indices(:,2) == ii))) == 0
                            idx2 = cat(2, idx2, iter.curridx(iteridx));
                            iteridx = iteridx + 1;
                        else
                            idx2 = cat(2, idx2, citer.curridx(citeridx));
                            citeridx = citeridx + 1;
                        end
                    end
                    idx2 = num2cell(idx2);
                    
                    idx3 = num2cell(iter.curridx);
                    C.A(idx3{:}) = C.A(idx3{:}) + obj.A(idx1{:})*T.A(idx2{:});
                    
                    citer.next();
                end
                
                iter.next();
            end
        end
        function C = trace(obj, indices)
            r = ndims(obj.A);
            
            if min(r >= indices(:,1)) == 0 || min(r >= indices(:,2)) == 0
                error('Index exceeds tensor rank');
            end
            
            s = size(obj.A);
            if norm(s(indices(:,1)') - s(indices(:,2)')) > 0
                error('Contracted index dimension mismatch');
            end
            cdim = s(indices(:,1)');
            
            s(reshape(indices, 1, [])) = [];
            csize = s;
            if min(size(csize)) == 0
                csize = 1;
            end
            C = Tensor(zeros(csize));
            
            iter = IndexIter(csize);
            
            while ~iter.end()
                citer = IndexIter(cdim);
                while ~citer.end()
                    iteridx = 1;
                    idx1 = [];
                    for ii=1:ndims(obj.A)
                        leftidx = find(indices(:,1) == ii);
                        rightidx = find(indices(:,2) == ii);
                        if isscalar(leftidx)
                            idx1 = cat(2, idx1, citer.curridx(leftidx));
                        elseif isscalar(rightidx)
                            idx1 = cat(2, idx1, citer.curridx(rightidx));
                        else
                            idx1 = cat(2, idx1, iter.curridx(iteridx));
                            iteridx = iteridx + 1;
                        end
                    end
                    idx1 = num2cell(idx1);
                    
                    idx3 = num2cell(iter.curridx);
                    C.A(idx3{:}) = C.A(idx3{:}) + obj.A(idx1{:});
                    
                    citer.next();
                end
                
                iter.next();
            end
        end
        function d = rank(obj)
            s = size(obj.A);
            if min(s(:)) == 0
                d = -1;
            elseif length(s) > 2
                d = length(s);
            else
                d = 2;
                if s(2) == 1
                    d = d - 1;
                    
                    if s(1) == 1
                        d = d - 1;
                    end
                end
            end
        end
        function d = dims(obj)
            if isscalar(obj.A)
                d = 0;
                return
            end
            
            d = size(obj.A);
            if numel(d) == 2 && d(2) == 1
                d(2) = [];
            end
        end
        function d = dim(obj, idx)
            dv = obj.dims();
            d = dv(idx);
        end
        function A = matrix(obj)
            A = obj.A;
        end
        function e = equals(obj, T, tol)
            e = (obj.rank() == T.rank());
            if ~e
                return;
            end
            D = abs(obj.A - T.A);
            e = (max(D(:)) < tol);
        end
    end
end

function orig_indices = grouped_to_split(grouped_idx, orig_dim)
if size(orig_dim,2) == 1
    orig_indices = grouped_idx;
    return
end

grouped_idx = grouped_idx - 1;
orig_indices = zeros(size(orig_dim));
val = prod(orig_dim);
for ii=size(orig_dim,2):-1:1
    val = val/orig_dim(ii);
    orig_indices(ii) = floor(grouped_idx/val) + 1;
    grouped_idx = grouped_idx - (orig_indices(ii)-1)*val;
end
end