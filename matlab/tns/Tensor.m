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
        function T = group(obj, idxlist)
            if size(idxlist,1) ~= 1
                disp('ERROR - Expected idxlist to be a row vector of cells');
                T = Tensor();
                return
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
            
            % Create and populate the values of the new tensor
            if size(tdims, 2) == 1
                T = Tensor(zeros(tdims,1));
            else
                T = Tensor(zeros(tdims));
            end
            iter = IndexIter(tdims);
            while ~iter.end()
                origidx = zeros(size(dims));
                for ii=1:size(iter.curridx,2)
                    group_idx = grouped_to_orig(iter.curridx(ii), dims(idxlist{ii}));
                    origidx(idxlist{ii}) = group_idx;
                end
                origidx = num2cell(origidx);
                newidx = num2cell(iter.curridx);
                T.A(newidx{:}) = obj.A(origidx{:});
                iter.next();
            end
        end
        function T = split(obj, idxlist)
            % TODO
        end
        function [TU,TS,TV] = svd(obj)
            if obj.rank() ~= 2
                disp('ERROR - SVD can only be performed on a rank 2 tensor');
                return
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
                disp('ERROR - Index exceeds tensor rank');
                return
            end
            
            s1 = size(obj.A);
            s2 = size(T.A);
            if norm(s1(indices(:,1)') - s2(indices(:,2)')) > 0
                disp('ERROR - Contracted index dimension mismatch');
                return
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

function orig_indices = grouped_to_orig(grouped_idx, orig_dim)
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