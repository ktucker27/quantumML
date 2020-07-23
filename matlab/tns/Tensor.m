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