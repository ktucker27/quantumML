classdef MPS < handle
    properties
        tensors
        obc
    end
    methods
        function obj = MPS(n,bdim,pdim,obc)
            % MPS: Matrix Product State class constructor
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
            
            if obc == 1
                obj.obc = obc;
            else
                obj.obc = 0;
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
            
            obj.tensors = {};
            for ii=1:n
                if ii == 1
                    if obj.obc == 1
                        t = Tensor(zeros(1,bdim(1),pdim(1)));
                    else
                        t = Tensor(zeros(bdim(n-1),bdim(1),pdim(1)));
                    end
                elseif ii == n
                    if obj.obc == 1
                        t = Tensor(zeros(bdim(n-1),1,pdim(n)));
                    else
                        t = Tensor(zeros(bdim(n-1),bdim(1),pdim(n)));
                    end
                else
                    t = Tensor(zeros(bdim(ii-1),bdim(ii),pdim(ii)));
                end
                obj.tensors = cat(2,obj.tensors,t);
            end
        end
        
        function psi = eval(obj,sigma)
            if norm(size(sigma) - size(obj.tensors)) ~= 0
                error('Index vector has incorrect rank');
            end
            
            psi = obj.tensors{1}.A(:,:,sigma(1))*obj.tensors{2}.A(:,:,sigma(2));
            for ii=3:size(obj.tensors,2)
                psi = psi*obj.tensors{ii}.A(:,:,sigma(ii));
            end
            
            if obj.obc ~= 1
                psi = trace(psi);
            end
        end
    end
end