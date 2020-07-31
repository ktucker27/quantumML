classdef MPO < handle
    properties
        tensors
    end
    methods
        function obj = MPO(tensors)
            % MPO: Class for representing a matrix product operator
            % tensors: A cell row vector of rank 4 tensors indexed as follows
            %      3
            %     _|_
            % 1__|   |__2
            %    |___|
            %      |
            %      4
            if size(tensors,1) ~= 1
                error('Expected row cell array of tensors');
            end
            
            obj.tensors = tensors;
            
            % Validate the incoming tensors
            n = size(tensors,2);
            for ii=1:n
                T = tensors{ii};
                
                if T.rank() ~= 4
                    error(['Expected rank 4 tensor at site ', num2str(ii)]);
                end
                
                if ii == 1
                    continue
                end
                
                if tensors{ii-1}.dim(2) ~= T.dim(1)
                    error(['Bond dimension mismatch between sites ', num2str(ii-1), ' and ', num2str(ii)]);
                end
            end
        end
        function val = eval(obj, sigma1, sigma2)
            if ~isequal(size(sigma1), size(obj.tensors))
                error('Index vector 1 has incorrect rank');
            end
            
            if ~isequal(size(sigma2), size(obj.tensors))
                error('Index vector 2 has incorrect rank');
            end
            
            val = obj.tensors{1}.A(:,:,sigma2(1),sigma1(1));
            for ii=2:size(obj.tensors,2)
                val = val*obj.tensors{ii}.A(:,:,sigma2(ii),sigma1(ii));
            end
            
            val = trace(val);
        end
        function n = num_sites(obj)
            n = size(obj.tensors,2);
        end
        function op = matrix(obj)
            pdim = obj.pdim();
            
            iter1 = IndexIter(pdim(2,:));
            op = zeros(prod(pdim(2,:)),prod(pdim(1,:)));
            ii = 1;
            while ~iter1.end()
                iter2 = IndexIter(pdim(1,:));
                jj = 1;
                while ~iter2.end()
                    op(ii,jj) = obj.eval(iter1.curridx, iter2.curridx);
                    jj = jj + 1;
                    iter2.reverse_next();
                end
                ii = ii + 1;
                iter1.reverse_next();
            end
        end
        function d = pdim(obj)
            % pdim: Returns physical dimensions of the MPO
            % pdim(1,:) = Physical dimension of the state operated on
            % pdim(2,:) = Physical dimension of the returned state
            
            d = zeros(2, obj.num_sites());
            for ii=1:size(d,2)
                d(1,ii) = obj.tensors{ii}.dim(3);
                d(2,ii) = obj.tensors{ii}.dim(4);
            end
        end
    end
end