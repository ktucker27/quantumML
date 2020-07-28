classdef MPO < handle
    properties
        tensors
    end
    methods
        function obj = MPS(tensors)
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
                
                if size(tensors{ii-1}.A,2) ~= size(T.A,1)
                    error(['Bond dimension mismatch between sites ', num2str(ii-1), ' and ', num2str(ii)]);
                end
            end
        end
    end
end