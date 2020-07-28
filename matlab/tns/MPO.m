classdef MPO < handle
    properties
        tensors
        obc
    end
    methods
        function obj = MPS(tensors)
            if size(tensors,1) ~= 1
                error('Expected row cell array of tensors');
            end
            
            obj.tensors = tensors;
            
            if tensors{end}.rank() == 3
                obj.obc = 1;
            else
                obj.obc = 0;
            end
            
            % Validate the incoming tensors
            n = size(tensors,2);
            for ii=1:n
                T = tensors{ii};
                if ii < n || obj.obc ~= 1
                    if T.rank() ~= 4
                        error(['Expected rank 4 tensor at site ', num2str(ii)]);
                    end
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