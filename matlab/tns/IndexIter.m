classdef IndexIter < handle
    properties
        dim
        curridx
    end
    methods
        function obj = IndexIter(dim)
            obj.dim = dim;
            obj.curridx = ones(size(obj.dim));
        end
        function val = end(obj)
            val = (norm(obj.curridx - obj.endidx()) == 0);
        end
        function idx = endidx(obj)
            idx = zeros(size(obj.dim));
        end
        function next(obj)
            for ii=1:size(obj.curridx,2)
                obj.curridx(ii) = obj.curridx(ii) + 1;
                if obj.curridx(ii) <= obj.dim(ii)
                    break;
                end
                
                if ii == size(obj.curridx,2)
                    obj.curridx = obj.endidx();
                else
                    obj.curridx(ii) = 1;
                end
            end
        end
    end
end