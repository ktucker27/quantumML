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
            val = isequal(obj.curridx, obj.endidx());
        end
        function idx = endidx(obj)
            idx = zeros(size(obj.dim));
        end
        function val = equals(obj, rhs)
            val = isequal(obj.curridx, rhs.curridx);
        end
        function next(obj)
            if obj.end()
                return
            end
            
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
        function reverse_next(obj)
            if obj.end()
                return
            end
            
            for ii=size(obj.curridx,2):-1:1
                obj.curridx(ii) = obj.curridx(ii) + 1;
                if obj.curridx(ii) <= obj.dim(ii)
                    break;
                end
                
                if ii == 1
                    obj.curridx = obj.endidx();
                else
                    obj.curridx(ii) = 1;
                end
            end
        end
    end
end