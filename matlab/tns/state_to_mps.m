function mps = state_to_mps(psi, n, pdim)

if ~isequal(size(psi), [pdim^n,1])
    error('Expected pdim^n column vector for psi');
end

T = Tensor(psi);
% The following splitting assumes the convention that the product basis is
% enumerated with the index on the last site toggling first
C = T.split({[n:-1:1;pdim*ones(1,n)]});

ms = {};
for ii=1:n-1
    if ii == 1
        T2 = C.group({1, 2:n});
    else
        T2 = C.group({[1,2], 3});
    end
    
    [TU, TS, TV] = T2.svd();
    
    ms = cat(2, ms, {TU.split({[1,3;size(TU.A,1)/pdim,pdim],2})});
    
    TSVdagger = TS.contract(TV.conjugate(), [2,2]);
    
    if ii < n-1
        C = TSVdagger.split({1,[2,3;pdim,size(TSVdagger.A,2)/pdim]});
    else
        ms = cat(2, ms, {TSVdagger.split({1,[2,3;1,size(TSVdagger.A,2)]})});
    end
end

mps = MPS(ms);
