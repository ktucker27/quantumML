function M = povm(type)
% povm: Generate the POVM matrices in the return tensor s.t. M(:,:,k) is
% the kth POVM operator. Options for POVM type are:
% tetra
% pauli4

% Get Pauli xyz
[~, ~, sz, sx, sy] = local_ops(2);
sx = 2*sx;
sy = 2*sy;
sz = 2*sz;

if strcmp(type, 'tetra')
    s = (1/3)*[0,0,3; 2*sqrt(2),0,-1; -sqrt(2),sqrt(6),-1; -sqrt(2),-sqrt(6),-1];
    M = zeros(2,2,4);
    for k=1:4
        M(:,:,k) = (1/4)*(eye(2) + s(k,1)*sx + s(k,2)*sy + s(k,3)*sz);
    end
elseif strcmp(type, 'pauli4')
    M = zeros(2,2,4);
    M(:,:,1) = (1/3)*[1;0]*[1,0];
    M(:,:,2) = (1/6)*[1;1]*[1,1];
    M(:,:,3) = (1/6)*[1;1i]*[1,-1i];
    M(:,:,4) = eye(2) - sum(M,3);
elseif strcmp(type, 'tmp')
    M = zeros(2,2,4);
    M(:,:,1) = 1*[1;0]*[1,0];
    M(:,:,2) = (1/250)*[1;1]*[1,1];
    M(:,:,3) = (1/250)*[1;1i]*[1,-1i];
    M(:,:,4) = eye(2) - sum(M,3);
else
    error('Unrecognized POVM. See help for options');
end