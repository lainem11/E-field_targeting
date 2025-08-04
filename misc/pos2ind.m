function inds = pos2ind(pos,mesh)
% Selects nearest mesh vertex to the input position
% INPUTS:
%           -pos: Row-wise matrix of 3D coordinates
%           -mesh: Struct with vertex coordinates
inds = zeros(size(pos,1),1);
for i = 1:size(pos,1)
[~,inds(i)] = min(sqrt(sum((mesh.vertices-pos(i,:)).^2,2)));
end
end