function inds = pos2ind(pos,vertices)
% Selects nearest mesh vertex to the input position
% INPUTS:
%           -pos: Row-wise matrix of 3D coordinates
%           -vertices: Matrix of vertex coordinates
inds = zeros(size(pos,1),1);
for i = 1:size(pos,1)
[~,inds(i)] = min(sqrt(sum((vertices-pos(i,:)).^2,2)));
end
end