function [mesh, cell_E_matrix, mesh_indices, coil] = read_Efields(headmodel_path,E_path)
addpath(genpath(pwd))
%% Load mesh and calculate face normals
[pos,e] = convert_custom_bin_to_vtk(headmodel_path);
mesh.p = pos;
mesh.e = e+1;

% Step 1: Calculate face normals
numFaces = size(mesh.e, 1);
faceNormals = zeros(numFaces, 3);
for f = 1:numFaces
    v1 = mesh.p(mesh.e(f,1),:);
    v2 = mesh.p(mesh.e(f,2),:);
    v3 = mesh.p(mesh.e(f,3),:);
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    faceNormals(f, :) = cross(edge1, edge2);
end

% Step 2: Initialize vertex normals
numVertices = size(mesh.p, 1);
vertexNormals = zeros(numVertices, 3);

% Accumulate face normals for each vertex
for f = 1:numFaces
    for j = 1:3
        vertexNormals(mesh.e(f, j), :) = vertexNormals(mesh.e(f, j), :) + faceNormals(f, :);
    end
end

% Step 3: Normalize the vertex normals
for v = 1:numVertices
    normValue = norm(vertexNormals(v, :));
    if normValue > 0
        vertexNormals(v, :) = vertexNormals(v, :) / normValue;
    end
end

mesh.nn = vertexNormals;


%% Load E-fields

data = import_Efield_data(E_path);

% Load mesh indices
mesh_indices = data(18);
mesh_indices = str2num(mesh_indices)'+1;

% Load e-field set
str = char(data(16));
E_matrix = reshape(str2num(str),[],5,3);
E_matrix = E_matrix(mesh_indices,:,:);
cell_E_matrix = {};
for i = 1:size(E_matrix,2)
    cell_E_matrix{i} = squeeze(E_matrix(:,i,:))*1e-3;
end

% Read coil position
coil.pos_str = str2num(data(6));
coil.rot_str = str2num(data(4));

va = [-83.8370   52.7484];
ds_ratio = 16;
f=figure(1);clf
f.Position(3:4)=[1600,1000];
tiledlayout(2,3,"TileSpacing","none")

for i = 1:size(E_matrix,2)
    nexttile
    E_plot = zeros(size(pos));
    E_plot(mesh_indices,:) = squeeze(E_matrix(:,i,:));
    E_plot_mag = sqrt(sum(E_plot.^2,2));
    hp = patch('Faces',mesh.e,'Vertices',mesh.p,'FaceVertexCData',E_plot_mag,'FaceColor','interp');
    hold on
    quiver3(downsample(mesh.p(mesh_indices,1),ds_ratio),downsample(mesh.p(mesh_indices,2),ds_ratio),downsample(mesh.p(mesh_indices,3),ds_ratio),downsample(E_plot(mesh_indices,1),ds_ratio),downsample(E_plot(mesh_indices,2),ds_ratio),downsample(E_plot(mesh_indices,3),ds_ratio),2,'filled','Color',[0.70,0.70,0.70],'MaxHeadSize',1)
    colormap("parula")
    axis('tight','equal','off');
    camlight
    lighting gouraud
    view(va)
    %colorbar
    % Plot coil
    plot3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),'.g','MarkerSize',30)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(1),coil.rot_str(2),coil.rot_str(3),0.01,"filled",'Color','g','MaxHeadSize',10)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(4),coil.rot_str(5),coil.rot_str(6),0.01,"filled",'Color','g','MaxHeadSize',10)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(7),coil.rot_str(8),coil.rot_str(9),0.01,"filled",'Color','g','MaxHeadSize',10)
end

end