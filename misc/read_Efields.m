function [E_matrix,mesh,E_mesh,ROI,coil] = read_Efields(headmodel_path,E_path)

%% Load mesh and calculate face normals
[vertices,faces] = convert_custom_bin_to_vtk(headmodel_path);
mesh.vertices = vertices;
mesh.faces = faces+1;

% Step 1: Calculate face normals
numFaces = size(mesh.faces, 1);
faceNormals = zeros(numFaces, 3);
for f = 1:numFaces
    v1 = mesh.vertices(mesh.faces(f,1),:);
    v2 = mesh.vertices(mesh.faces(f,2),:);
    v3 = mesh.vertices(mesh.faces(f,3),:);
    edge1 = v2 - v1;
    edge2 = v3 - v1;
    faceNormals(f, :) = cross(edge1, edge2);
end

% Step 2: Initialize vertex normals
numVertices = size(mesh.vertices, 1);
vertexNormals = zeros(numVertices, 3);

% Accumulate face normals for each vertex
for f = 1:numFaces
    for j = 1:3
        vertexNormals(mesh.faces(f, j), :) = vertexNormals(mesh.faces(f, j), :) + faceNormals(f, :);
    end
end

% Step 3: Normalize the vertex normals
for v = 1:numVertices
    normValue = norm(vertexNormals(v, :));
    if normValue > 0
        vertexNormals(v, :) = vertexNormals(v, :) / normValue;
    end
end

mesh.normals = vertexNormals;


%% Load E-fields

data = import_Efield_data(E_path);

% Load mesh indices
ROI = data(18);
ROI = str2num(ROI)'+1;

% Load e-field set
str = char(data(16));
E_matrix = reshape(str2num(str),[],5,3);
E_matrix = E_matrix(ROI,:,:)*1e-3;
E_matrix = permute(E_matrix,[2,1,3]);

% Region-of-interest mesh
E_mesh.vertices = mesh.vertices(ROI,:);
E_mesh.normals = mesh.normals(ROI,:);

% Read coil position
coil.pos_str = str2num(data(6));
coil.rot_str = str2num(data(4));

% Plot
va = [-83.8370   52.7484];
ds_ratio = 16;
f=figure(1);clf
f.Position(3:4)=[1600,1000];
tiledlayout(2,3,"TileSpacing","none")
ROI_mesh = mesh.vertices(ROI,:);
mesh_inds = subsample_mesh(ROI_mesh,0.005);

for i = 1:size(E_matrix,1)
    nexttile
    E_plot = zeros(size(vertices));
    E_plot(ROI,:) = squeeze(E_matrix(i,:,:));
    E_plot_mag = sqrt(sum(E_plot.^2,2));
    hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',E_plot_mag,'FaceColor','interp','LineStyle','none');
    hold on
    quiver3(ROI_mesh(mesh_inds,1),ROI_mesh(mesh_inds,2),ROI_mesh(mesh_inds,3),E_plot(ROI(mesh_inds),1),E_plot(ROI(mesh_inds),2),E_plot(ROI(mesh_inds),3),2,'filled','Color',[0.70,0.70,0.70],'MaxHeadSize',1)
    colormap("parula")
    axis('tight','equal','off');
    camlight
    lighting gouraud
    view(va)
    %colorbar
    % Plot coil
    plot3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),'.','Color',[0.5,0.5,0.5],'MarkerSize',30)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(1),coil.rot_str(2),coil.rot_str(3),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',10)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(4),coil.rot_str(5),coil.rot_str(6),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',10)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(7),coil.rot_str(8),coil.rot_str(9),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',10)
    title(sprintf("Coil %i",i))
end

end