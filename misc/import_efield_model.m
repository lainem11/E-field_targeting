function efield_model = import_efield_model(headmodel_path,E_path)

%% Load mesh and calculate face normals
[vertices,faces] = bin_to_vtk(headmodel_path);
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
efield_set = reshape(str2num(str),[],5,3);
efield_set = efield_set(ROI,:,:);
efield_set = permute(efield_set,[2,1,3]);

% Region-of-interest mesh
ROI_mesh.vertices = mesh.vertices(ROI,:);
ROI_mesh.normals = mesh.normals(ROI,:);

% Calculate camera angle facing the ROI
N = mean(ROI_mesh.normals,1,'Omitnan');
N = N/norm(N);
mesh.view_angle = [-calculate_vector_angle([0,-1,0],N),calculate_vector_angle([-1,0,0],N)];

% Read coil position
coil.pos_str = str2num(data(6));
coil.rot_str = str2num(data(4));

% Plot
f=figure(1);clf
f.Position(3:4)=[1600,1000];
tiledlayout(2,3,"TileSpacing","none")
arrow_inds = subsample_mesh(ROI_mesh.vertices);

for i = 1:size(efield_set,1)
    nexttile
    E_plot = zeros(size(vertices));
    E_plot(ROI,:) = squeeze(efield_set(i,:,:));
    E_plot_mag = sqrt(sum(E_plot.^2,2));
    hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',E_plot_mag,'FaceColor','interp','LineStyle','none');
    hold on
    quiver3(mesh.vertices(ROI(arrow_inds),1),mesh.vertices(ROI(arrow_inds),2),mesh.vertices(ROI(arrow_inds),3),E_plot(ROI(arrow_inds),1),E_plot(ROI(arrow_inds),2),E_plot(ROI(arrow_inds),3),2,'filled','Color',[0.70,0.70,0.70],'MaxHeadSize',1)
    
    colormap(viridis)
    axis('tight','equal','off');
    camlight
    lighting gouraud
    material dull
    view(mesh.view_angle)

    % Plot coil
    plot3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),'.','Color',[0.0,1.0,0.0],'MarkerSize',30)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(1),coil.rot_str(2),coil.rot_str(3),0.01,"filled",'Color',[0.0,1.0,0.0],'MaxHeadSize',10)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(4),coil.rot_str(5),coil.rot_str(6),0.01,"filled",'Color',[0.0,1.0,0.5],'MaxHeadSize',10)
    quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(7),coil.rot_str(8),coil.rot_str(9),0.01,"filled",'Color',[0.0,1.0,0.0],'MaxHeadSize',10)
    title(sprintf("Coil %i",i))
end

efield_model.efield_set = efield_set;
efield_model.mesh = mesh;
efield_model.ROI = ROI;
efield_model.coil = coil;

function [points, elements] = bin_to_vtk(filename)
    % Check if the file exists
    if exist(filename, 'file') == 2
       % Read the binary file
        fid = fopen(filename, 'rb');
        numbers = fread(fid, 3, 'int32');
        points = fread(fid, [3, numbers(2)], 'float32')';
        elements = fread(fid, [3, numbers(3)], 'int32')';
        fclose(fid);
    else
        disp('File does not exist');
    end
end

function EfieldData = import_Efield_data(filename)

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 13);

% Specify range and delimiter
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["MarkerID", "T_rot", "CoilCenter", "CoilPositionInWorldCoordinates", "InVesaliusCoordinates", "Enorm", "IDCellMax", "EfieldVectors", "EnormCellIndexes", "FocalFactors", "EfieldThreshold", "EfieldROISize", "Mtms_coord"];
opts.VariableTypes = ["string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";

% Specify variable properties
opts = setvaropts(opts, ["MarkerID", "T_rot", "CoilCenter", "CoilPositionInWorldCoordinates", "InVesaliusCoordinates", "Enorm", "IDCellMax", "EfieldVectors", "EnormCellIndexes", "FocalFactors", "EfieldThreshold", "EfieldROISize", "Mtms_coord"], "WhitespaceRule", "preserve", "EmptyFieldRule", "auto");

% Import the data
EfieldData = readmatrix(filename, opts);

end

function ThetaInDegrees = calculate_vector_angle(u,v)
        CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
        ThetaInDegrees = real(acosd(CosTheta));
end

end