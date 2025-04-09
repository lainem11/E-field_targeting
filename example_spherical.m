% Example of E-field targeting on a spherical cortex model.

%% Create a spherical cortex and a set of electric fields

% Create a sphere with radius of 7 cm.
mesh = create_half_sphere(0.07,50);

% Generate circular field
Efield1 = generate_ring_field(mesh);
Efield1_mag = sqrt(sum(Efield1.^2,2));

% Generate dual ring field
Efield2 = generate_dual_ring_field(mesh,0);
Efield2_mag = sqrt(sum(Efield2.^2,2));

% Generate another dual ring field with 90 degree rotation
Efield3 = generate_dual_ring_field(mesh,pi/2);
Efield3_mag = sqrt(sum(Efield3.^2,2));

% Generate four-ring field
Efield4 = generate_four_ring_field(mesh,0);
Efield4_mag = sqrt(sum(Efield4.^2,2));

% Generate another four-ring field with 45 degree rotation
Efield5 = generate_four_ring_field(mesh,pi/4);
Efield5_mag = sqrt(sum(Efield4.^2,2));

% Plot
E_set = {Efield1,Efield2,Efield3,Efield4,Efield5};
plotEfields(E_set,mesh)

%% Optimize E-field

% Specify the location where the E-field should be focused at, and the 
% direction it should have. A target specified as [0,0,0] focuses the
% E-field at the apex of the half-sphere, with direction towards the
% positive y-axis. In general, a target [x,y,theta] translates the E-field
% focus x mm horizontally, y mm vertically, and theta radians clockwise.

% Example targets
targets = [0,0,0; 8,0,0; 8,0,45; 8,8,-45];

% Optimize
X = [];
for i = 1:size(targets,1)
    result = optimize_Efield_spherical(targets(i,:),mesh.vertices,E_set);
    X(i,:) = result.weights;
end
%%
% Plot results
plot_result_spherical(targets,mesh.vertices,E_set,X)

%% Helper functions

function plotEfields(Efield_cell,mesh)
mesh_inds = subsample_mesh(mesh.vertices,0.005);
N_E = length(Efield_cell);
f=figure;
f.Position(3:4) = [1600,300];
tiledlayout(1,N_E,"TileSpacing","tight")
axs = [];
for i = 1:N_E
    Efield = Efield_cell{i};
    Efield_mag = sqrt(sum(Efield.^2,2));
    axs(i) = nexttile;
    hold on
    patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',Efield_mag,'FaceColor','interp','LineStyle','None');
    quiver3(mesh.vertices(mesh_inds,1),mesh.vertices(mesh_inds,2),mesh.vertices(mesh_inds,3),Efield(mesh_inds,1),Efield(mesh_inds,2),Efield(mesh_inds,3),1,"filled",'Color',[0.70,0.70,0.70],'MaxHeadSize',1)
    axis('tight','equal');
    title_str = sprintf("Field %i",i);
    title(title_str)
    c1 = colorbar;
    c1.TickDirection = 'in';
    c1.LineWidth = 1.5;
    c1.TickLength =.01;
    c1.Ticks = linspace(0,1,5);
end
linkaxes(axs,'xy')
end

function field = generate_dual_ring_field(cortex,phi)
    field = generate_ring_field(cortex,pi/2,phi)-generate_ring_field(cortex,-pi/2,phi);
    field = field/max(sqrt(sum(field.^2,2)));
end

function field = generate_four_ring_field(cortex,phi)
    field = generate_ring_field(cortex,pi/2,phi)+generate_ring_field(cortex,-pi/2,phi)- generate_ring_field(cortex,pi/2,phi-pi/2)-generate_ring_field(cortex,-pi/2,phi-pi/2);
    field = field/max(sqrt(sum(field.^2,2)));
end