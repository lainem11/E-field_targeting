% Example of E-field targeting with a spherical cortex model.

%% Acquire a set of E-fields

addpath("misc/")

surface_type = 'spherical';
[efield_set,mesh] = generate_example_efields(surface_type);

plot_efields(efield_set,mesh)

%% Optimize E-field

% Specify the location where the E-field should be focused at, and the 
% direction it should have. A target specified as [0,0,0] focuses the
% E-field at the apex of the half-sphere, with direction towards the
% positive y-axis. In general, a target [x,y,theta] translates the E-field
% focus x mm horizontally, y mm vertically, and theta radians clockwise.

% Example targets
targets = [0,0,0; 
           8,0,0;
           8,0,45;
           8,8,-45];

% Optimize
X = [];
tic
for i = 1:size(targets,1)
    result = optimize_efield_spherical(targets(i,:),mesh.vertices,efield_set);
    X(i,:) = result.weights;
end
toc
% Plot
plot_result_spherical(targets,mesh.vertices,efield_set,X)

%%