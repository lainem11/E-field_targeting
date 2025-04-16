% Example of E-field targeting in a complex mesh geometry.
%
% This script optimizes E-fields in a paired stimulus scenario, where the
% E-field of first conditioning stimulus is resticted on the focus area of 
% the following test stimulus.

%% Set file paths

addpath("misc/")

% E-fields of saved coil pose
e_field_path = "";

% Headmodel for visualizing the whole mesh
headmodel_path = "";

%% Load E-fields and head model

% Load mesh and e-field
[E_set,full_mesh,E_mesh,ROI,coil] = read_Efields(headmodel_path,e_field_path);

plot_highlight = zeros(size(full_mesh.vertices,1),1); % Highlight ROI in plots
plot_highlight(ROI) = 1;

% Figure view angle facing the mesh
va = get_view_angle(E_mesh);

%% Create conditioning stimulus targets

CS_targets = GUI_point_and_dir_select(full_mesh,plot_highlight,va);

%% Create test stimulus targets

TS_targets = GUI_point_and_dir_select(full_mesh,plot_highlight,va);

%% Select region to avoid when applying CS

avoid_inds = GUI_area_select(full_mesh,plot_highlight,va);

% Find corresponding indices in ROI
[~,avoid_inds_E_mesh] = ismember(avoid_inds,ROI);

%% Plot targets

plot_targets(full_mesh,CS_targets,TS_targets,avoid_inds,va)

%% Optimize E-fields (may take a minute)

% Set flag to use (1) or not use (0) parallel computation
run_parallel = 1;

% Set maximum errors for the stimulation location and direction.
distance_constr = 0.003; % m
angle_constr = 10; % deg

% Run optimization
results_CS = optimize_Efields(CS_targets,distance_constr,angle_constr,E_mesh,E_set,run_parallel,avoid_inds_E_mesh);
results_TS = optimize_Efields(TS_targets,distance_constr,angle_constr,E_mesh,E_set,run_parallel,[]);

%% Plot results

lables_CS = {};
labels_TS = {};
for i = 1:length(CS_targets)
    labels_CS{i} = sprintf("CS%i",i);
end
for i = 1:length(TS_targets)
    labels_TS{i} = sprintf("TS%i",i);
end

plot_result_complex_geom(results_CS,E_set,full_mesh,ROI,labels_CS,coil);
plot_result_complex_geom(results_TS,E_set,full_mesh,ROI,labels_TS,coil);

%% Functions

function plot_targets(mesh,CS_targets,TS_targets,TS_inds,va)
    f=figure(2);clf
    f.Position(3:4)=[1200,800];
    % Ts inds
    p1=plot3(mesh.vertices(TS_inds,1),mesh.vertices(TS_inds,2),mesh.vertices(TS_inds,3),'.','Color','#EDB120','MarkerSize',20);
    hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',ones(size(mesh.vertices)),'FaceColor','interp','LineStyle','none');
    hold on
    % CS targets
    for i = 1:length(CS_targets)
        p2=plot3(CS_targets(i).pos(1),CS_targets(i).pos(2),CS_targets(i).pos(3),'.','Color','#0072BD','MarkerSize',30);
        quiver3(CS_targets(i).pos(1),CS_targets(i).pos(2),CS_targets(i).pos(3),CS_targets(i).dir(1),CS_targets(i).dir(2),CS_targets(i).dir(3),0.04,"filled",'Color','#0072BD','MaxHeadSize',1,'LineWidth',2)
    end
    % TS targets
    for i = 1:length(TS_targets)
        p3=plot3(TS_targets(i).pos(1),TS_targets(i).pos(2),TS_targets(i).pos(3),'.b','Color','#D95319','MarkerSize',30);
        quiver3(TS_targets(i).pos(1),TS_targets(i).pos(2),TS_targets(i).pos(3),TS_targets(i).dir(1),TS_targets(i).dir(2),TS_targets(i).dir(3),0.04,"filled",'Color','#D95319','MaxHeadSize',1,'LineWidth',2)
    end
    
    legend([p1,p2,p3],'Avoid E-field','CS','TS')
    view(va)
    axis('tight','equal','off');
    camlight
    lighting gouraud
    material dull
end

function va = get_view_angle(mesh)
    N = mean(mesh.normals,1,'Omitnan');
    N = N/norm(N);
    va = [-calculate_vector_angle([0,-1,0],N),calculate_vector_angle([-1,0,0],N)];
end

function targeting_results = optimize_Efields(targets,dist_constr,angle_constr,c_mesh,E_set,par_flag,restrict_inds)

    if par_flag
        if isempty(gcp('nocreate'))
            parpool; % Create a parallel pool of workers if not already open
        end
        parfor i = 1:length(targets)
            if isempty(restrict_inds)
                targeting_results{i} = optimize_Efield_complex_geom(targets(i).pos,targets(i).dir,c_mesh,E_set,'DistConstr',dist_constr,'AngleConstr',angle_constr);
            else
                targeting_results{i} = optimize_Efield_complex_geom(targets(i).pos,targets(i).dir,c_mesh,E_set,'restrictEF',restrict_inds,'DistConstr',dist_constr,'AngleConstr',angle_constr);
            end
        end
    else
        for i = 1:length(targets)
            if isempty(restrict_inds)
                targeting_results{i} = optimize_Efield_complex_geom(targets(i).pos,targets(i).dir,c_mesh,E_set,'DistConstr',dist_constr,'AngleConstr',angle_constr);
            else
                targeting_results{i} = optimize_Efield_complex_geom(targets(i).pos,targets(i).dir,c_mesh,E_set,'restrictEF',restrict_inds,'DistConstr',dist_constr,'AngleConstr',angle_constr);
            end
        end
    end
end
