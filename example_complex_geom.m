% Example of E-field targeting in a complex mesh geometry.
%
% This script optimizes E-fields in a paired stimulus scenario, where the
% E-field of first conditioning stimulus (CS) is resticted on the focus area of 
% the following test stimulus (TS).

%% Acquire a set of E-fields

addpath("misc/")

surface_type = 'complex';
[efield_set,mesh] = generate_example_efields(surface_type);

% Plot
plot_efields(efield_set,mesh)

%% Construct a targeting model structure

targeting_model.efield_set = efield_set;
targeting_model.mesh = mesh;

%% Create conditioning stimulus targets

targeting_model.CS_targets = GUI_select_target(targeting_model);

%% Create test stimulus targets

targeting_model.TS_targets = GUI_select_target(targeting_model);

%% Select region to avoid when applying CS

[targeting_model.CS_targets.restrict_inds] = deal(GUI_select_area(targeting_model));

%% Plot targets

plot_model(targeting_model)

%% Optimize E-fields (takes ~20 seconds)

% Set flag to use (1) or not use (0) parallel computation
run_parallel = 0;

% Set maximum errors for the stimulation location and direction.
distance_constr = 0.003; % m
angle_constr = 10; % deg

% Run optimization
results = optimize_efields(targeting_model,distance_constr,angle_constr,run_parallel);

plot_result_complex_geom(targeting_model,results)

%% Save results

save("targeting_results.mat",'results')

%% Helper functions

function targeting_results = optimize_efields(targeting_model,dist_constr,angle_constr,parallel_flag)
% Parses the targeting_model structure and runs E-field optimization with
% the specified targets.

if isfield(targeting_model,'ROI')
    mesh.vertices = targeting_model.mesh.vertices(ROI,:);
    mesh.normals = targeting_model.mesh.normals(ROI,:);
else
    mesh = targeting_model.mesh;
end

efield_set = targeting_model.efield_set;

CS_targets = targeting_model.CS_targets;
TS_targets = targeting_model.TS_targets;

% Fill restrict_inds fields if not present
if ~isfield(CS_targets,'restrict_inds')
    [CS_targets.restrict_inds] = deal([]);
end
if ~isfield(TS_targets,'restrict_inds')
    [TS_targets.restrict_inds] = deal([]);
end

% Add labels if not present
if ~isfield(CS_targets,'label')
    CS_labels = compose("CS_target%d",1:length(CS_targets));
    [CS_targets.label] = CS_labels{:};
end
if ~isfield(TS_targets,'label')
    TS_labels = compose("TS_target%d",1:length(TS_targets));
    [TS_targets.label] = TS_labels{:};
end
 
targets = [CS_targets,TS_targets];

% Optimize
if parallel_flag
    if isempty(gcp('nocreate'))
        parpool; % Create a parallel pool of workers if not already open
    end
    parfor i = 1:length(targets)
        targeting_results{i} = optimize_efield_complex_geom(targets(i).pos,targets(i).dir,mesh,efield_set,'restrictEF',targets(i).restrict_inds,'DistConstr',dist_constr,'AngleConstr',angle_constr);
        targeting_results{i}.label = targets(i).label;
    end
else
    for i = 1:length(targets)
        targeting_results{i} = optimize_efield_complex_geom(targets(i).pos,targets(i).dir,mesh,efield_set,'restrictEF',targets(i).restrict_inds,'DistConstr',dist_constr,'AngleConstr',angle_constr);
        targeting_results{i}.label = targets(i).label;
    end
end

end