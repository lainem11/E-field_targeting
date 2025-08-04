% Example of E-field targeting with imported E-fields and stimulation
% targets from InVesalius.
%

%% Acquire a set of E-fields

addpath("misc/")

subject_root_path = "MY_SUBJECT_DATA_DIRECTORY";
subject_id = "SUB_00";

efield_path = fullfile(subject_root_path,subject_id,"MY_EFIELD_FILE.csv");
headmodel_path = fullfile(subject_root_path,subject_id,"MY_HEADMESH_FILE.bin");

%% Load E-fields and head model

targeting_model = import_efield_model(headmodel_path,efield_path);

%% Load stimulation targets from the marker file

marker_file = fullfile(subject_root_path,subject_id,"MY_MARKER_FILE.mkss");

CS_target_labels = {'MY_LABEL1','MY_LABEL2'};
TS_target_labels = {'MY_LABEL3'};

targeting_model.CS_targets = import_targets(marker_file,CS_target_labels);
targeting_model.TS_targets = import_targets(marker_file,TS_target_labels);

plot_model(targeting_model)

%% (ALTERNATIVE #1) Create conditioning stimulus targets

targeting_model.CS_targets = GUI_select_target(targeting_model);

%% (ALTERNATIVE #1) Create test stimulus targets

targeting_model.TS_targets = GUI_select_target(targeting_model);

%% Restrict E-field on TS when targeting CS, and vica versa.

% Find indices on mesh
CS_target_inds = pos2ind(vertcat(targeting_model.CS_targets.pos),targeting_model.mesh.vertices(targeting_model.ROI,:));
TS_target_inds = pos2ind(vertcat(targeting_model.TS_targets.pos),targeting_model.mesh.vertices(targeting_model.ROI,:));

% Update the targeting model
[targeting_model.CS_targets.restrict_inds] = deal(TS_target_inds);
[targeting_model.TS_targets.restrict_inds] = deal(CS_target_inds);
plot_model(targeting_model)

%% (ALTERNATIVE #2) Select region to avoid when applying CS

[targeting_model.CS_targets.restrict_inds] = deal(GUI_select_area(targeting_model));

%% (ALTERNATIVE #2) Select region to avoid when applying TS

[targeting_model.TS_targets.restrict_inds] = deal(GUI_select_area(targeting_model));

%% Plot

plot_model(targeting_model)

%% Optimize E-fields (takes ~10 seconds per target)

% Set flag to use (1) or not use (0) parallel computation
run_parallel = 0;

% Set maximum errors for the stimulation location and direction.
distance_constr = 0.003; % m
angle_constr = 10; % deg

% Run optimization
results = optimize_efields(targeting_model,distance_constr,angle_constr,run_parallel);

plot_result_complex_geom(targeting_model,results)

%% Save results

save(fullfile(subject_root_path,subject_id,"targeting_results.mat"),'results')

%% Helper functions

function targeting_results = optimize_efields(targeting_model,dist_constr,angle_constr,parallel_flag)
% Parses the targeting_model structure and runs E-field optimization with
% the specified targets.

if isfield(targeting_model,'ROI')
    mesh.vertices = targeting_model.mesh.vertices(targeting_model.ROI,:);
    mesh.normals = targeting_model.mesh.normals(targeting_model.ROI,:);
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
