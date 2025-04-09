% Example of E-field targeting in a complex mesh geometry.
%
% This script optimizes E-fields in a paired stimulus scenario, where the
% E-field of first stimulus is resticted on the focus area of the next
% stimulus.

%% Set file paths

addpath("misc/")

% E-fields of saved coil pose
e_field_path = "";

% Headmodel for visualizing the whole mesh
headmodel_path = "";

%% Load E-fields and head model

% Load mesh and e-field
[mesh,E_set,ROI,coil] = read_Efields(headmodel_path,e_field_path);

% Region-of-interest mesh
c_mesh = mesh;
c_mesh.vertices = mesh.vertices(ROI,:);
c_mesh.normals = mesh.normals(ROI,:);
c_mesh_highlight = zeros(size(mesh.vertices,1),1); % Highlight ROI in plots
c_mesh_highlight(ROI) = 1;

% Figure view angle facing ROI
va = get_view_angle(mesh,ROI);

%% Create conditioning stimulus targets

CS_targets = GUI_point_and_dir_select(mesh,c_mesh_highlight,va);

%% Create test stimulus targets

TS_targets = GUI_point_and_dir_select(mesh,c_mesh_highlight,va);

%% Select region to avoid when applying CS

avoid_inds = GUI_area_select(mesh,c_mesh_highlight,va);

% Find corresponding indices in ROI
[~,avoid_inds_cmesh] = ismember(avoid_inds,ROI);

%% Plot targets

plot_targets(mesh,CS_targets,TS_targets,avoid_inds,va)

%% Optimize E-fields (may take a minute)

% Set flag to use (1) or not use (0) parallel computation
run_parallel = 0;

% Set maximum errors for the stimulation location and direction.
distance_constr = 0.003; % m
angle_constr = 10; % deg

% Run optimization
targeting_results_CS = optimize_Efields(CS_targets,distance_constr,angle_constr,c_mesh,E_set,run_parallel,avoid_inds_cmesh);
targeting_results_TS = optimize_Efields(TS_targets,distance_constr,angle_constr,c_mesh,E_set,run_parallel,[]);

%% Plot results

plot_results(targeting_results_CS,targeting_results_TS,E_set,mesh,ROI,coil,va)

%% Functions

function ThetaInDegrees = vector_angle(u,v)
    CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
    ThetaInDegrees = real(acosd(CosTheta));
end

function [E,E_mag] = get_total_Efield(weights,cell_E_matrix)
    E = 0;
    for j = 1:length(weights)
        E = E + cell_E_matrix{j}*weights(j);
    end
    E_mag = sqrt(sum(E.^2,2));
end

function plot_targets(mesh,CS_targets,TS_targets,TS_inds,va)
    f=figure(2);clf
    f.Position(3:4)=[1200,800];
    % Ts inds
    p1=plot3(mesh.vertices(TS_inds,1),mesh.vertices(TS_inds,2),mesh.vertices(TS_inds,3),'.','Color','#EDB120','MarkerSize',20);
    hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',ones(size(mesh.vertices)),'FaceColor','interp');
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
    light
    lighting gouraud
end

function plot_results(CS_targeting_results,TS_targeting_results,cell_E_matrix,mesh,mesh_indices,coil,va)
    CS_plots = length(CS_targeting_results);
    TS_plots = length(TS_targeting_results);
    num_plots = CS_plots + TS_plots;
    ds_ratio = 16;    % Downsampling ratio for arrow plot       
    
    for i = 1:num_plots
        f=figure;
        f.Position(3:4)=[1200,800];

        if i <= CS_plots
            targeting_results = CS_targeting_results{i};
            label = sprintf("CS%i",i);
        else
            targeting_results = TS_targeting_results{i-CS_plots};
            label = sprintf("TS%i",i-CS_plots);
        end
        weights = targeting_results.weights;
        [E,~] = get_total_Efield(weights,cell_E_matrix);
        E_plot = zeros(size(mesh.vertices));
        E_plot(mesh_indices,:) = E;
        E_plot_mag = sqrt(sum(E_plot.^2,2));
        [E_mag_max,E_mag_max_ind] = max(E_plot_mag);

        % Plot mesh
        hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',E_plot_mag,'FaceColor','interp','LineStyle','none');
        hold on
        % Add arrows
        quiver3(downsample(mesh.vertices(mesh_indices,1),ds_ratio),downsample(mesh.vertices(mesh_indices,2),ds_ratio),downsample(mesh.vertices(mesh_indices,3),ds_ratio),downsample(E_plot(mesh_indices,1),ds_ratio),downsample(E_plot(mesh_indices,2),ds_ratio),downsample(E_plot(mesh_indices,3),ds_ratio),1,"filled",'Color',[0.70,0.70,0.70],'MaxHeadSize',1)
        % Add target point and direction
        plot3(targeting_results.inputs.pos(1),targeting_results.inputs.pos(2),targeting_results.inputs.pos(3),'.r','MarkerSize',30)
        q1= quiver3(targeting_results.inputs.pos(1),targeting_results.inputs.pos(2),targeting_results.inputs.pos(3),targeting_results.inputs.direction(1),targeting_results.inputs.direction(2),targeting_results.inputs.direction(3),0.02,'filled','r','LineWidth',2,'MaxHeadSize',10);
        % Add adjusted target direction
        q2 = quiver3(targeting_results.inputs.pos(1),targeting_results.inputs.pos(2),targeting_results.inputs.pos(3),targeting_results.target.Dir(1),targeting_results.target.Dir(2),targeting_results.target.Dir(3),0.01,'filled','Color',[1,0.5,0.5],'LineWidth',2,'MaxHeadSize',10);
        % Add generated stimulation point and direction
        centroid = getCentroid(mesh.vertices,E_plot);
        centroid_lifted = centroid.p + targeting_results.N*0.01;
        plot3(centroid_lifted(1),centroid_lifted(2),centroid_lifted(3),'.g','MarkerSize',30)
        q3 = quiver3(centroid_lifted(1),centroid_lifted(2),centroid_lifted(3),centroid.dir(1),centroid.dir(2),centroid.dir(3),0.01,'filled','g','LineWidth',2,'MaxHeadSize',10);
        plot3([centroid.p(1),centroid_lifted(1)],[centroid.p(2),centroid_lifted(2)],[centroid.p(3),centroid_lifted(3)],'-g','LineWidth',2)
        % Add maximum point
        plot3(mesh.vertices(E_mag_max_ind,1),mesh.vertices(E_mag_max_ind,2),mesh.vertices(E_mag_max_ind,3),'.m','MarkerSize',10)
        E_dir_norm = E_plot(E_mag_max_ind,:)/norm(E_plot(E_mag_max_ind,:));
        q4 = quiver3(mesh.vertices(E_mag_max_ind,1),mesh.vertices(E_mag_max_ind,2),mesh.vertices(E_mag_max_ind,3),E_dir_norm(1),E_dir_norm(2),E_dir_norm(3),0.005,'filled','m','LineWidth',2,'MaxHeadSize',10);
        colormap("parula")
        %colorbar
        axis('tight','equal','off');
        camlight
        lighting gouraud

        view(va)
        % Add coil
        plot3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),'.','Color',[0.5,0.5,0.5],'MarkerSize',30)
        q5=quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(1),coil.rot_str(2),coil.rot_str(3),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',1);
        quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(4),coil.rot_str(5),coil.rot_str(6),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',1)
        quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(7),coil.rot_str(8),coil.rot_str(9),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',1)
        legend([q1,q2,q3,q4,q5],'Orig. target','Adj. target','Centroid','Max','Coil')
        title_str = sprintf("Target: %s, loc: %.2f mm, dir: %.2f deg",label,targeting_results.err.location,targeting_results.err.angle);
        title(title_str)

        %c1 = colorbar;
    end
end

function centroid = getCentroid(pos,E)
    E_mag = sqrt(sum(E.^2,2));
    Emagn = E_mag/max(E_mag);
    weightedEF = Emagn.^10;

    centroid.p = sum(pos .* weightedEF,1) / sum(weightedEF);
    [~,loc_i] = min(sqrt(sum((pos-centroid.p).^2,2)));
    centroid.dir = E(loc_i,:);
    centroid.dir = centroid.dir/norm(centroid.dir);
    centroid.ind = loc_i;
end

function va = get_view_angle(mesh,ROI)
    N = mean(mesh.normals(ROI,:),1,'Omitnan');
    N = N/norm(N);
    va = [-vector_angle([0,-1,0],N),vector_angle([-1,0,0],N)];
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
