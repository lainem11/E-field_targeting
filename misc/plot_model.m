function plot_model(targeting_model)
% Visualizes the current state of the efield targeting model.
%
% Inputs:
%
% targeting_model - Struct that contains fields:
%      mesh: Struct with vertices, faces, normals, and view angle.
%      ROI: Array of mesh vertex indices matching the E-field computation region.
%      CS_targets (optional): Struct with pos, dir, and possibly restrict_inds
%           for the conditioning stimulus targets.
%      TS_targets (optional): Like CS_targets but for test stimulus targets.
%

f=figure(2);clf
f.Position(3:4)=[1200,800];

% Initialize arrays for legend handles and labels
legend_handles = [];
legend_labels = {};

% Plot mesh
data = zeros(size(targeting_model.mesh.vertices,1),1);
data(targeting_model.ROI) = 1;
patch('Faces',targeting_model.mesh.faces,'Vertices',targeting_model.mesh.vertices,'FaceVertexCData',data,'FaceColor','interp','LineStyle','none');
hold on

% Add targets
if isfield(targeting_model,'CS_targets')
    CS_targets = targeting_model.CS_targets;
    for i = 1:length(CS_targets)
        % Plot the target point
        p1=plot3(CS_targets(i).pos(1),CS_targets(i).pos(2),CS_targets(i).pos(3),'.','Color','#0072BD','MarkerSize',30);
        quiver3(CS_targets(i).pos(1),CS_targets(i).pos(2),CS_targets(i).pos(3),CS_targets(i).dir(1),CS_targets(i).dir(2),CS_targets(i).dir(3),0.01,"filled",'Color','#0072BD','MaxHeadSize',1,'LineWidth',2)
        
        % Add the first target handle to the legend
        if i == 1
            legend_handles(end+1) = p1;
            legend_labels{end+1} = 'CS target';
        end
    end
    if isfield(targeting_model.CS_targets,'restrict_inds')
        % Plot the restricted area
        p1_1 = scatter3(targeting_model.ROI_mesh.vertices(CS_targets(1).restrict_inds,1),...
                    targeting_model.ROI_mesh.vertices(CS_targets(1).restrict_inds,2),...
                    targeting_model.ROI_mesh.vertices(CS_targets(1).restrict_inds,3),...
                30, 'filled', 'MarkerFaceColor', '#7db1d4','MarkerEdgeColor','none');

        % Add the handle for the restricted area to the legend
        legend_handles(end+1) = p1_1;
        legend_labels{end+1} = 'CS restrict area';
    end
end

if isfield(targeting_model,'TS_targets')
    TS_targets = targeting_model.TS_targets;
    for i = 1:length(TS_targets)
        % Plot the target point
        p2=plot3(TS_targets(i).pos(1),TS_targets(i).pos(2),TS_targets(i).pos(3),'.','Color','#D95319','MarkerSize',30);
        quiver3(TS_targets(i).pos(1),TS_targets(i).pos(2),TS_targets(i).pos(3),TS_targets(i).dir(1),TS_targets(i).dir(2),TS_targets(i).dir(3),0.01,"filled",'Color','#D95319','MaxHeadSize',1,'LineWidth',2)
        
        % Add the first target handle to the legend
        if i == 1
            legend_handles(end+1) = p2;
            legend_labels{end+1} = 'TS target';
        end
    end
    if isfield(targeting_model.TS_targets,'restrict_inds')
        % Plot the restricted area
        p2_1 = scatter3(targeting_model.ROI_mesh.vertices(TS_targets(1).restrict_inds,1),...
            targeting_model.ROI_mesh.vertices(TS_targets(1).restrict_inds,2),...
            targeting_model.ROI_mesh.vertices(TS_targets(1).restrict_inds,3),...
            30, 'filled', 'MarkerFaceColor', '#d69c83','MarkerEdgeColor','none');

        % Add the handle for the restricted area to the legend
        legend_handles(end+1) = p2_1;
        legend_labels{end+1} = 'TS restrict area';
    end
end

% Create the legend only if there are items to display
if ~isempty(legend_handles)
    legend(legend_handles,legend_labels)
end

axis('tight','equal','off');
camlight
lighting gouraud
material dull
view(targeting_model.mesh.view_angle)
colormap(viridis)
end
