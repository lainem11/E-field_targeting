function p_inds = GUI_select_area(targeting_model,brushsize)

% Function for interactively selecting mesh indices by brushing over the
% mesh.
%
% Inputs:       targeting_model - Struct that contains fields:
%                   mesh: Struct with vertices, faces, normals, and view angle.
%                   ROI: Array of mesh vertex indices matching the E-field
%                       computation region.
%
% Output:       p_inds = indices of selected mesh vertices.
%
if nargin == 1
    brushsize = 0.005; % mm radius
end

mesh = targeting_model.mesh;
sc = get(0,'screensize');
f = figure(99);clf
set(f,'Position',[sc(3)/4 sc(4)/5 sc(3)/2 sc(4)*2/3])

if ~isfield(targeting_model,'ROI')
    ROI = 1:size(targeting_model.mesh.vertices,1);
else
   ROI = targeting_model.ROI; 
end

% Initialize arrays for legend handles and labels
legend_handles = [];
legend_labels = {};

% Plot mesh
data = zeros(size(mesh.vertices,1),1);
data(ROI) = 1;
hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',data,'FaceColor','interp','ButtonDownFcn',@brushFcn);
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
    if isfield(targeting_model.CS_targets,'restrict_inds') && ~isempty(targeting_model.CS_targets(1).restrict_inds)
        % Plot the restricted area
        p1_1 = plot3(targeting_model.mesh.vertices(ROI(CS_targets(1).restrict_inds),1),...
                    targeting_model.ROI_mesh.vertices(ROI(CS_targets(1).restrict_inds),2),...
                    targeting_model.ROI_mesh.vertices(ROI(CS_targets(1).restrict_inds),3),...
                    '.','Color','#7db1d4','MarkerSize',10);
        
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
    if isfield(targeting_model.TS_targets,'restrict_inds') && ~isempty(targeting_model.TS_targets(1).restrict_inds)
        % Plot the restricted area
        p2_1 = plot3(targeting_model.mesh.vertices(ROI(TS_targets(1).restrict_inds),1),...
                    targeting_model.ROI_mesh.vertices(ROI(TS_targets(1).restrict_inds),2),...
                    targeting_model.ROI_mesh.vertices(ROI(TS_targets(1).restrict_inds),3),...
                    '.','Color','#d69c83','MarkerSize',10);
        
        % Add the handle for the restricted area to the legend
        legend_handles(end+1) = p2_1;
        legend_labels{end+1} = 'TS restrict area';
    end
end

% Create the legend only if there are items to display
if ~isempty(legend_handles)
    legend(legend_handles,legend_labels)
end

if isfield(targeting_model.mesh,'view_angle')
    view(targeting_model.mesh.view_angle);
end
colormap(viridis)
axis('tight','equal','off');
camlight
lighting gouraud
material dull
title('Click and drag with mouse to select areas. Click OK when done.')
% Add Ok and Reset button
hb_ok = uicontrol('Parent',f,'String','Ok','Units','normalized','Position',[0.85 0.01 0.1 0.08],'Callback', @hButtonCallback);
hb_reset = uicontrol('Parent',f,'String','Reset','Units','normalized','Position',[0.1 0.01 0.1 0.08],'Callback', @hButtonCallback);

% Enable datapoint clicks
dh = datacursormode(f);
set(dh,'Enable','on','UpdateFcn',@myupdatefcn);
p_inds = [];
drawMode = 1;
while drawMode
    try
        % Wait for click and get mesh position
        waitfor(f,'UserData');
        pos = get(f,'UserData');
        % Find neighbours
        dist2point = sqrt(sum((mesh.vertices-pos).^2,2));
        p_ind = find(dist2point < brushsize);
        % Set color vertices
        hp.FaceVertexCData(p_ind) = 0.5;
        % Store vertices
        p_inds = unique([p_inds;p_ind]);
    catch
        %disp("broke")
        break
    end
end

    function txt = myupdatefcn(obj,evt)
        % Update datacursor info when moving mouse
        pos = get(obj,'Position');
        set(gcf,'UserData',pos);
        txt = {''};
        obj.Visible = 'off';
    end

    function hButtonCallback(obj, evt)
        switch obj
            case hb_ok
                % End brushing
                drawMode = 0;
                hold on
                plot3(mesh.vertices(p_inds,1),mesh.vertices(p_inds,2),mesh.vertices(p_inds,3),'.r','MarkerSize',10)
                % Transform p_inds to ROI indices
                p_inds = find(ismember(ROI,p_inds));
                pause(1)
                close(f)
            case hb_reset
                % Reset changes
                hp.FaceVertexCData = data;
                %hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',data,'FaceColor','interp');
                p_inds = [];
        end
        drawnow
    end
end