function p_inds = GUI_area_select(mesh,data,va,brushsize)

% Function for interactively selecting mesh indices by brushing over the
% mesh.
%
% INPUT:        mesh = Structure that contains mesh vertices in mesh.p,
%                      faces in mesh.e and vertex normals in mesh.nn.
%               data (optional) = Datapoints to plot with size mesh.p.
%               va = view angle for the plot
%               brushsize = Vertex selection distance from cursor
%
% OUTPUT:       p_inds = indices of selected mesh vertices.
%
% v290922 Mikael Laine
%

if nargin == 1
    data = ones(size(mesh.p,1),1);
    % Set view angle with average mesh normal
    N = mean(mesh.nn,1,'Omitnan');
    N = N/norm(N);
    va = [-vectorAngle([0,-1,0],N),vectorAngle([-1,0,0],N)];
    brushsize = 0.005; % mm radius
elseif nargin == 2
    N = mean(mesh.nn,1,'Omitnan');
    N = N/norm(N);
    va = [-vectorAngle([0,-1,0],N),vectorAngle([-1,0,0],N)];
    brushsize = 0.005; % mm radius
elseif nargin == 3
    brushsize = 0.005; % mm radius
end

sc = get(0,'screensize');
f = figure(99);clf
set(f,'Position',[sc(3)/4 sc(4)/5 sc(3)/2 sc(4)*2/3])

% Plot mesh
hp = patch('Faces',mesh.e,'Vertices',mesh.p,'FaceVertexCData',data,'FaceColor','interp','ButtonDownFcn',@brushFcn);
view(va);
colormap("parula")
axis('tight','equal','off');
camlight
%lightangle(va(1),va(2))
lighting gouraud
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
        dist2point = sqrt(sum((mesh.p-pos).^2,2));
        p_ind = find(dist2point < brushsize);
        % Set color vertices
        hp.FaceVertexCData(p_ind) = 0;
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
                plot3(mesh.p(p_inds,1),mesh.p(p_inds,2),mesh.p(p_inds,3),'.r','MarkerSize',10)
                pause(2)
                close(f)
            case hb_reset
                % Reset changes
                hp = patch('Faces',mesh.e,'Vertices',mesh.p,'FaceVertexCData',data,'FaceColor','interp');
                p_inds = [];
        end
        drawnow
    end

    function ThetaInDegrees = vectorAngle(u,v)
        CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
        ThetaInDegrees = real(acosd(CosTheta));
    end

end