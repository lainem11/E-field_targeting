function targets = GUI_point_and_dir_select(mesh,data,va)

% Function for interactively selecting mesh indices by brushing over the
% mesh.
%
% Inputs:       mesh - Structure that contains mesh vertices in mesh.vertices,
%                       faces in mesh.faces and face normals in mesh.normals.
%               data (optional) - Datapoints to plot with size mesh.vertices.
%               va (optional) - view angle
%
% Output:       targets - struct with fields:
%                   p_ind: vector of selected indices
%                   dir: (n_vertices,3) matrix of selected directions
%
rotSens = 10;  % Arrow rotation sensitivity

f = figure(99);clf
f.Position(3:4)=[1200,800];

% Plot mesh
if nargin == 1
    data = ones(size(mesh.vertices,1),1);
    % Set view angle with average mesh normal
    N = mean(mesh.normals,1,'Omitnan');
    N = N/norm(N);
    va = [-vectorAngle([0,-1,0],N),vectorAngle([-1,0,0],N)];
elseif nargin==2
    N = mean(mesh.normals,1,'Omitnan');
    N = N/norm(N);
    va = [-vectorAngle([0,-1,0],N),vectorAngle([-1,0,0],N)];
end

hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',data,'FaceColor','interp');
view(va);
colormap("parula")
axis('tight','equal','off');
camlight
%lightangle(va(1),va(2))
lighting gouraud
title('CLICK to select a point. Rotate direction with KEYS. Press ENTER to accept. Click OK when done.')
hold on

% Add Ok button
hb_ok = uicontrol('Parent',f,'String','Ok','Units','normalized','Position',[0.85 0.01 0.1 0.08],'Callback', @hButtonCallback);

% Enable datapoint clicks
dh = datacursormode(f);
set(dh,'Enable','on','SnapToDataVertex','on','UpdateFcn',@myupdatefcn);

p_inds = [];
dirs = [];
drawMode = 1;
while drawMode
    try
        % Wait for click and get mesh position
        waitfor(f,'UserData');
        pos = get(f,'UserData');
        set(dh,'enable','off');
        p_ind = find(all(mesh.vertices == pos,2),1);
        p_inds = [p_inds;p_ind];
        % Take average normal from neighborhood
        n = mean(mesh.normals(sqrt(sum((mesh.vertices(p_ind,:)-mesh.vertices).^2,2)) < 0.03,:),1,'Omitnan');
        %n=N;
        
        % Set and plot initial direction
        dir = cross([-1,0,0],n);
%         if (exist('hpoint','var') && exist('hquiv','var'))
%             delete([hpoint,hquiv]);
%         end
        hpoint = plot3(pos(1),pos(2),pos(3),'.r','MarkerSize',30);
        hquiv = quiver3(gca,pos(1),pos(2),pos(3),dir(1),dir(2),dir(3),0.04,'r-','filled','LineWidth',2);
        rmat_l = rotationMatrixAxis3D(rotSens*pi/180,n);
        rmat_r = rotationMatrixAxis3D(-rotSens*pi/180,n);
        set(f,'KeyPressFcn',@rotateArrow)
        % Wait until direction is set
        dirReady = 0;
        while ~dirReady
            waitforbuttonpress
        end
        dirs = [dirs;dir];
        set(dh,'Enable','on','SnapToDataVertex','on','UpdateFcn',@myupdatefcn);
    catch
        continue
    end
end

    function rotateArrow(src,evt)
        % Change direction with arrow keys
        switch evt.Key
            case 'leftarrow'
                %delete(hquiv)
                dir = (rmat_l*dir')';
            case 'rightarrow'
                %delete(hquiv)
                dir = (rmat_r*dir')';
            case 'return'
                dirReady = 1;
            otherwise
                return
        end
        % Update arrow
        set(hquiv,'Udata',dir(1),'Vdata',dir(2),'WData',dir(3));
        drawnow
    end

    function txt = myupdatefcn(obj,evt)
        % Update datacursor info when moving mouse
        pos = get(obj,'Position');
        set(gcf,'UserData',pos);
        txt = {''};
        obj.Visible = 'off';
    end

    function hButtonCallback(obj, evt)
                % End selection
                drawMode = 0;

                for i = 1:length(p_inds)
                    targets(i).pos = mesh.vertices(p_inds(i),:);
                    targets(i).dir = dirs(i,:);
                end

                close(f)
    end

    function ThetaInDegrees = vectorAngle(u,v)
        CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
        ThetaInDegrees = real(acosd(CosTheta));
    end

    function R = rotationMatrixAxis3D(r,axis)
        %function R= rotationmat3D(radians,Axis)
        %
        % creates a rotation matrix such that R * x
        % operates on x by rotating x around the origin r radians around line
        % connecting the origin to the point "Axis"
        %
        % example:
        % rotate around a random direction a random amount and then back
        % the result should be an Identity matrix
        %
        %r = rand(4,1);
        %rotationmat3D(r(1),[r(2),r(3),r(4)]) * rotationmat3D(-r(1),[r(2),r(3),r(4)])
        %
        % example2:
        % rotate around z axis 45 degrees
        % Rtest = rotationmat3D(pi/4,[0 0 1])
        %
        %Bileschi 2009
        if nargin == 1
            if(length(rotX) == 3)
                rotY = rotX(2);
                rotZ = rotZ(3);
                rotX = rotX(1);
            end
        end
        % useful intermediates
        L = norm(axis);
        if (L < eps)
            error('axis direction must be non-zero vector');
        end
        axis = axis / L;
        L = 1;
        u = axis(1);
        v = axis(2);
        w = axis(3);
        u2 = u^2;
        v2 = v^2;
        w2 = w^2;
        c = cos(r);
        s = sin(r);
        %storage
        R = nan(3);
        %fill
        R(1,1) =  u2 + (v2 + w2)*c;
        R(1,2) = u*v*(1-c) - w*s;
        R(1,3) = u*w*(1-c) + v*s;
        R(2,1) = u*v*(1-c) + w*s;
        R(2,2) = v2 + (u2+w2)*c;
        R(2,3) = v*w*(1-c) - u*s;
        R(3,1) = u*w*(1-c) - v*s;

        R(3,2) = v*w*(1-c)+u*s;
        R(3,3) = w2 + (u2+v2)*c;

    end

end