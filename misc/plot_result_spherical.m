function plot_result_spherical(targets,vertices,E_set,X)
r = max(sqrt(sum(vertices.^2,2)));

centerCortex = [0,0,r];
pos = vertices;
% Rotation matrices
Rmatx = @(a) [1 0 0;0 cos(a) -sin(a);0 sin(a) cos(a)];
Rmaty = @(a) [cos(a) 0 sin(a);0 1 0;-sin(a) 0 cos(a)];
Rmatz = @(a) [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];

N_targets = size(targets,1);
figure
tiledlayout(1,N_targets)
show_leg = 1;
for t = 1:N_targets
    target_shift = targets(t,:);
    x = X(t,:);
    % Orient EF shift with the third coil orientation
    phi = target_shift(2)/1000/r;
    pha = -target_shift(1)/1000/r;
    pho = deg2rad(target_shift(3));
    % Shift EF using the inputs
    target.p = centerCortex*Rmatx(phi)*Rmaty(pha);
    % Find closest mesh index
    [~,target.Ind] = min(sqrt(sum((pos-target.p).^2,2)));
    % Define target direction
    centerDir = [0,1,0];
    target.Dir = centerDir*Rmatz(pho)*Rmatx(phi)*Rmaty(pha);

    % Calculate centroid position and E field direction
    E = squeeze(sum(E_set.*x',1));
    Emag = sqrt(sum(E.^2,2));
    [Emax,ind_Emax] = max(Emag);
    Emagn = Emag/Emax;

    weightedEF = Emagn.^10;
    centroid_p = sum(pos.*weightedEF,1)/sum(weightedEF);
    % Scale to sphere edge
    centroid_p = centroid_p*(r/sqrt(sum(centroid_p.^2,2)));
    % Select nearest mesh vertex
    [~,centroid_i] = min(sqrt(sum((pos-centroid_p).^2,2)));

    Edir = E(centroid_i,:);
    Edir = Edir/sqrt(sum(Edir.^2));

    % Plot
    mesh_inds = subsample_mesh(pos,0.005);
    faces = delaunay(pos(:,1),pos(:,2));
    nexttile; hold on
    va = [0 90];
    hp = patch('Faces',faces,'Vertices',pos-[0,0,0.001],'FaceVertexCData',Emag,'FaceColor','interp','EdgeColor','interp');
    view(va);
    % Arrows
    quiver3(pos(mesh_inds,1),pos(mesh_inds,2),pos(mesh_inds,3),E(mesh_inds,1),E(mesh_inds,2),E(mesh_inds,3),0.5,"filled",'Color',[0.7,0.7,0.7],'MaxHeadSize',1,'LineWidth',1)
    % Center
    p1 = plot3(centerCortex(1),centerCortex(2),centerCortex(3),'.k','MarkerSize',40);
    q1 = quiver3(centerCortex(1),centerCortex(2),centerCortex(3),centerDir(1),centerDir(2),centerDir(3),0.01,"filled",'Color','k','MaxHeadSize',1,'LineWidth',3);
    % Target
    p2 = plot3(target.p(1),target.p(2),target.p(3)+0.0005,'.','Color',"#0072BD",'MarkerSize',30);
    q2 = quiver3(target.p(1),target.p(2),target.p(3)+0.0005,target.Dir(1),target.Dir(2),target.Dir(3),0.01,"filled",'Color',"#0072BD",'MaxHeadSize',1,'LineWidth',2);
    % Result
    p3 = plot3(centroid_p(1),centroid_p(2),centroid_p(3)+0.001,'.','Color',"#D95319",'MarkerSize',20);
    q3 = quiver3(centroid_p(1),centroid_p(2),centroid_p(3)+0.001,Edir(1),Edir(2),Edir(3),0.01,"filled",'Color',"#D95319",'MaxHeadSize',1,'LineWidth',1.5);
    axis('tight','equal');
    material dull
    
    c1 = colorbar;
    c1.TickDirection = 'in';
    c1.LineWidth = 1.5;
    c1.TickLength =.01;
    c1.Ticks = linspace(0,1,5);
    tstring = sprintf("Target: [%i mm,%i mm,%i%c]",target_shift(1),target_shift(2),target_shift(3),char(176));
    title(tstring)
    
    if show_leg
        legend([p1,p2,p3],'Center','Target','Result')
        show_leg = 0;
    end
end

end

function ThetaInDegrees = vectorAngle(u,v)
CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
ThetaInDegrees = real(acosd(CosTheta));
end