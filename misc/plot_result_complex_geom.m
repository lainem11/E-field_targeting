function plot_result_complex_geom(results,E_set,mesh,ROI,labels,coil)
    num_plots = length(results);
    N = mean(mesh.normals(ROI,:),1,'Omitnan');
    N = N/norm(N);
    va = [-calculate_vector_angle([0,-1,0],N),calculate_vector_angle([-1,0,0],N)];
    f=figure;
    tiledlayout(1,num_plots,"TileSpacing","none")
    f.Position(3:4)=[400*num_plots,600];
    arrow_inds = subsample_mesh(mesh.vertices);
    for i = 1:num_plots
        result = results{i};
        E = squeeze(sum(E_set.*result.weights,1));
        E_plot = zeros(size(mesh.vertices));
        E_plot(ROI,:) = E;
        [E_plot_mag,E_mag_max,E_mag_max_ind] = E_to_mag(E_plot);
        E_plot_mag_n = E_plot_mag/E_mag_max;

        nexttile; hold on
        % Plot mesh
        hp = patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',E_plot_mag,'FaceColor','interp','LineStyle','none');
        % Add arrows
        quiver3(mesh.vertices(arrow_inds,1),mesh.vertices(arrow_inds,2),mesh.vertices(arrow_inds,3),E_plot(arrow_inds,1),E_plot(arrow_inds,2),E_plot(arrow_inds,3),0.5,"filled",'Color',[0.70,0.70,0.70],'MaxHeadSize',1)
        
        % Add target point and direction
        plot3(result.inputs.pos(1),result.inputs.pos(2),result.inputs.pos(3),'.','Color',"#D95319",'MarkerSize',30)
        q1= quiver3(result.inputs.pos(1),result.inputs.pos(2),result.inputs.pos(3),result.inputs.direction(1),result.inputs.direction(2),result.inputs.direction(3),0.01,'filled','r','LineWidth',2,'MaxHeadSize',1);
        % Add adjusted target direction
        q2 = quiver3(result.inputs.pos(1),result.inputs.pos(2),result.inputs.pos(3),result.target.Dir(1),result.target.Dir(2),result.target.Dir(3),0.01,'filled','Color',"#D95319",'LineWidth',2,'MaxHeadSize',1);
        
        % Add generated stimulation point and direction
        [centroid,centroid_ind] = calculate_WCOG(E_plot_mag_n,mesh.vertices);
        centroid_dir = E_plot(centroid_ind,:);
        centroid_dir = centroid_dir/norm(centroid_dir);
        centroid_lifted = centroid + result.N*0.01;
        plot3(centroid_lifted(1),centroid_lifted(2),centroid_lifted(3),'.','Color',"#0072BD",'MarkerSize',30)
        q3 = quiver3(centroid_lifted(1),centroid_lifted(2),centroid_lifted(3),centroid_dir(1),centroid_dir(2),centroid_dir(3),0.015,'filled','Color',"#0072BD",'LineWidth',2,'MaxHeadSize',1);
        plot3([centroid(1),centroid_lifted(1)],[centroid(2),centroid_lifted(2)],[centroid(3),centroid_lifted(3)],'-','Color',"#0072BD",'LineWidth',2)
        
        % Add maximum point
        plot3(mesh.vertices(E_mag_max_ind,1),mesh.vertices(E_mag_max_ind,2),mesh.vertices(E_mag_max_ind,3),'.','Color',"black",'MarkerSize',30)
        E_dir_norm = E_plot(E_mag_max_ind,:)/norm(E_plot(E_mag_max_ind,:));
        q4 = quiver3(mesh.vertices(E_mag_max_ind,1),mesh.vertices(E_mag_max_ind,2),mesh.vertices(E_mag_max_ind,3),E_dir_norm(1),E_dir_norm(2),E_dir_norm(3),0.01,'filled','Color',"black",'LineWidth',2,'MaxHeadSize',1);
        
        % Add coil (if available)
        if nargin == 6
            plot3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),'.','Color',[0.5,0.5,0.5],'MarkerSize',30)
            q5=quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(1),coil.rot_str(2),coil.rot_str(3),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',1);
            quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(4),coil.rot_str(5),coil.rot_str(6),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',1)
            quiver3(coil.pos_str(1),coil.pos_str(2),coil.pos_str(3),coil.rot_str(7),coil.rot_str(8),coil.rot_str(9),0.01,"filled",'Color',[0.5,0.5,0.5],'MaxHeadSize',1)
            if i == 1
                legend([q1,q2,q3,q4,q5],'Input target','Used target','WCOG','Max','Coil')
            end
        else
            if i == 1
                legend([q1,q2,q3,q4],'Input target','Used target','WCOG','Max')
            end
        end
        axis('tight','equal','off');
        camlight
        lighting gouraud
        material dull
        view(va)
        colormap(viridis)
        c1 = colorbar('southoutside');
        c1.TickDirection = 'in';
        c1.LineWidth = 1.5;
        c1.TickLength =.01;
        c1.Ticks = linspace(0,1,5);
        c1.Position(3) = c1.Position(3)*0.8;

        
        title_str = sprintf("Target: %s\nloc: %.2f mm, dir: %.2f deg",labels{i},result.err.location,result.err.angle);
        title(title_str)
    end
end