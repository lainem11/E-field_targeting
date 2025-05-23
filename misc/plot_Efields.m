function plot_Efields(E_set,mesh)
mesh_inds = subsample_mesh(mesh.vertices,0.005);
N_E = size(E_set,1);
f=figure;
f.Position(3:4) = [1600,300];
tiledlayout(1,N_E,"TileSpacing","tight")
axs = [];
for i = 1:N_E
    Efield = squeeze(E_set(i,:,:));
    Efield_mag = sqrt(sum(Efield.^2,2));
    axs(i) = nexttile;
    hold on
    patch('Faces',mesh.faces,'Vertices',mesh.vertices,'FaceVertexCData',Efield_mag,'FaceColor','interp','LineStyle','None');
    quiver3(mesh.vertices(mesh_inds,1),mesh.vertices(mesh_inds,2),mesh.vertices(mesh_inds,3),Efield(mesh_inds,1),Efield(mesh_inds,2),Efield(mesh_inds,3),1,"filled",'Color',[0.70,0.70,0.70],'MaxHeadSize',1)
    axis('tight','equal');
    title_str = sprintf("Field %i",i);
    title(title_str)
    c1 = colorbar;
    c1.TickDirection = 'in';
    c1.LineWidth = 1.5;
    c1.TickLength =.01;
    c1.Ticks = linspace(0,1,5);
end
linkaxes(axs,'xy')
end