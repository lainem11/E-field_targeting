function plot_result_complex_geom(targeting_model, results)
    %   Plots targeting results on complex geometries.
    %
    %   This function visualizes the results of an E-field targeting simulation
    %   on a 3D mesh. It generates two figures:
    %   1.  A figure showing results for Conditioning Stimulus (CS) targets.
    %   2.  A figure showing results for Test Stimulus (TS) targets.
    %
    %   Each figure is maximized, contains subplots in a grid (max 4 per row),
    %   a single shared colorbar, and a single legend. The color on the mesh
    %   represents the normalized E-field magnitude. It also visualizes
    %   any restricted vertex areas defined in the targets.
    %
    %   Args:
    %       targeting_model (struct): A struct containing model information,
    %                                 including the mesh, ROI data, and targets.
    %       results (cell array): A cell array of result structs, each
    %                             containing weights and target information.

    % --- Initial Setup ---
    num_CS_plots = length(targeting_model.CS_targets);
    num_TS_plots = length(targeting_model.TS_targets);
    num_total_plots = length(results);

    if ~isfield(targeting_model,'ROI')
        ROI = 1:size(targeting_model.mesh.vertices,1);
    else
       ROI = targeting_model.ROI; 
    end

    % Pre-calculate indices for quiver arrows to reduce redundant calculations
    arrow_inds = subsample_mesh(targeting_model.mesh.vertices);

    % --- Plot CS Targets ---
    if num_CS_plots > 0
        % Create a new maximized figure for CS targets
        figure('Name', 'CS Targets', 'Color', 'w', 'WindowState', 'maximized');
        
        % Determine layout grid, with a maximum of 4 columns
        n_cols_cs = min(num_CS_plots, 4);
        n_rows_cs = ceil(num_CS_plots / n_cols_cs);
        tl_cs = tiledlayout(n_rows_cs, n_cols_cs, 'TileSpacing', 'compact', 'Padding', 'loose');
        sgtitle(tl_cs, sprintf('Conditioning Stimulus (CS) Targeting Results\n'), 'FontSize', 16, 'FontWeight', 'bold');

        legend_handles_cs = []; % To store handles for the legend
        legend_labels_cs = {};  % To store labels for the legend

        % Loop through and plot each CS result
        for i = 1:num_CS_plots
            ax = nexttile;
            % Call the helper function to draw the subplot
            plot_handles = plot_single_result(ax, results{i}, targeting_model, arrow_inds);
            
            % Store handles and labels from the first plot for the legend
            if i == 1
                legend_handles_cs = [plot_handles.target, plot_handles.result, plot_handles.max_efield];
                legend_labels_cs = {'Target', 'Realized', 'Max E-Field'};
                if isfield(plot_handles, 'restrict') && isgraphics(plot_handles.restrict)
                    legend_handles_cs(end+1) = plot_handles.restrict;
                    legend_labels_cs{end+1} = 'CS Restrict Area';
                end
            end
        end
        
        % Add a single colorbar for the entire figure
        cb = colorbar(ax);
        cb.Layout.Tile = 'east'; % Place colorbar to the east of the tiles
        cb.Label.String = 'Normalized E-Field Magnitude';
        cb.Label.FontSize = 12;

        % Add a single legend for the entire figure
        if ~isempty(legend_handles_cs)
            lgd = legend(legend_handles_cs, legend_labels_cs, 'Location', 'southoutside');
            lgd.Layout.Tile = 'south'; % Place legend below the tiles
            lgd.Orientation = 'horizontal';
        end
    end

    % --- Plot TS Targets ---
    if num_TS_plots > 0
        % Create a new maximized figure for TS targets
        figure('Name', 'TS Targets', 'Color', 'w', 'WindowState', 'maximized');
        
        % Determine layout grid, with a maximum of 4 columns
        n_cols_ts = min(num_TS_plots, 4);
        n_rows_ts = ceil(num_TS_plots / n_cols_ts);
        tl_ts = tiledlayout(n_rows_ts, n_cols_ts, 'TileSpacing', 'compact', 'Padding', 'loose');
        
        legend_handles_ts = []; % To store handles for the legend
        legend_labels_ts = {};  % To store labels for the legend

        % Loop through and plot each TS result
        for i = 1:num_TS_plots
            result_idx = num_CS_plots + i;
            if result_idx > num_total_plots; continue; end % Safety check
            
            ax = nexttile;
            % Call the helper function to draw the subplot
            plot_handles = plot_single_result(ax, results{result_idx}, targeting_model, arrow_inds);
            
            % Store handles and labels from the first plot for the legend
            if i == 1
                legend_handles_ts = [plot_handles.target, plot_handles.result, plot_handles.max_efield];
                legend_labels_ts = {'Target', 'Realized', 'Max E-Field'};
                if isfield(plot_handles, 'restrict') && isgraphics(plot_handles.restrict)
                    legend_handles_ts(end+1) = plot_handles.restrict;
                    legend_labels_ts{end+1} = 'TS Restrict Area';
                end
            end
        end

        % Add a single colorbar for the entire figure
        cb = colorbar(ax);
        cb.Layout.Tile = 'east';
        cb.Label.String = 'Normalized E-Field Magnitude';
        cb.Label.FontSize = 12;

        % Add a single legend for the entire figure
        if ~isempty(legend_handles_ts)
            lgd = legend(legend_handles_ts, legend_labels_ts, 'Location', 'southoutside');
            lgd.Layout.Tile = 'south';
            lgd.Orientation = 'horizontal';
        end
        
        sgtitle(tl_ts, sprintf('Test Stimulus (TS) Targeting Results\n'), 'FontSize', 16, 'FontWeight', 'bold');
    end
end


function plot_handles = plot_single_result(ax, result, targeting_model, arrow_inds)
    %PLOT_SINGLE_RESULT Helper function to plot one targeting result in a given axes.
    %
    %   This function encapsulates the logic for plotting the mesh, E-field,
    %   target/result vectors, and restricted areas for a single simulation outcome.
    %
    %   Args:
    %       ax (handle): The axes handle to draw on.
    %       result (struct): The result struct for a single target.
    %       targeting_model (struct): The main model struct.
    %       arrow_inds (array): Indices for placing E-field quiver arrows.
    %
    %   Returns:
    %       plot_handles (struct): A struct containing handles to the plotted
    %                              elements needed for the legend.

    % --- Initialize plot_handles ---
    plot_handles = struct();

    % --- E-Field Calculation ---
    mesh = targeting_model.mesh;

    if ~isfield(targeting_model,'ROI')
        ROI = 1:size(targeting_model.mesh.vertices,1);
    else
       ROI = targeting_model.ROI; 
    end
    
    N = mean(targeting_model.mesh.normals(ROI,:),1,'omitnan');
    N = N/norm(N);
    
    % Calculate the E-field based on the optimized weights
    E_ROI = squeeze(sum(targeting_model.efield_set .* result.weights, 1));
    E = zeros(size(mesh.vertices));
    E(ROI, :) = E_ROI;

    % Calculate magnitude and normalize it
    E_mag = vecnorm(E, 2, 2);
    E_mag_max = max(E_mag);
    if E_mag_max > 0
        E_mag_normalized = E_mag / E_mag_max;
    else
        E_mag_normalized = E_mag; % Avoid division by zero
    end
    [~, E_mag_max_ind] = max(E_mag);

    % --- Plotting ---
    hold(ax, 'on');

    % 1. Plot the mesh with E-field magnitude as color
    patch(ax, 'Faces', mesh.faces, 'Vertices', mesh.vertices, ...
        'FaceVertexCData', E_mag_normalized, 'FaceColor', 'interp', 'LineStyle', 'none');

    % 2. Plot E-field direction arrows (quivers)
    quiver3(ax, mesh.vertices(arrow_inds, 1), mesh.vertices(arrow_inds, 2), mesh.vertices(arrow_inds, 3), ...
        E(arrow_inds, 1), E(arrow_inds, 2), E(arrow_inds, 3), ...
        0.5, "filled", 'Color', [0.7, 0.7, 0.7], 'MaxHeadSize', 1);

    % 3. Plot the target location and direction
    target_pos = result.inputs.pos;
    target_dir = result.inputs.direction;
    plot3(ax, target_pos(1), target_pos(2), target_pos(3), '.', 'Color', "#D95319", 'MarkerSize', 30);
    q_target = quiver3(ax, target_pos(1), target_pos(2), target_pos(3), ...
        target_dir(1), target_dir(2), target_dir(3), ...
        0.01, 'filled', 'Color', '#D95319', 'LineWidth', 2, 'MaxHeadSize', 1);

    % 4. Plot the realized (achieved) stimulation centroid and direction
    realized_pos = result.realized_target.p;
    if dot(N, realized_pos) < 0
        N = -N;
    end
    lifted_realized_pos = realized_pos  + N*0.01;
    realized_dir = result.realized_target.Dir;
    plot3(ax, lifted_realized_pos(1), lifted_realized_pos(2), lifted_realized_pos(3), '.', 'Color', "#0072BD", 'MarkerSize', 30);
    q_result = quiver3(ax, lifted_realized_pos(1), lifted_realized_pos(2), lifted_realized_pos(3), ...
        realized_dir(1), realized_dir(2), realized_dir(3), ...
        0.015, 'filled', 'Color', "#0072BD", 'LineWidth', 2, 'MaxHeadSize', 1);
    plot3([realized_pos(1),lifted_realized_pos(1)],[realized_pos(2),lifted_realized_pos(2)],[realized_pos(3),lifted_realized_pos(3)],'-','Color',"#0072BD",'LineWidth',2)
        

    % 5. Plot the point of maximum E-field
    max_pos = mesh.vertices(E_mag_max_ind, :);
    max_dir_norm = E(E_mag_max_ind, :) / norm(E(E_mag_max_ind, :));
    plot3(ax, max_pos(1), max_pos(2), max_pos(3), '.', 'Color', "black", 'MarkerSize', 30);
    q_max = quiver3(ax, max_pos(1), max_pos(2), max_pos(3), ...
        max_dir_norm(1), max_dir_norm(2), max_dir_norm(3), ...
        0.01, 'filled', 'Color', "black", 'LineWidth', 2, 'MaxHeadSize', 1);
    
    % 6. Plot the restricted area, if it exists
    if isfield(result.inputs, 'RestrictEF') && ~isempty(result.inputs.RestrictEF)
        if isfield(targeting_model, 'ROI')
            restrict_vertices = targeting_model.mesh.vertices(ROI(result.inputs.RestrictEF), :);
            restrict_color = '#EDB120';
            
            p_restrict = scatter3(ax, restrict_vertices(:,1), restrict_vertices(:,2), restrict_vertices(:,3), ...
                10, 'filled', 'MarkerFaceColor', restrict_color,'MarkerEdgeColor','none');
            plot_handles.restrict = p_restrict; % Store handle for the legend
        end
    end

    hold(ax, 'off');

    % --- Axes and View Configuration ---
    axis(ax, 'tight', 'equal', 'off');
    if isfield(targeting_model.mesh,'view_angle')
        view(targeting_model.mesh.view_angle);
    end
    colormap(ax, viridis); % Use a perceptually uniform colormap
    clim(ax, [0 1]); % IMPORTANT: Set consistent color limits for all subplots

    % Set lighting for better 3D visualization
    camlight(ax, 'headlight');
    lighting(ax, 'gouraud');
    material(ax, 'dull');

    % Set title with error metrics
    title_str = sprintf('%s\nLocation Error: %.1f mm, Angle Error: %.0f deg\nE-field strength in restricted area: %.0f%%', ...
        result.label, result.err.location, result.err.angle, result.realized_target.restriction_strength*100);
    title(ax, title_str, 'FontSize', 10);

    % --- Return Handles for Legend ---
    plot_handles.target = q_target;
    plot_handles.result = q_result;
    plot_handles.max_efield = q_max;
end
