function sparse_indices = subsample_mesh(nodes, spacing)    
    % Number of nodes
    num_nodes = size(nodes, 1);
    
    % Initialize mask to track selected nodes
    selected = false(1, num_nodes);
    
    % Start with the first node
    selected(1) = true;
    
    % Iterate through nodes and select those spaced apart by 'spacing'
    for i = 2:num_nodes
        % Compute distance to all previously selected nodes
        distances = sqrt(sum((nodes(selected,:) - nodes(i,:)).^2, 2));
        
        % Select node if it is sufficiently far from all selected nodes
        if all(distances > spacing)
            selected(i) = true;
        end
    end
    
    sparse_indices = find(selected);
end