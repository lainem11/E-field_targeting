function fieldVectors = generate_ring_field(halfSphere, theta, phi)
    % Generates a ring-like field on a half-sphere
    % Input:
    %   halfSphere - structure with vertices (p) and normals (nn)
    %   theta      - parameter controlling the doughnut center along x-axis (default: 0)
    %   phi        - rotation angle around z-axis in radians (default: 0)
    % Output:
    %   fieldVectors - toroidal electric field vectors at each vertex
    
    % Default theta and phi to 0 if not provided
    if nargin < 2
        theta = 0;
    end
    if nargin < 3
        phi = 0;
    end

    % Extract vertices and determine radius
    vertices = halfSphere.vertices;
    numVertices = size(vertices, 1);
    radius = max(vertices(:, 3));  % Top at z = R
    
    % Define base offset distance
    offset = radius / 2;
    
    % Compute toroid center in xy-plane
    x01 = offset * sin(theta);    % Base x-coordinate adjusted by theta
    cx = x01 * cos(phi);          % Rotate x01 around z-axis by phi
    cy = x01 * sin(phi);
    
    % Toroid parameters
    R_toroid = radius / 2;
    sigma = radius / 2;
    
    % Initialize field vectors
    fieldVectors = zeros(numVertices, 3);
    eps = 1e-10;              % Singularity threshold
    
    % Compute field at each vertex
    for i = 1:numVertices
        x = vertices(i, 1);
        y = vertices(i, 2);
        z = vertices(i, 3);
        
        % --- Toroid ---
        rho_offset1 = sqrt((x - cx)^2 + (y - cy)^2);  % Distance to new center
        if rho_offset1 < eps
            field1 = [0 0 0];
        else
            % Azimuthal direction around the shifted axis
            baseField1 = [-(y - cy) / rho_offset1, (x - cx) / rho_offset1, 0];
            % Magnitude
            fieldMag1 = exp(-((rho_offset1 - R_toroid)^2) / sigma^2) * (z / radius);
            field1 = fieldMag1 * baseField1;
        end
        
        % Sum the fields
        fieldVectors(i, :) = field1;
        
        % Project onto surface
        normal = halfSphere.normals(i, :);
        normalComp = dot(fieldVectors(i, :), normal);
        fieldVectors(i, :) = fieldVectors(i, :) - normalComp * normal;
    end

    % Normalize field
    fieldVectors = fieldVectors/max(sqrt(sum(fieldVectors.^2,2)));
end