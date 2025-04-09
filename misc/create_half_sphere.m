function halfSphere = create_half_sphere(radius, numPoints)
    % Creates a half-spherical surface (upper hemisphere)
    % Inputs:
    %   radius - radius of the sphere (default: 1)
    %   numPoints - number of points along each dimension (default: 20)
    % Outputs:
    %   halfSphere - structure containing:
    %       .vertices - 3D coordinates of surface points
    %       .faces - triangulation connectivity
    %       .normals - vertex normals
    
    % Set default values if not provided
    if nargin < 1
        radius = 1;
    end
    if nargin < 2
        numPoints = 20;
    end
    
    % Create spherical coordinates for upper hemisphere
    [theta, phi] = meshgrid(linspace(0, 2*pi, numPoints), ...    % azimuthal angle (full circle)
                           linspace(0, pi/2, numPoints/2));     % polar angle (0 to pi/2 for half sphere)
    
    % Convert to Cartesian coordinates
    x = radius * sin(phi) .* cos(theta);
    y = radius * sin(phi) .* sin(theta);
    z = radius * cos(phi);
    
    % Create vertex array
    vertices = unique([x(:), y(:), z(:)],'rows');
    
    % Create triangulation
    triFaces = delaunay(vertices(:,1),vertices(:,2));
    
    % Calculate vertex normals (same as vertex positions for a sphere centered at origin)
    normals = vertices ./ radius;
    
    % Store in structure
    halfSphere.vertices = vertices;
    halfSphere.faces = triFaces;
    halfSphere.normals = normals;
end