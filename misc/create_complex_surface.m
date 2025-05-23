function complexSurface = create_complex_surface()
    % Creates a complex surface.
    %
    % Outputs:
    %   complexSurface - structure containing:
    %       .vertices - 3D coordinates of surface points
    %       .faces - triangulation connectivity
    %       .normals - vertex normals
    
    numPoints = 40;
    length_constant = 0.07;
    
    % Create grid in xy-plane
    [X, Y] = meshgrid(linspace(-length_constant, length_constant, numPoints), linspace(-length_constant, length_constant, numPoints));
    Z = zeros(size(X));
    
    % Apply multiple sine waves for displacement
    numLowFreqWaves = 3;
    numHighFreqWaves = 0;
    
    % Low frequency waves for large-scale undulations
    for w = 1:numLowFreqWaves
        f = (rand() * 1 + 1)*10; % frequency between 1 and 2
        a = (rand() * 1 + 1)*length_constant*0.05; % amplitude between 0.1 and 0.2
        phi = rand() * 2 * pi; % random direction
        dx = cos(phi);
        dy = sin(phi);
        phase = rand() * 2 * pi; % random phase
        Z = Z + a * sin(2*pi*f*(dx*X + dy*Y) + phase);
    end
    
    % High frequency waves for finer details
    for w = 1:numHighFreqWaves
        f = rand() * 2 + 3; % frequency between 3 and 5
        a = rand() * 0.05 + 0.05; % amplitude between 0.05 and 0.1
        phi = rand() * 2 * pi; % random direction
        dx = cos(phi);
        dy = sin(phi);
        phase = rand() * 2 * pi; % random phase
        Z = Z + a * sin(2*pi*f*(dx*X + dy*Y) + phase);
    end
    
    % Create vertex array
    vertices = [X(:), Y(:), Z(:)];

    % Shift z-coordinates to positive axis
    vertices(:,3) = vertices(:,3) - min(vertices(:,3));
    
    % Create triangulation based on grid
    m = numPoints;
    n = numPoints;
    faces = [];
    for i = 1:m-1
        for j = 1:n-1
            v1 = (i-1)*n + j;
            v2 = (i-1)*n + j + 1;
            v3 = i*n + j;
            v4 = i*n + j + 1;
            faces = [faces; v1, v2, v3];
            faces = [faces; v2, v4, v3];
        end
    end
    
    % Compute face normals
    numFaces = size(faces, 1);
    faceNormals = zeros(numFaces, 3);
    for f = 1:numFaces
        v1 = vertices(faces(f, 1), :);
        v2 = vertices(faces(f, 2), :);
        v3 = vertices(faces(f, 3), :);
        vec1 = v2 - v1;
        vec2 = v3 - v1;
        normal_f = cross(vec1, vec2);
        norm_f = norm(normal_f);
        if norm_f > 0
            faceNormals(f, :) = normal_f / norm_f;
        else
            faceNormals(f, :) = [0, 0, 1]; % Default for degenerate triangles
        end
    end
    
    % Compute vertex normals by averaging face normals
    normals = zeros(size(vertices));
    for f = 1:numFaces
        for v = 1:3
            normals(faces(f, v), :) = normals(faces(f, v), :) + faceNormals(f, :);
        end
    end
    for i = 1:size(vertices, 1)
        norm_n = norm(normals(i, :));
        if norm_n > 0
            normals(i, :) = normals(i, :) / norm_n;
        else
            normals(i, :) = [0, 0, 1]; % Default normal
        end
    end
    
    % Store in structure
    complexSurface.vertices = vertices;
    complexSurface.faces = faces;
    complexSurface.normals = normals;
end