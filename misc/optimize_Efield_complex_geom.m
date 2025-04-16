function results = optimize_Efield_complex_geom(pos,direction,mesh,E_set,varargin)
% Optimizes the weights for each field in the E_set such that their sum
% focuses E-field at the specified target. The computation mesh can be a
% complex 3D shape.
%
% Inputs:
%         pos - Vector of shape (3) with E-field focus point coordinates.
%         direction - Vector of shape (3) with E-field direction at the focus.
%         mesh - Struct with the computational mesh.
%             vertices: Matrix of shape (n_vertices,3) with vertex coordinates.
%             normals: Matrix of shape (n_vertices,3) with mesh normals.
%         E_set - A matrix of shape (n_E_set,n_vertices, 3) with E-field values for each E-field in the set.
%         varargin - Options for optimization, file saving, and plotting.
%             StimMetric - Defines the measure of stimulation location given an E-field.
%                 'WCOG': Calculates a weighted center of gravity (default).
%                 'Max': Uses the location of the maximum norm.
%             RestrictEF - Avoids E-field magnitude at the specified indices. Empty by default.
%             SaveDir - Save directory. Empty by default (no saving).
%             Objective - After constraints are met, the objective is optimized.
%                 'Focality': Maximize E-field focality (default).
%                 'minEnergy': Minimize energy.
%             DistConstr - Number in meters to constraint stimulation location
%                 within a range of the specified target. Default is 0.002.
%             AngleConstr - Number in degrees to constraint stimulation direction
%                 within a range of the specified target. Default is 5.
%
% Output:
%         results - Struct containing optimization results.
%             weights: Vector of shape (n_coils) with E-field weights.
%             inputs: Struct of the function inputs.
%             err: Struct of final stimulation location and angle errors.
%             N: Vector of the mesh average normal around the target.
%
% Computation takes typically 5-20s. The optimization algorithm used is stochastic:
% set a seed to produce deterministic output.

myStream = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(myStream);

%% Initialize
defaultStimMetric = 'WCOG';
defaultRestrictEF = [];
defaultSaveDir = [];
defaultObjective = 'Focality';
defaultDistConstr = 0.001;
defaultAngleConstr = 5;

p = inputParser;
addRequired(p,'pos',@(x) isnumeric(x))
addRequired(p,'direction',@(x) isnumeric(x) && length(x)==3)
addRequired(p,'mesh',@(x) isstruct(x))
addRequired(p,'EFs',@(x) isnumeric(x))
addParameter(p,'StimMetric',defaultStimMetric, @(x) ischar(x))
addParameter(p,'RestrictEF',defaultRestrictEF, @(x) isnumeric(x))
addParameter(p,'SaveDir',defaultSaveDir,@(x) ischar(x) || isstring(x))
addParameter(p,'Objective',defaultObjective,@(x) ischar(x))
addParameter(p,'DistConstr',defaultDistConstr, @(x) isnumeric(x))
addParameter(p,'AngleConstr',defaultAngleConstr, @(x) isnumeric(x))

parse(p,pos,direction,mesh,E_set,varargin{:})
stimMetric = p.Results.StimMetric;
restrictEF = p.Results.RestrictEF;
savedir = p.Results.SaveDir;
objectiveType = p.Results.Objective;
distConstr = p.Results.DistConstr;
angleConstr = p.Results.AngleConstr;
maxConstr = 0.01;   % E-field maximum within 1 cm of stimulation location
E_set = double(E_set);

% Calculate average normal within 3 cm radius around the target
masked_indices = sqrt(sum((mesh.vertices-pos).^2,2)) < 0.03;
N = mean(mesh.normals(masked_indices,:),1,'omitnan');
N = N/norm(N);

% To intepret the E-field direction, the E-field is mapped to 2D subspace
% using the E-field principal components at a specific vertex. This 
% intepretation discards the most energy-intensive E-field dimension.
% The subspace is defined from the closest vertex the target position.
[~,closest_vertex] = min(sqrt(sum((mesh.vertices-pos).^2,2)));
E_vertex = squeeze(E_set(:,closest_vertex,:));
E_centered = E_vertex - sum(E_vertex,1);
[~, ~, V] = svd(E_centered,"econ");
% Select two largest principal components
V = V(:,1:2);   % Shape: (3,2)

% Set target direction
direction = direction/norm(direction);
direction = direction * V;

switch stimMetric
    case 'Max'
        mesh_vertices = mesh.vertices;
        target.p = mesh.vertices(closest_vertex,:);
        target.Dir = direction;
    case 'WCOG'
        % Project mesh to the average normal plane and transform to 2D.
        mesh_vertices = projectAndFlatten(mesh.vertices,N);
        target.p = projectAndFlatten(pos,N);
        % Find closest index in mesh
        [~,loc_i] = min(sqrt(sum((mesh.vertices-pos).^2,2)));
        target.ind = loc_i;
        target.Dir = direction;
    otherwise
        warning('Unrecognized StimMetric.')
end

%% Optimize weights

Nc = size(E_set,1);

% Set bounds
lb = [];
ub = [];

% Set optimisation options
creationOptions = optimoptions("fmincon","Algorithm","sqp");
options = optimoptions('ga','PopulationSize',100,'CrossoverFcn','crossoverlaplace',...
    'MaxStallGenerations',50,'EliteCount',15,'Display','off','CreationFcn',{@gacreationnonlinearfeasible,'SolverOpts',creationOptions},...
    'NonlinearConstraintAlgorithm','penalty','InitialPopulationRange',[-1;1]);

% Optimize weights
[x] = ga(@objectiveFcn,Nc,[],[],[],[],lb,ub,@locNdirConsFcn,options);

% Calculate error
E = squeeze(sum(E_set.*x',1));
[loc,~,dir,E_max_ind] = stimulatedTarget(E,stimMetric);
E_dir_2d = dir * V;
err.location = 1000*sqrt(sum((target.p-loc).^2,2));         % mm
err.angle = calculate_vector_angle(target.Dir,E_dir_2d);    % deg
fprintf('Constraints: Loc: %.2f mm, Angle: %.2f deg, Max: %.2f mm.   Result: Loc: %.2f mm, Angle: %.2f deg, Max: %.2f.\n',distConstr*1000,angleConstr,10,err.location,err.angle,1000*norm(mesh_vertices(E_max_ind,:)-loc))

% Reconstruct target direction in 3D
target.Dir = V * direction';

% Normalize weights to produce E-field of 1.
x = x/norm(E(E_max_ind,:));

% Store output
results.weights = x';
results.target = target;
results.inputs = p.Results;
results.err = err;
results.N = N;

if ~isempty(savedir)
    if not(isfolder(savedir))
        mkdir(savedir);
    end
    filename = fullfile(savedir,['optimize_Efield_realistic_results_' datestr(datetime('now'),'yymmddHHMMSS') '.mat']);
    results.filename = filename;
    save(filename,'results')
    disp('Optimization results saved.')
end

%% Functions
    function coordinates = projectAndFlatten(V, N)
        % Projects 3D vertices onto a plane and maps to 2D.
        %
        % Inputs:
        %   V - Matrix of shape (n_vertices, 3) with 3D vertex coordinates.
        %   N - Vector of length 3 representing the plane's normal vector.
        %
        % Output:
        %   coordinates - Matrix of shape (n_vertices, 2) with 2D coordinates.

        % Ensure N is a column vector (3x1)
        N = N(:);

        % Step 1: Project vertices onto the plane
        dot_products = V * N;
        N_norm_sq = N' * N;
        % Project vertices: V - (V · N / ||N||^2) * N
        P = V - (dot_products / N_norm_sq) .* N';

        % Step 2: Find an orthonormal basis {U, V} on the plane
        % Define standard basis vectors (3x3 identity matrix)
        es = eye(3);
        % Repeat N into a (3x3) matrix, each column is N
        N_rep = repmat(N, 1, 3);
        cross_products = cross(N_rep, es', 1);  % (3x3)
        norms = vecnorm(cross_products, 2, 1);  % (1x3)
        % Find index of the largest norm
        [~, max_idx] = max(norms);
        % Select U as the cross product with the largest norm
        U = cross_products(:, max_idx);
        U = U / norm(U);
        % Compute V as N x U and normalize
        V = cross(N, U);
        V = V / norm(V);

        % Step 3: Compute 2D coordinates
        % Form basis matrix with U and V as columns (3x2)
        basis = [U, V];
        % Project P onto the basis to get 2D coordinates (n_vertices x 2)
        coordinates = P * basis;
    end

    function [loc,loc_i,dir,E_max_ind] = stimulatedTarget(E,stimMetric)
        switch stimMetric
            case 'Max'
                [~,~,loc_i] = E_to_mag(E);
                loc = mesh.vertices(loc_i,:);
                dir = E(loc_i,:);
                dir = dir/norm(dir);
                E_max_ind = loc_i;
            case 'WCOG'
                [E_mag,E_max,E_max_ind] = E_to_mag(E);
                E_mag_n = E_mag/E_max;
                [loc,loc_i] = calculate_WCOG(E_mag_n,mesh_vertices);
                dir = E(loc_i,:);
                dir = dir/norm(dir);
        end
    end

    function obj = objectiveFcn(x)
        E_f = squeeze(sum(E_set.*x',1));
        [E_mag,E_max,~] = E_to_mag(E_f);
        E_mag_n = E_mag/E_max;
        switch objectiveType
            case 'Focality'
                if ~isempty(restrictEF)
                    penalty = mean(E_mag_n(restrictEF).^2);
                else
                    penalty = 0;
                end
                obj = mean(E_mag_n.^2)+penalty;
            case 'minEnergy'
                if ~isempty(restrictEF)
                    penalty = mean(E_mag_n(restrictEF))*10;
                else
                    penalty = 0;
                end
                obj = sum((x/max(abs(x))).^2)+penalty;
            otherwise
                warning('Unkown objective type.')
                return
        end
    end

    function [c,ceq] = locNdirConsFcn(x)
        % Calculate Efield constraints
        E_f = squeeze(sum(E_set.*x',1));   
        [loc_f,loc_i_f,dir_f,E_max_ind_f] = stimulatedTarget(E_f,stimMetric);
        d2target = sqrt(sum((target.p-loc_f).^2,2));
        E_dir_2d_f = dir_f * V;
        diff_dir = calculate_vector_angle(E_dir_2d_f,target.Dir);

        % Define constraints
        ceq = [];
        c(1) = (d2target - distConstr)*1000;             % Location error < distance constraint
        c(2) = diff_dir - angleConstr;    % Angle error < angle constraint
        c(3) = 1000*(sqrt(sum((mesh.vertices(E_max_ind_f,:)-mesh.vertices(loc_i_f,:)).^2,2)) - maxConstr) ; % Distance between stimulation location and E-field maximum < 10 mm.
    end
end
