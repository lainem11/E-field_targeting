function results = optimize_efield_complex_geom(pos,direction,mesh,E_set,varargin)
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
addRequired(p,'E_set',@(x) isnumeric(x))
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
closest_vertex = pos2ind(pos,mesh.vertices);
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
        plane_origin = pos;
        % Project mesh vertices to 2D relative to the plane_origin
        [mesh_vertices, projection_basis] = projectAndFlatten(mesh.vertices,N,plane_origin);
        % The target in the new 2D system is the origin [0,0]
        target.p = [0,0];
        % Find closest index in mesh
        loc_i = pos2ind(pos,mesh.vertices);
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

% Reconstruct target in 3D
realized_target.Dir = dir;
switch stimMetric
    case 'Max'
        [~,~,loc_i] = E_to_mag(E);
        realized_target.p = mesh.vertices(loc_i,:);
    case 'WCOG'
        % Transform the 2D location 'loc' using the basis
        loc_3d_shifted = (projection_basis * loc')';
        % Add the plane's origin back to get the final 3D coordinates
        realized_target.p = loc_3d_shifted + plane_origin;
end

% Normalize weights to produce E-field of 1.
x = x/norm(E(E_max_ind,:));

% Calculate goodness of E-field restriction
if ~isempty(restrictEF)
    realized_target.restriction_strength = mean(sqrt(sum(E(restrictEF,:).^2,2)))/norm(E(E_max_ind,:));
else
    realized_target.restriction_strength = NaN;
end

% Store output
results.weights = x';
results.realized_target = realized_target;
results.inputs = p.Results;
results.err = err;

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
    function [coordinates, basis] = projectAndFlatten(V, N, p0)
        % Projects 3D vertices onto a plane defined by normal N and point p0.
        %
        % Inputs:
        %   V  - Matrix of shape (n_vertices, 3) with 3D vertex coordinates.
        %   N  - Vector of length 3 representing the plane's normal vector.
        %   p0 - Vector of length 3 representing the origin point of the plane.
        %
        % Output:
        %   coordinates - Matrix of shape (n_vertices, 2) with 2D coordinates.
        %   basis       - The 3x2 orthonormal basis for the plane.
    
        % Ensure N is a column vector (3x1)
        N = N(:);
    
        % Step 1: Shift vertices to be relative to the plane's origin p0
        V_shifted = V - p0;
    
        % Step 2: Project the shifted vertices onto the plane
        dot_products = V_shifted * N;
        N_norm_sq = N' * N;
        % Projected points (still in 3D, relative to p0)
        P_shifted = V_shifted - (dot_products / N_norm_sq) .* N';
    
        % Step 3: Find an orthonormal basis {U, W} on the plane
        es = eye(3);
        N_rep = repmat(N, 1, 3);
        cross_products = cross(N_rep, es', 1);
        [~, max_idx] = max(vecnorm(cross_products, 2, 1));
        U = cross_products(:, max_idx);
        U = U / norm(U);
        W = cross(N, U);
        W = W / norm(W);
        basis = [U, W]; % This is the 3x2 basis matrix
    
        % Step 4: Compute 2D coordinates by projecting onto the new basis
        coordinates = P_shifted * basis;
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
