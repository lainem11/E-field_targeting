function results = mTMS_optWeights_3D(pos,direction,cortex,EFs,varargin)
%mTMS_optWeights_3D(pos_ind,direction,cortex,EFs,stimMetric,restrictEF,failSafe,savedir)
% Calculates weights for each field given in "EFs" that together produce
% maximum at a point at "index" with the speciefied field "direction".
%
% Inputs: 
%         pos - Vector of shape (3) with stimulation location coordinates
%         direction - Vector of shape (3) with stimulation direction
%         cortex - Struct with the computational mesh. 
%             p: Matrix of shape (n_vertices,3) with vertex coordinates
%             nn: Matrix of shape (n_vertices,3) with vertex normals
%         EFs - A cell array of shape (n_coils) containing E-field matrices of shape (n_vertices,3)
%         varargin - Options for optimization, file saving, and plotting
%         StimMetric - Defines the measure of stimulation location given an E-field.
%             'Centroid': (default) calculates a weighted center of mass.
%             'Max': uses the location of the maximum norm.
%         RestrictEF - Avoids E-field magnitude at the specified indices. Empty by default.
%         SaveDir - Save directory. Empty by default (no saving)
%         Objective - After constraints are met, the objective is optimized. 
%             'minEnergy': Minimize pulse energy (default)
%             'Focality': Maximize E-field focality
%         DistConstr - Number in meters to constraint stimulation
%         location within a range of the specified target. Default is 0.002
%         AngleConstr - Number in degrees to constraint stimulation
%         direction within a range of the specified target. Default is 5.
%         PlotFlag - Enable(=1)/disable(=0) plotting. Default is 0.
%
% OUTPUT:
%         results - Struct containing optimization results.
%             weights: Vector of shape (n_coils) of the 
%
% Computation takes typically 5-20s. The Genetic Algorithm used is stochastic:
% set a seed to produce deterministic output.
%
%% Initialize
rng('default') % Set seed for reproducability
defaultStimMetric = 'Centroid';
defaultRestrictEF = [];
defaultSaveDir = [];
defaultObjective = 'minEnergy';
defaultDistConstr = 0.002;
defaultAngleConstr = 10;
defaultPlotFlag = 0;

p = inputParser;
addRequired(p,'pos',@(x) isnumeric(x))
addRequired(p,'direction',@(x) isnumeric(x) && length(x)==3)
addRequired(p,'cortex',@(x) isstruct(x))
addRequired(p,'EFs',@(x) iscell(x))
addParameter(p,'StimMetric',defaultStimMetric, @(x) ischar(x))
addParameter(p,'RestrictEF',defaultRestrictEF, @(x) isnumeric(x))
addParameter(p,'SaveDir',defaultSaveDir,@(x) ischar(x) || isstring(x))
addParameter(p,'Objective',defaultObjective,@(x) ischar(x))
addParameter(p,'DistConstr',defaultDistConstr, @(x) isnumeric(x))
addParameter(p,'AngleConstr',defaultAngleConstr, @(x) isnumeric(x))
addParameter(p,'PlotFlag',defaultPlotFlag, @(x)x==0 || x==1)

parse(p,pos,direction,cortex,EFs,varargin{:})
stimMetric = p.Results.StimMetric;
restrictEF = p.Results.RestrictEF;
savedir = p.Results.SaveDir;
objectiveType = p.Results.Objective;
distConstr = p.Results.DistConstr;
angleConstr = p.Results.AngleConstr;
maxConstr = 0.01;   % E-field maximum within 1 cm of stimulation location
plotFlag = p.Results.PlotFlag;

% Calculate average normal within 3 cm radius around the target
masked_indices = sqrt(sum((cortex.p-pos).^2,2)) < 0.03;
N = mean(cortex.nn(masked_indices,:),1,'omitnan');
N = N/norm(N);

% Set target
direction = direction/norm(direction);
switch stimMetric
    case 'Max'
        target.p = pos;
        target.Dir = direction;
    case 'Centroid'
        % Project cortex to the average normal plane and transform to 2D.
        projMesh2D = projectAndFlatten(cortex.p,N);
        target.p = projectAndFlatten(pos,N);
        % Find closest index in mesh
        [~,loc_i] = min(sqrt(sum((cortex.p-pos).^2,2)));
        target.ind = loc_i;
        target.Dir = direction;
    case 'Threshold_centroid'
        target.p = pos;
        target.Dir = direction;
    otherwise
        warning('Unrecognized StimMetric.')
end

if plotFlag
    figure;
    tiledlayout(1,2)
end

%% Optimize weights
Nc = length(EFs);
% 'Quick' optimization of direction at target, to set a realizable goal.
x0 = ones(Nc,1)*0.5;
options=optimoptions("patternsearch","Display","off");
x_dir = patternsearch(@optimize_direction,x0,[],[],[],[],[],[],[],options);
E_surr = 0;
for i = 1:Nc
    E_surr = E_surr + double(EFs{i}).*x_dir(i);
end
E_surr_norm = E_surr./sqrt(sum(E_surr.^2,2));
E_surr_dir = E_surr_norm(target.ind,:);

if plotFlag
    nexttile; hold on
    quiver3(cortex.p(:,1),cortex.p(:,2),cortex.p(:,3),E_surr(:,1),E_surr(:,2),E_surr(:,3),1,"filled",'Color',[1,0,0],'MaxHeadSize',1)
    % Add target point and direction
    q1=quiver3(pos(1),pos(2),pos(3),E_surr_dir(1),E_surr_dir(2),E_surr_dir(3),0.02,'filled','g','LineWidth',2,'MaxHeadSize',1);
    q2=quiver3(pos(1),pos(2),pos(3),target.Dir(1),target.Dir(2),target.Dir(3),0.02,'filled','b','LineWidth',2,'MaxHeadSize',1);
    q3=quiver3(pos(1),pos(2),pos(3),N(1),N(2),N(3),0.02,'filled','k','LineWidth',2,'MaxHeadSize',1);
    va = [-vectorAngle([0,-1,0],N),vectorAngle([-1,0,0],N)];
    view(va)
    legend([q1,q2,q3],'new target dir', 'orig target dir','normal dir');
    axis equal off
end

angle_diff = vectorAngle(E_surr_dir,target.Dir);
if angle_diff > 45
    warning(sprintf('Target direction %.0f degrees from what is possible',angle_diff))
    return
end
% Set optimized direction as a new target
target.Dir = E_surr_dir;

% Set bounds
lb = [];
ub = [];
%lb = ones(Nc,1)*(-1);
%ub = ones(Nc,1);

% Set optimisation options
creationOptions = optimoptions("fmincon","Algorithm","sqp");
options = optimoptions('ga','PopulationSize',100,'CrossoverFcn','crossoverlaplace',...
    'MaxStallGenerations',50,'EliteCount',15,'Display','off','CreationFcn',{@gacreationnonlinearfeasible,'SolverOpts',creationOptions},...
    'NonlinearConstraintAlgorithm','penalty','InitialPopulationRange',[-1;1]);

% Optimize weights
[x] = ga(@objectiveFcn,Nc,[],[],[],[],lb,ub,@locNdirConsFcn,options);

% Calculate error
E = 0;
for i = 1:length(EFs)
    E = E + double(EFs{i}).*x(i);
end
Emag = sqrt(sum(E.^2,2));
[Emax,Emax_ind] = max(Emag);
Emagn = Emag/Emax;
[loc,loc_i] = stimulationLoc(Emagn,stimMetric);
err.location = 1000*sqrt(sum((target.p-loc).^2,2)); % mm
err.angle = vectorAngle(target.Dir,E(loc_i,:));

%[c,ceq] = locNdirConsFcn(x);
fprintf('Constraints: Loc: %.2f mm, Angle: %.2f deg, Max: %.2f mm.   Result: Loc: %.2f mm, Angle: %.2f deg, Max: %.2f.\n',distConstr*1000,angleConstr,10,err.location,err.angle,1000*norm(projMesh2D(Emax_ind,:)-loc))
results.weights = x';
results.target = target;
results.inputs = p.Results;
results.err = err;
results.loc = loc;
results.loc_i = loc_i;
results.N = N;

if ~isempty(savedir)
    if not(isfolder(savedir))
        mkdir(savedir);
    end
    filename = fullfile(savedir,['optWeights_3D_results_' datestr(datetime('now'),'yymmddHHMMSS') '.mat']);
    results.filename = filename;
    save(filename,'results')
    disp('Optimization results saved.')
end

if plotFlag
    dir = E(loc_i,:)/sqrt(sum(E(loc_i,:).^2,2));
    nexttile; hold on
    switch stimMetric
        case 'Centroid'
            plot(projMesh2D(:,1),projMesh2D(:,2),'.k')
            p1 = plot(target.p(1),target.p(2),'.b','MarkerSize',30);
            p2 = plot(loc(1),loc(2),'.g','MarkerSize',25);
            p3 = plot(projMesh2D(Emax_ind,1),projMesh2D(Emax_ind,2),'.r','MarkerSize',20);
            legend([p1,p2,p3],'Target','Result','Max')
            title('2D Centroid location')
    end
end

%% Functions
    function ThetaInDegrees = vectorAngle(u,v)
        CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
        ThetaInDegrees = real(acosd(CosTheta));
    end

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

    function [loc,loc_i] = stimulationLoc(Emagn,stimMetric)
        switch stimMetric
            case 'Max'
                [~,loc_i] = max(Emagn);
                loc = cortex.p(loc_i,:);
            case 'Centroid'
                weightedEF = Emagn.^10;
                loc = sum(projMesh2D.*weightedEF,1)/sum(weightedEF);
                % Select nearest mesh vertex
                [~,loc_i] = min(sqrt(sum((projMesh2D-loc).^2,2)));
            case 'Threshold_centroid'
                th_indices = find(Emagn > 0.90);
                loc = sum(cortex.p(th_indices,:).*Emagn(th_indices),1)/sum(Emagn(th_indices));
                [~,loc_i] = min(sqrt(sum((cortex.p-loc).^2,2)));
        end       
    end

    function obj = objectiveFcn(x)
        E_f = 0;
        for k = 1:length(EFs)
            E_f = E_f + double(EFs{k}).*x(k);
        end
        Emag_f = sqrt(sum(E_f.^2,2));
        Emagn_f = Emag_f/max(Emag_f);
        switch objectiveType
            case 'Focality'
                if ~isempty(restrictEF)
                    penalty = sum(Emagn_f(restrictEF).^2);
                else
                    penalty = 0;
                end
                obj = sum(Emagn_f.^2)+penalty;
            case 'minEnergy'
                if ~isempty(restrictEF)
                    penalty = mean(Emagn_f(restrictEF))*10;
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
        E_f = 0;
        for k = 1:length(EFs)
            E_f = E_f + double(EFs{k}).*x(k);
        end
        Emag_f = sqrt(sum(E_f.^2,2));
        [Emag_f_max,E_mag_f_ind] = max(Emag_f);
        Emagn_f = Emag_f/Emag_f_max;
        [loc_f,loc_i_f] = stimulationLoc(Emagn_f,stimMetric);
        d2target = sqrt(sum((target.p-loc_f).^2,2));
        Edir_f = E_f(E_mag_f_ind,:);
        Edir_fn = Edir_f/norm(Edir_f);
        Diff_dir = vectorAngle(Edir_fn,target.Dir);

        % Define constraints
        ceq = [];
        c(1) = (d2target - distConstr)*1000;             % Location error < distance constraint
        c(2) = Diff_dir - angleConstr;    % Angle error < angle constraint
        c(3) = 1000*(sqrt(sum((cortex.p(E_mag_f_ind,:)-cortex.p(loc_i_f,:)).^2,2)) - maxConstr) ; % Distance between stimulation location and E-field maximum < 10 mm.
    end

    function Diff_dir = optimize_direction(x)
        E_f = 0;
        for k = 1:length(EFs)
            E_f = E_f + double(EFs{k}).*x(k);
        end
        Edir_f = E_f(target.ind,:);
        Edir_fn = Edir_f/norm(Edir_f);
        Diff_dir = vectorAngle(Edir_fn,target.Dir);
    end

end
