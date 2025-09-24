function results = optimize_efield_spherical(target_shift,vertices,E_set,varargin)
% Optimizes the weights for each field in the E_set such that their sum
% focuses E-field at the specified target. Assumes vertices defines a spherical mesh.
%
% Inputs: 
%       target_shift - Horizontal and vertical shift and rotation of EF. A target specified as [0,0,0] focuses 
%                      the E-field at coordinate (0,0,r) in the coordinate space of pos, where r is the radius 
%                      of a sphere. In general, a target [x,y,theta] translates the E-field focus x mm 
%                      horizontally, y mm vertically, and theta radians clockwise.            
%       vertices - Matrix Size: (n_vertices,3) of vertex coordinates on a spherical surface.
%       E_set - A matrix of shape (n_E_set,n_vertices, 3) with E-field values for each E-field in the set.
%       varargin - Options for the optimization.
%           stimMetric = Defines the measure of stimulation location given an E-field. 
%               'WCOG': Calculates a weighted center of gravity (default).
%               'Max': Uses the location of the maximum norm.
%           RestrictEF - Avoids E-field magnitude at the specified indices. Empty by default.
%           SaveDir - Save directory. Empty by default (no saving).
%           Objective - After constraints are met, the objective is optimized. 
%               'minEnergy': Minimize energy (default).
%               'Focality': Maximize E-field focality.
%           DistConstr - Number in meters to constraint stimulation location 
%               within a range of the specified target. Default is 0.001.
%           AngleConstr - Number in degrees to constraint stimulation direction 
%               within a range of the specified target. Default is 5.
% Output: 
%         results - Struct containing optimization results.
%             weights: Vector of shape (n_coils) with E-field weights.
%             target: Struct with the coordinate and direction vectors for the target.
%             inputs: Struct of the function inputs.
%             err: Struct of final stimulation location and angle errors.
%
% Computation takes typically 2-10s. The optimization algorithm used is stochastic:
% set a seed to produce deterministic output.
%
myStream = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(myStream);

%% Initialize settings

defaultStimMetric = 'WCOG';
defaultRestrictEF = [];
defaultSaveDir = [];
defaultObjective = 'minEnergy';
defaultDistConstr = 0.001;
defaultAngleConstr = 5;

p = inputParser;
addRequired(p,'target_shift',@(x) isnumeric(x) && length(x)==3)
addRequired(p,'pos',@(x) isnumeric(x))
addRequired(p,'E_set',@(x) isnumeric(x))
addParameter(p,'StimMetric',defaultStimMetric, @(x) ischar(x))
addParameter(p,'RestrictEF',defaultRestrictEF, @(x) isnumeric(x))
addParameter(p,'SaveDir',defaultSaveDir,@(x) ischar(x) || isstring(x))
addParameter(p,'Objective',defaultObjective,@(x) ischar(x))
addParameter(p,'DistConstr',defaultDistConstr, @(x) isnumeric(x))
addParameter(p,'AngleConstr',defaultAngleConstr, @(x) isnumeric(x))

parse(p,target_shift,vertices,E_set,varargin{:})
stimMetric = p.Results.StimMetric;
restrictEF = p.Results.RestrictEF;
savedir = p.Results.SaveDir;
objectiveType = p.Results.Objective;
distConstr = p.Results.DistConstr;
angleConstr = p.Results.AngleConstr;

%% Define target

r = max(sqrt(sum(vertices.^2,2)));
meshApex = [0,0,r];

% Rotation matrices
Rmatx = @(a) [1 0 0;0 cos(a) -sin(a);0 sin(a) cos(a)];
Rmaty = @(a) [cos(a) 0 sin(a);0 1 0;-sin(a) 0 cos(a)];
Rmatz = @(a) [cos(a) -sin(a) 0; sin(a) cos(a) 0; 0 0 1];
% Orient EF shift with the z-axis, directed towards positive y
phi = target_shift(2)/1000/r;
pha = -target_shift(1)/1000/r;
pho = deg2rad(target_shift(3));
% Shift EF using the inputs
target.p = meshApex*Rmatx(phi)*Rmaty(pha);
% Find closest mesh index
[~,target.Ind] = min(sqrt(sum((vertices-target.p).^2,2)));
% Define target direction
centerDir = [0,1,0];
target.Dir = centerDir*Rmatz(pho)*Rmatx(phi)*Rmaty(pha);

% The target.p may not lie exactly on a mesh vertex, where E-field values are specified.
% If using stimMetric 'Max', the target is shifted to the nearest vertex. 
if strcmp(stimMetric,'Max')
    target.p = vertices(target.Ind,:);
end

%% Optimize weights

Nc = size(E_set,1);
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
Emag = sqrt(sum(E.^2,2));
[Emag_max,Emag_maxi] = max(Emag);
Emagn = Emag/Emag_max;
[loc,loc_i] = stimulationLoc(Emagn,stimMetric);
Edir = E(loc_i,:);
err.location = 1000*sqrt(sum((target.p-loc).^2,2));
err.angle = vectorAngle(target.Dir,Edir);

fprintf('Constraints: Loc: %.2f mm, Angle: %.2f deg, Max: %.2f mm.   Result: Loc: %.2f mm, Angle: %.2f deg, Max: %.2f.\n',distConstr*1000,angleConstr,5,err.location,err.angle,1000*sqrt(sum((loc-vertices(Emag_maxi,:)).^2,2)))

% Normalize weights to produce Emag of 1
x = x'/max(Emag);
results.weights = x;
results.target = target;
results.inputs = p.Results;
results.err = err;

if ~isempty(savedir)
    if not(isfolder(savedir))
        mkdir(savedir);
    end
    filename = fullfile(savedir,['optimize_Efield_spherical_results_' datestr(datetime('now'),'yymmddHHMMSS') '.mat']);
    results.filename = filename;
    save(filename,'results')
    disp('Optimization results saved.')
end

%% Functions
    function ThetaInDegrees = vectorAngle(u,v)
        CosTheta = max(min(dot(u,v)/(norm(u)*norm(v)),1),-1);
        ThetaInDegrees = real(acosd(CosTheta));
    end

    function [loc,loc_i] = stimulationLoc(Emagn,stimMetric)
        switch stimMetric
            case 'Max'
                [~,loc_i] = max(Emagn);
                loc = vertices(loc_i,:);
            case 'WCOG'
                weightedEF = Emagn.^10;
                loc = sum(vertices.*weightedEF,1)/sum(weightedEF);
                % Scale to sphere edge
                loc = loc*(r/sqrt(sum(loc.^2,2)));
                % Select nearest mesh vertex
                [~,loc_i] = min(sqrt(sum((vertices-loc).^2,2)));
        end
    end

    function obj = objectiveFcn(x)
        E_f = squeeze(sum(E_set.*x',1));
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
                realized_x = x/E_max;
                obj = sum(realized_x.^2)+penalty;
            otherwise
                warning('Unkown objective type.')
                return
        end
    end

    function [c,ceq] = locNdirConsFcn(x)
        % Calculate Efield constraints
        E_f = squeeze(sum(E_set.*x',1));
        Emag_f = sqrt(sum(E_f.^2,2));
        [maxEmag_f,imaxEmag_f] = max(Emag_f);
        Emagn_f = Emag_f/maxEmag_f;
        [loc_f,loc_i_f] = stimulationLoc(Emagn_f,stimMetric);
        d2target = sqrt(sum((target.p-loc_f).^2,2));
        Edir_f = E_f(loc_i_f,:);
        Diff_dir = vectorAngle(Edir_f,target.Dir);

        % Define constraints
        ceq = [];
        c(1) = (d2target - distConstr)*1000;    % Location error < distConstr
        c(2) = Diff_dir - angleConstr;          % Angle error < angleConstraint
        c(3) = sqrt(sum((target.p-vertices(imaxEmag_f,:)).^2,2))-0.005; % Distance between stimulation location and E-field maximum < 5 mm.
    end
end