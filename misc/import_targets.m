function targets = import_targets(marker_file,labels)

targets = [];
count = 1;
for i = 1:length(labels)
    [all_positions, all_orientations, all_labels] =extract_brain_target_list(marker_file);
    ind = find(strcmp(all_labels, labels{i}));

    if isempty(ind)
        fprintf("Label '%s' not found.\n",labels{i})
    else
        orientation = get_direction_for_brain_targets(all_orientations(ind,:));
        position = all_positions(ind, :);
    
        % Correct position unit and flip y-axis
        pos = position/1000;
        pos(2) = -pos(2);
    
        dir = orientation./norm(orientation);
        targets(count).pos = pos;
        targets(count).dir = dir';
        targets(count).label = labels{i};
        count = count + 1;
    end
end
fprintf("%i targets imported.\n",length(targets))

end

%% Helper function
function [all_positions, all_orientations, all_labels] = extract_brain_target_list(filename)

markers = import_invesalius_markers(filename);

% Find the column index of 'brain_target_list'
idx = find(strcmp(markers(2,:), 'brain_target_list'));
% Extract all rows starting from row 3 in that column
y = markers(3:end, idx);

% Initialize output containers (use cell array in case rows vary in size)
orientations = {};
positions = {};
labels ={};
for j = 1:length(y)
    raw_str = y{j};

    % Skip if the cell is empty or not a char/string
    if isempty(raw_str) || ~(ischar(raw_str) || isstring(raw_str))
        orientations{end+1} = NaN;
        positions{end+1} = NaN;
        labels{end+1} = NaN;
        continue;
    end

    % Fix Python-like syntax for JSON
    raw_str = strrep(raw_str, 'False', 'false');
    raw_str = strrep(raw_str, 'True', 'true');
    raw_str = strrep(raw_str, 'None', 'null');
    raw_str = strrep(raw_str, '''', '"');  % Replace all single quotes with double quotes

    try
        data = jsondecode(raw_str);
        n = numel(data);

        orient = zeros(n, 3);
        pos = zeros(n, 3);
        label = cell(n,1);
        for i = 1:n
            orient(i, :) = data(i).orientation;
            pos(i, :) = data(i).position;
            label{i} = data(i).label;
        end

        orientations{end+1} = orient;
        positions{end+1} = pos;
        labels{end+1} = label;

    catch ME
        warning('Row %d could not be parsed: %s', j, ME.message);
        orientations{end+1} = NaN;
        positions{end+1} = NaN;
        labels{end+1} = NaN;
    end
end

% Initialize lists to hold valid data
all_orientations = [];
all_positions = [];
all_labels = [];
for j = 1:length(orientations)
    orient = orientations{j};
    pos = positions{j};
    label = labels{j};
    % Check if it's a valid numeric matrix
    if isnumeric(orient) && ~any(isnan(orient(:)))
        all_orientations = [all_orientations; orient];  % Concatenate vertically
        all_positions = [all_positions; pos];
        all_labels = [all_labels;label];
    end
end
end

function markers = import_invesalius_markers(filename)
%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 35);

% Specify range and delimiter
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["marker_id", "x", "y", "z", "alpha", "beta", "gamma", "r", "g", "b", "size", "label", "x_seed", "y_seed", "z_seed", "is_target", "is_point_of_interest", "session_id", "x_cortex", "y_cortex", "z_cortex", "alpha_cortex", "beta_cortex", "gamma_cortex", "marker_type", "z_rotation", "z_offset", "mep_value", "brain_target_list", "x_world", "y_world", "z_world", "alpha_world", "beta_world", "gamma_world"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "char", "double", "double", "double", "char", "char", "double", "char", "char", "char", "char", "char", "char", "double", "double", "double", "char", "char", "double", "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["label", "is_target", "is_point_of_interest", "x_cortex", "y_cortex", "z_cortex", "alpha_cortex", "beta_cortex", "gamma_cortex", "mep_value", "brain_target_list"], "WhitespaceRule", "preserve", "EmptyFieldRule", "auto");

% Import the data
markers  = readtable(filename, opts);

%% Convert to output type
markers  = table2cell(markers );
numIdx = cellfun(@(x) ~isnan(str2double(x)), markers );
markers (numIdx) = cellfun(@(x) {str2double(x)}, markers(numIdx));
end

function v_global = get_direction_for_brain_targets(orientation_from_invesalius)
eul = deg2rad(orientation_from_invesalius);
alpha = eul(1);  % roll (X)
beta  = eul(2);  % pitch (Y)
gamma = eul(3);  % yaw (Z)

% Rotation matrices
Rx = [1 0 0;
      0 cos(alpha) -sin(alpha);
      0 sin(alpha) cos(alpha)];
Ry = [cos(beta) 0 sin(beta);
      0 1 0;
     -sin(beta) 0 cos(beta)];
Rz = [cos(gamma) -sin(gamma) 0;
      sin(gamma) cos(gamma) 0;
      0 0 1];

% Compose full rotation matrix in XYZ order
R = Rz * Ry * Rx;  % R = Rz * Ry * Rx (applied right-to-left)

% Local vector (e.g., z-axis)
v_local = [1;0; 0];

% Rotate to global space
v_global = R * v_local;
end