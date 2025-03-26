function all_data = import_invesalius_markers(filename, dataLines)
%IMPORTFILE1 Import data from a text file
% Author: Ana Soto

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 30);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = "\t";

% Specify column names and types
opts.VariableNames = ["x", "y", "z", "alpha", "beta", "gamma", "r", "g", "b", "size", "label", "x_seed", "y_seed", "z_seed", "is_target", "session_id", "is_brain_target", "is_efield_target", "x_cortex", "y_cortex", "z_cortex", "alpha_cortex", "beta_cortex", "gamma_cortex", "x_world", "y_world", "z_world", "alpha_world", "beta_world", "gamma_world"];
opts.VariableTypes = ["string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, ["x", "y", "z", "alpha", "beta", "gamma", "r", "g", "b", "size", "label", "x_seed", "y_seed", "z_seed", "is_target", "session_id", "is_brain_target", "is_efield_target", "x_cortex", "y_cortex", "z_cortex", "alpha_cortex", "beta_cortex", "gamma_cortex", "x_world", "y_world", "z_world", "alpha_world", "beta_world", "gamma_world"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["x", "y", "z", "alpha", "beta", "gamma", "r", "g", "b", "size", "label", "x_seed", "y_seed", "z_seed", "is_target", "session_id", "is_brain_target", "is_efield_target", "x_cortex", "y_cortex", "z_cortex", "alpha_cortex", "beta_cortex", "gamma_cortex", "x_world", "y_world", "z_world", "alpha_world", "beta_world", "gamma_world"], "EmptyFieldRule", "auto");

% Import the data
all_data = readmatrix(filename, opts);

end