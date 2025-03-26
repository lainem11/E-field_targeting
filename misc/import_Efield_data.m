function EfieldData = import_Efield_data(filename, dataLines)
%IMPORTFILE1 Import data from a text file
%  REP1 = IMPORTFILE1(FILENAME) reads data from text file FILENAME for
%  the default selection.  Returns the data as a string array.
%
%  REP1 = IMPORTFILE1(FILE, DATALINES) reads data for the specified row
%  interval(s) of text file FILENAME. Specify DATALINES as a positive
%  scalar integer or a N-by-2 array of positive scalar integers for
%  dis-contiguous row intervals.
%
%  Author: Ana Soto

%% Input handling

% If dataLines is not specified, define defaults
if nargin < 2
    dataLines = [1, Inf];
end

%% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 13);

% Specify range and delimiter
opts.DataLines = dataLines;
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["MarkerID", "T_rot", "CoilCenter", "CoilPositionInWorldCoordinates", "InVesaliusCoordinates", "Enorm", "IDCellMax", "EfieldVectors", "EnormCellIndexes", "FocalFactors", "EfieldThreshold", "EfieldROISize", "Mtms_coord"];
opts.VariableTypes = ["string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";

% Specify variable properties
opts = setvaropts(opts, ["MarkerID", "T_rot", "CoilCenter", "CoilPositionInWorldCoordinates", "InVesaliusCoordinates", "Enorm", "IDCellMax", "EfieldVectors", "EnormCellIndexes", "FocalFactors", "EfieldThreshold", "EfieldROISize", "Mtms_coord"], "WhitespaceRule", "preserve");
opts = setvaropts(opts, ["MarkerID", "T_rot", "CoilCenter", "CoilPositionInWorldCoordinates", "InVesaliusCoordinates", "Enorm", "IDCellMax", "EfieldVectors", "EnormCellIndexes", "FocalFactors", "EfieldThreshold", "EfieldROISize", "Mtms_coord"], "EmptyFieldRule", "auto");

% Import the data
EfieldData = readmatrix(filename, opts);

end