function [points, elements] = convert_custom_bin_to_vtk(filename)
    % Check if the file exists
    if exist(filename, 'file') == 2
       % Read the binary file
        fid = fopen(filename, 'rb');
        numbers = fread(fid, 3, 'int32');
        points = fread(fid, [3, numbers(2)], 'float32')';
        elements = fread(fid, [3, numbers(3)], 'int32')';
        fclose(fid);
    else
        disp('File does not exist');
        polydata = [];
    end
end
