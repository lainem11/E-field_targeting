function [efield_set,mesh] = generate_example_efields(surface_type)

% Create surface
if strcmp(surface_type,'spherical')
    mesh = create_half_sphere();
elseif strcmp(surface_type,'complex')
    mesh = create_complex_surface();
end

% Generate circular field
efield_set(1,:,:) = generate_ring_field(mesh);

% Generate dual ring field
efield_set(2,:,:) = generate_dual_ring_field(mesh,0);

% Generate another dual ring field with 90 degree rotation
efield_set(3,:,:) = generate_dual_ring_field(mesh,pi/2);

% Generate four-ring field
efield_set(4,:,:) = generate_four_ring_field(mesh,0);

% Generate another four-ring field with 45 degree rotation
efield_set(5,:,:) = generate_four_ring_field(mesh,pi/4);

function field = generate_dual_ring_field(cortex,phi)
    field = generate_ring_field(cortex,pi/2,phi)-generate_ring_field(cortex,-pi/2,phi);
    field = field/max(sqrt(sum(field.^2,2)));
end

function field = generate_four_ring_field(cortex,phi)
    field = generate_ring_field(cortex,pi/2,phi)+generate_ring_field(cortex,-pi/2,phi)- generate_ring_field(cortex,pi/2,phi-pi/2)-generate_ring_field(cortex,-pi/2,phi-pi/2);
    field = field/max(sqrt(sum(field.^2,2)));
end

end