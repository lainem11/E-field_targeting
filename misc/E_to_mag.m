function [E_mag,E_max,E_max_ind] = E_to_mag(E)
E_mag = sqrt(sum(E.^2,2));
[E_max,E_max_ind] = max(E_mag);
end