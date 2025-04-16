function [centroid_pos,centroid_ind] = calculate_WCOG(E_mag_n,vertices)
weighted_E = E_mag_n.^10;
centroid_pos = sum(vertices.*weighted_E,1)/sum(weighted_E);
[~,centroid_ind] = min(sqrt(sum((vertices-centroid_pos).^2,2)));
end