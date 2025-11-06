clear
clc

% Add Functions to Path
addpath(genpath('Functions'));

for num = 3201;
disp(num)
C = load(['/gpfs/share/home/2401112587/neuralFWI/ali_toft/speed/breast_',num2str(num),'.mat']).mat;
C = double(C);
C = imresize(C,[801,801]);

dxi = 0.15e-3; xmax = 60e-3;
xi = -xmax:dxi:xmax; zi = xi;
Nxi = numel(xi); Nzi = numel(zi);
[Xi, Zi] = meshgrid(xi, zi);
circle_radius = 55e-3; numElements = 256;
circle_rad_pixels = floor(circle_radius/dxi);
theta = -pi:2*pi/numElements:pi-2*pi/numElements;
x_circ = circle_radius*cos(theta); 
z_circ = circle_radius*sin(theta); 
[x_idx, z_idx, ind] = sampled_circle(Nxi, Nzi, circle_rad_pixels, theta);
msk = zeros(Nzi, Nxi); msk(ind) = 1;



% Build Sparse System Matrix for Straight-Path Forward Projection
times = eikTimes(xi, zi, C, ind);

% Save Results to File
save(['travel_times_eikonal/travel_times_',num2str(num),'.mat']);
end