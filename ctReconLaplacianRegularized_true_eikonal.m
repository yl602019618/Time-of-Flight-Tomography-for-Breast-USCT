clear
clc

% Add Functions to Path
addpath(genpath('Functions'));

% Load Ring CT Data

% Show Sound Speed Map with Ring Transducers

% Simulated Travel Times Using the Eikonal Equation
for num = 3201:1:3201

load(['./travel_times_eikonal/travel_times_',num2str(num),'.mat'])
times = times(:); %+ (1e-8)*randn(size(times)); %1e-8


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

%% Solve Using Laplacian Regularization Approach

% Reconstruction Parameters
c_guess = 1540; % Uniform Sound Speed Guess
m = (1/c_guess)*ones(Nzi*Nxi,1); % Running Reconstruction
M = numElements*numElements; % Number of Observations
d = [times; zeros(Nxi*Nzi,1)]; % Observations Vector

% Laplacian Matrix for Regularization
mu_reg = 2e-1;%1e-2 
L = mu_reg*laplacianOperator(Nxi, Nzi);

% Gauss-Newton Method
numGaussNewtonIterations = 4;
for iter_gn = 1:numGaussNewtonIterations
    % Matrix Encoding Projections Along Integration Paths
    C_current = reshape(1./m, [Nzi, Nxi]);
    H = eikProjMat(xi, zi, C_current, ind);
    % Forward Model
    G = @(m) [H*m; L*m];
    GT = @(s) H'*s(1:M) + L'*s(M+1:end);
    % Initial Conditions
    p = zeros(size(m)); % Conjugate Direction Vector
    beta = 0; % Variable for Updating Conjugate Direction
    s = G(m)-d; % Current Residual Vector
    r = GT(s); % Current Gradient Direction
    % Iterative Reconstruction by Conjugate-Gradient Least-Squares
    figure; numIter = 200; % Total Number of Iterations 400
    for iter = 1:numIter
        % CGLS Updates
        p = -r + beta*p;
        Gp = G(p); r_norm_sq_last = r'*r;
        alpha = r_norm_sq_last/(Gp'*Gp);
        m = m + alpha*p;
        s = s + alpha*Gp;
        r = GT(s);
        beta = (r'*r)/r_norm_sq_last;  
        getframe;
    end
    % Show Final Result
    mat = reshape(1./m, [Nzi, Nxi]);
    imagesc(xi, zi, mat);
    xlabel('X Coordinate [m]'); ylabel('Z Coordinate [m]'); 
    axis image; colormap gray; colorbar; caxis([1400, 1560]);
    hold on; plot(x_circ, z_circ, 'w.');
    title('Sound Speed Reconstruction by Conjugate Gradient');
    exportgraphics(gcf,['./laplacian_true/result_',num2str(num),'_',num2str(iter_gn),'_eikonal.png'],'Resolution',600)
end
save(['./laplacian_true/result_',num2str(num),'_eikonal.mat'],'mat');
end