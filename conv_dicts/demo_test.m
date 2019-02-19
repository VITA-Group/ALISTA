clear;
addpath('./dicts_bsd500');
addpath('./dicts_CVPRSR08');
addpath('./Util');

% Load dictionary
load('sgd_D8_M64_lam0.05.mat');

% load training sets.
K = 2;
S0 = cell(1,K);
%foldername = '../BSR_data/test';
foldername = '../CVPRSR/Test';
folder = dir(foldername);
for index = 1:K
    namenow= [foldername,'/',folder(index + 2).name];
    imagenow = imread(namenow);
    if size(imagenow,3)>1
        imagenow = rgb2gray(imagenow);
    end
    imagenow = single(imagenow)/255;
    imagenow = imagenow - mean(imagenow(:));
	S0{index} = imagenow;
end
clear imagenow;

% Compute representation
lambda = 0.001;
opt = [];
opt.Verbose = 1;
opt.MaxMainIter = 1000;
opt.rho = 100*lambda + 1;
opt.RelStopTol = 1e-3;
opt.AuxVarObj = 0;
opt.HighMemSolve = 1;

for index = 1:K
    s = S0{index};
    [X, optinf] = cbpdn(D, s, lambda, opt);

    % Compute reconstruction
    DX = ifft2(sum(bsxfun(@times, fft2(D, size(X,1), size(X,2)), fft2(X)),3), ...
           'symmetric');

    figure;
    subplot(1,3,1);
    imdisp(s);
    title('Original image');
    subplot(1,3,2);
    imdisp(DX);
    title(sprintf('Reconstructed image (SNR: %.2fdB)', psnr(s, DX)));
    subplot(1,3,3);
    imagesc(DX - s);
    axis image; axis off;
    title('Difference');

end