addpath('./dicts_bsd500');
addpath('./dicts_CVPRSR08');
addpath('./Util');

load('sgd_D16_M100_lam0.1.mat');

figure;
imdisp(tiledict(D));
