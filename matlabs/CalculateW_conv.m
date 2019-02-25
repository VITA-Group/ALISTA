%% A script to calculate convolutional kernels with minimal coherence:
%  min \|W_conv^T D_conv\|^2 subject to diag(W_conv^T D_conv) = 1
%  This is an implement of Algorithm 2 in Appendix E.2 of(Liu et al, 2019).
%  Just run this script without any arguments.

%  Author: Jialin Liu, UCLA math department (danny19921123@gmail.com)
%  Last Modified: 2019-2-15

%% Note: if the algorithm diverges, then try a smaller step size: eta.

%% Script starts.
clear;

% Load convolutional kernels
load('./D_conv.mat','D');

% Get the dimensions
Ds = size(D,1);
M = size(D,3);
N = 2*Ds - 1; % Due to Theorem 3 in our paper

% Initialization
Df = fft2(D,N,N);
Df = reshape(Df, [N,N,1,M]);
Dft = reshape(Df, [N,N,M,1]);
Dfh = reshape(conj(Df), [N,N,M,1]);
W = reshape(D,[Ds,Ds,1,M]);
Wf = Df;

% Step size in the optimization
eta = 0.002; % step size

% Main iterations
fprintf('Calculation Starts...\n');
for t = 1:500

    % calculate residuals and function values
    res = bsxfun(@times, Dfh, Wf);
    f = norm(res(:))^2;
    if mod(t,50)==0, fprintf('t: %d\t f: %.3f\n',t,f); end

    % calculate gradient
    gra = bsxfun(@times, Dft, res);
    gra = sum(gra, 3);

    % gradient descent in the fourier domain
    Wf = Wf - eta * gra;

    % back to the spacial domain and do projection
    W = ifft2(Wf, 'symmetric');
    W = reshape(W, [N,N,M]);
    W = W(1:Ds,1:Ds,:);
    W = proj(W,D);

    % calculate FFT for the next step
    Wf = fft2(W,N,N);
    Wf = reshape(Wf, [N,N,1,M]);
end

% save to file
save('W_conv.mat','W');
fprintf('Calculation ends. Results are saved in W_conv.mat.\n');

% Visualization
% Please download SPORCO: http://brendt.wohlberg.net/software/SPORCO/
% And copy "util/imdisp.m" and "util/tiledict.m" to the current folder
% Then uncomment the following two lines. You can get the visualization.

% figure;
% imdisp(tiledict(W));


%% Functions
function W_out = proj(W_in,D)
% projection of the dictionary on "diag(W^TD)=1"
M = size(D,3);
N = size(D,1);
A = zeros(N*N,M);
W = zeros(N*N,M);
for ii = 1:M
    A(:,ii) = reshape(D(:,:,ii),[N*N,1]);
    W(:,ii) = reshape(W_in(:,:,ii),[N*N,1]);
end
aw = diag(A'*W);
aw = repmat(aw',[size(A,1), 1]);
W_next = W + (1-aw).*A;
W_out = zeros(N,N,M);
for ii = 1:M
    W_out(:,:,ii) = reshape(W_next(:,ii),[N,N]);
end
end
