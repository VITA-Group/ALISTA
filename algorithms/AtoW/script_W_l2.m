clear;
%% initialization: generating A
% rng(0); % random seed
m = 250;
n = 500;
A = randn(m,n); % good A
%A = ones(m,n) + randn(m,n) * 0.1; % bad A
for i = 1:n
    A(:,i) = A(:,i)/norm(A(:,i));
end

%% calculate W
B = (A * A')^(-1);
W = B * A;

%% display W
visualizationW(A,W);