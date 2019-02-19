clear;

%% initialization
%rng(0);
m = 250;
n = 500;
load ('A.mat');
%A = randn(m,n); % good A
%A = ones(m,n) + randn(m,n) * 0.5; % bad A
for i = 1:n
    A(:,i) = A(:,i)/norm(A(:,i));
end

%% iteration: projected gradient descent
W = A;
f = func(A,A);
maxit = 5000;
gamma0 = 1;
c = 1e-3;
tau = 0.5;
Q = ones(n,n)+eye(n)*(-1);
for t = 1: maxit
    res = A' * W - eye(n);
    gra = A * ( Q .* res);
    dir = gra;
    innerprod = sum(sum(dir.*gra));
    gamma = gamma0;
    is_break = 0;
    for attempt = 1:100
        W_next = W - gamma * dir;
        W_next = proj(W_next,A);
        f_next = func(W_next,A);
        if f_next - f < - gamma * c * innerprod
            if abs(f-f_next)/f < 1e-12, is_break=1; end
            W = W_next;
            f = f_next;
            fprintf('t: %d\t, attempts:%d\t, func: %f\n', t, attempt, f);
            break;
        end
        gamma = tau * gamma;
    end
    if is_break == 1, break; end
    if attempt >= 100, break; end
    fprintf('t: %d\t, attempts:%d\t, func: %f\n', t, attempt, f);
end

%% display
visualizationW2(A,W);
%visualizationW(A,A);

%% functions
function f = func(W,A)
n = size(A,2);
res = A' * W - eye(n);
Q = ones(n,n)+eye(n)*(-1);
res = res .* sqrt(Q);
f = sum(sum(res.*res)); 
end

function W_next = proj(W,A)
aw = diag(A'*W);
aw = repmat(aw',[size(A,1), 1]);
W_next = W + (1-aw).*A;
end

