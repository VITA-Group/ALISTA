% cirular boundary, frequency domain update
clear;
addpath('/ssd1/xchen/sporco/Util');

N = 30;

%load('./sgd_D12_M64_lam0.05.mat');
load('/home/xhchrn/DeepBregmanISS/conv_dicts/dicts_bsd500/sgd_D8_M64_lam0.05.mat');
% load('./sgd_D16_M100_lam0.05.mat');

Ds = size(D,1);
M = size(D,3);

% myzpad = @(D) padarray(D,[N-Ds,N-Ds],'post');
% 
% 
% Dexd = myzpad(D);

Df = fft2(D,N,N);

% % check strong convexity
% mineig = zeros(1,M);
% for m = 1:M
%     testtt = Df(:,:,m);
%     eigens = conj(testtt).*testtt;
%     mineig(m) = min(eigens(:));
% end
% max(mineig)


Df = reshape(Df, [N,N,1,M]);
Dft = reshape(Df, [N,N,M,1]);
Dfh = reshape(conj(Df), [N,N,M,1]);
% Identy = zeros(N,N,M,M);
% for i = 1:N
%     for j = 1:N
%         Identy(i,j,:,:) = eye(M);
%     end
% end

W = reshape(D,[Ds,Ds,1,M]);
Wf = Df;

eta = 0.002;%5*5
%eta = 0.001;%12*12
%eta = 0.0005;%16*16

for t = 1:700
    
    res = bsxfun(@times, Dfh, Wf);
%     res = res;% - Identy;
    
    f = norm(res(:))^2;
    
    gra = bsxfun(@times, Dft, res);
    gra = sum(gra, 3);
    
    Wf = Wf - eta * gra;
    W = ifft2(Wf, 'symmetric');
    W = reshape(W, [N,N,M]);
    W = bcrop(W,[Ds Ds]);
    
%     test1 = zeros(1,M);
%     for iii = 1:M
%         test1(iii) = sum(sum(W(:,:,iii).*D(:,:,iii)));
%     end
%     test1
    
    W = proj(W,D);
    
%     test2 = zeros(1,M);
%     for iii = 1:M
%         test2(iii) = sum(sum(W(:,:,iii).*D(:,:,iii)));
%     end
%     test2
    
    Wf = fft2(W,N,N);
    Wf = reshape(Wf, [N,N,1,M]);
    
    fprintf('t: %d\t f: %.3f\n',t,f);
end
%figure;
%imdisp(tiledict(W));
% load('Ws_D5_normal.mat','Wref');
% norm(W(:)-Wref(:))^2/norm(Wref(:))^2

save ('bsd500_fft_N30_D8_M64_lam0.05.mat', 'W');

%% functions 
function u = bcrop(v, sz)

  if numel(sz) <= 2
    if numel(sz) == 1
      cs = [sz sz];
    else
      cs = sz;
    end
    u = v(1:cs(1), 1:cs(2), :);
  else
    if size(sz,1) < size(sz,2), sz = sz'; end
    cs = max(sz);
    u = zeros(cs(1), cs(2), size(v,3), class(v));
    for k = 1:size(v,3),
      u(1:sz(k,1), 1:sz(k,2), k) = v(1:sz(k,1), 1:sz(k,2), k);
    end
  end

end

function Wout = proj(Win,D)
M = size(D,3);
N = size(D,1);
A = zeros(N*N,M);
W = zeros(N*N,M);
for ii = 1:M
    A(:,ii) = reshape(D(:,:,ii),[N*N,1]);
    W(:,ii) = reshape(Win(:,:,ii),[N*N,1]);
end
aw = diag(A'*W);
aw = repmat(aw',[size(A,1), 1]);
W_next = W + (1-aw).*A;
Wout = zeros(N,N,M);
for ii = 1:M
    Wout(:,:,ii) = reshape(W_next(:,ii),[N,N]);
end
end
