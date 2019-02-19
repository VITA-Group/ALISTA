function visualizationW(A,W)

n = size(A,2);

res = A' * W - eye(n);

figure ('Units', 'pixels', 'Position', [300 300 800 275]) ;

subplot(1,2,1);
histogram(res(~eye(n)),'BinWidth',1e-2);
title('off-diagonal');
subplot(1,2,2);
histogram(res(logical(eye(n))),'BinWidth',1e-5);
title('diagonal');

end