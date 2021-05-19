function X1=pca_svm(feat,n)

W=feat;
sampleMat=W;
% n // Reduction values

nSamples = size(sampleMat,1);
nDim = size(sampleMat,2);

meanSample = mean(sampleMat,1);

sampleMat = sampleMat - repmat(meanSample, nSamples, 1);

if nDim > nSamples
      sampleMat = sampleMat.';
end

C = sampleMat.' * sampleMat ./ nSamples;

[V,D] = eig(C);
D = diag(D);
D = flipud(D); 
V = fliplr(V); 

if nDim > nSamples
    V = sampleMat * V;
    for i = 1:nSamples
        normV = norm(V(:,i));
        V(:,i) = V(:,i) ./ normV;
    end
end

if exist('n','var') 
    if n > nDim
        error('error')
    end
    D = D(1:n);
    V5 = V(:,1:n);
end

X1=V5;
