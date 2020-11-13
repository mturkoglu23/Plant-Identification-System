function accuracy=pca_svm(feat,label)

W=feat;
sampleMat=W;
n=200; // Reduction values

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
Y1=double(labels);

t = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
Md1 = fitcecoc(X1,Y1,'Learners',t,'Coding', 'onevsall');
CVMd1=crossval(Md1);
accuracy=1-kfoldLoss(CVMd1);

