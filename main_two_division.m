clc
clear

layer ='fc1000';
net1 = densenet201;
net2 = resnet101;

%%
imds = imageDatastore('...\',...
    'IncludeSubfolders',true,...
    'LabelSource','FolderNames');
imds.ReadFcn = @(loc)imresize(imread(loc),[448 448]);
[imdsTrain,idmsTest] = splitEachLabel(imds,0.9,'randomized');

uzunluk=numel(imds.Labels);

for i=1:uzunluk
    
    imm=readimage(imds,i);
    % img=onislem (imm);  // In some datasets, the cropping process is used in the pre-processing stage.
    
    img11=imresize(img,[448 448]); 
    D11=img11(1:224,1:224,:);
    D22=img11(1:224,225:448,:);
    D33=img11(225:448,1:224,:);
    D44=img11(225:448,225:448,:);

   img111=imresize(img,[224 224]); 

   dense_Feats1(:,i) = activations(net1,D11,layer);
   dense_Feats2(:,i) = activations(net1,D22,layer);
   dense_Feats3(:,i) = activations(net1,D33,layer);
   dense_Feats4(:,i) = activations(net1,D44,layer);
   
   resnet101_Feats1(:,i) = activations(net2,D11,layer);
   resnet101_Feats2(:,i) = activations(net2,D22,layer);
   resnet101_Feats3(:,i) = activations(net2,D33,layer);
   resnet101_Feats4(:,i) = activations(net2,D44,layer);

end

labels=imds.Labels;
n=30; // selected features
selected_dense_Feats1=pca_method(dense_Feats1,n);
selected_dense_Feats2=pca_method(dense_Feats2,n);
selected_dense_Feats3=pca_method(dense_Feats3,n);
selected_dense_Feats4=pca_method(dense_Feats4,n);
selected_resnet101_Feats1=pca_method(resnet101_Feats1,n);
selected_resnet101_Feats2=pca_method(resnet101_Feats2,n);
selected_resnet101_Feats3=pca_method(resnet101_Feats3,n);
selected_resnet101_Feats4=pca_method(resnet101_Feats4,n);

% In addition, Link to be used for the PCA method: https://github.com/UMD-ISL/Matlab-Toolbox-for-Dimensionality-Reduction

feat=[selected_dense_Feats1;selected_dense_Feats2;selected_dense_Feats3;selected_dense_Feats4;selected_resnet101_Feats1;selected_resnet101_Feats2;selected_resnet101_Feats3;selected_resnet101_Feats4];

t = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
Md1 = fitcecoc(feat,double(labels),'Learners',t,'Coding', 'onevsall');
CVMd1=crossval(Md1);
accuracy=1-kfoldLoss(CVMd1);

fprintf('Result : %f \n',accuracy);
