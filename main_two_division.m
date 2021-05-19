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
selected_dense_Feats1=pca_svm(dense_Feats1,labels);
selected_dense_Feats2=pca_svm(dense_Feats2,labels);
selected_dense_Feats3=pca_svm(dense_Feats3,labels);
selected_dense_Feats4=pca_svm(dense_Feats4,labels);
selected_dense_Feats5=pca_svm(dense_Feats5,labels);
selected_dense_Feats6=pca_svm(dense_Feats6,labels);
selected_dense_Feats7=pca_svm(dense_Feats7,labels);
selected_dense_Feats8=pca_svm(dense_Feats8,labels);

feat=[selected_dense_Feats1;selected_dense_Feats2;selected_dense_Feats3;selected_dense_Feats4;selected_dense_Feats5;selected_dense_Feats6;selected_dense_Feats7;selected_dense_Feats8];

accuracy=pca_svm(feat,labels);
fprintf('Result : %f \n',accuracy);


    
