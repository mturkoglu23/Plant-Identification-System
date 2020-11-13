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
    img=onislem (imm); 
    
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

feat=[dense_Feats1;dense_Feats2;dense_Feats3;dense_Feats4;resnet101_Feats1;resnet101_Feats2;resnet101_Feats3;resnet101_Feats4];
labels=imds.Labels;

accuracy=pca_svm(feat,labels);
fprintf('Result : %f \n',accuracy);


    
