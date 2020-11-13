 function im1=onislem (I)
 im=rgb2gray(I);
%figure,imshow(im)
g=im2double(im);
g=g-min(min(g));
g=round(255*g/max(max(g)));
g=uint8(g);
%figure,imshow(g)
gson1=(g<125);
%figure,imshow(gson1)
gsonx=imfill(gson1,'holes');
%figure,imshow(gsonx)
gson=imfill(gson1,'holes');
%figure,imshow(gson)
SE =strel('disk',3) ;
   for i=1:3
      gson = imerode(gson,SE);  
  end
[rr,c]=find(gson);
 im1=bwselect(gsonx,c,rr);
  imshow(im1)
