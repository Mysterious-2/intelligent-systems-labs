function metric = apvalumas_roundness(Im)

imgray = rgb2gray(Im);
BW = im2double(imgray);

BW = imbinarize(imgray, 0.95);

BW = imfill(~BW,'holes');

BW = imopen(BW,strel('disk',12));

BWpr = regionprops(double(BW),{'perimeter','area'});
metric = 4*pi*BWpr(1).Area/BWpr(1).Perimeter^2;