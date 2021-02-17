function [gabor] = generategabor(xdim,ydim,fSize,fX,fY,GratingPixels,contrast,angle)
% xdim = image width; ydim = image height
% fSize = size of gaussian filter; fX = filt x center; fY = filt y center
% GratingPixels=pixels per 1/2 cycle of grating (bar width)
% contrast = blend of grating to gray. 1 = full grating, 0= no grating
% angle = angle of ccw rotation from horizontal of grating

% generate image frame
frame=ones(ydim,xdim);

% generate gaussian filter at specified location and size in frame
Xgauss = gaussmf(1:ydim, [fSize fY]);
Ygauss = gaussmf(1:xdim, [fSize fX]);
GaussFilt=frame.*Xgauss'.*Ygauss;

% generate gray background and horizontal grating
gray=ones(size(frame))*0.5;
grating=ones(length(frame)*5,length(frame)*5)*0.5;

% rotate, and resize grating
grating=imrotate(grating+sin((1:xdim*5)*(pi/(GratingPixels))),angle);
sgrating=size(grating);
finalgrating=grating(round(sgrating(1)/2-(-ydim/2:ydim/2-1)),round(sgrating(2)/2-(-xdim/2:xdim/2-1)));

% expand to gabor patch image
for i=1:3
gabor(1:ydim,1:xdim,i)=gray+finalgrating*(.5*contrast).*GaussFilt;
end



