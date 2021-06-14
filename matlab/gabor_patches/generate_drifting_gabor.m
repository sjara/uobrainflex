xdim=1024                   % total horizontal pixels
ydim=768                    % total vertical pixels
fSize=150                   % size of gaussian filter
fX=512                      % horizontal center of filter
fY=384                      % vertical center of filter
GratingPixels=127           % pixels per cycle of grating. determine based on screen pixel density and distance from eye
contrast=1                  % weight of blend between gray&% grating
angle=0                     % rotation degrees

refresh_rate = 30 %hz
drift_frames = refresh_rate*2 %2 seconds of frames

drift_cycles = 3
drift_distance = GratingPixels*drift_cycles

cd '\\mammatus2.uoregon.edu\home\brainflex\gabor_patches'
savefolder=['\\mammatus2.uoregon.edu\home\brainflex\gabor_patches\drifting'];
for angle = 0:9:90
    mkdir([savefolder '\' num2str(angle)])
    cd([savefolder '\' num2str(angle)])
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
grating=zeros(length(frame)*5,length(frame)*5)*0.5;


% rotate, and resize grating

grating=imrotate(grating+sin((1:xdim*5)*(pi/(GratingPixels/2))),angle);
sgrating=size(grating);
finalgrating=grating(round(sgrating(1)/2-(-ydim/2:ydim/2-1)),round(sgrating(2)/2-(-xdim/2:xdim/2-1)));

x_extent = round(sgrating(1)/2-(-ydim/2:ydim/2-1));
y_extent = round(sgrating(2)/2-(-xdim/2:xdim/2-1));

n=0;
for offset = linspace(0,drift_distance,drift_frames)
n=n+1;
xoff=round(offset*cosd(angle));
yoff=round(offset*sind(angle));
%offsetgrating = grating(xoff+(round(sgrating(1)/2-(-ydim/2:ydim/2-1))),yoff+(round(sgrating(2)/2-(-xdim/2:xdim/2-1))));
offsetgrating = grating(x_extent-yoff,y_extent+xoff);

% expand to gabor patch image
for i=1:3
gabor(1:ydim,1:xdim,i)=(offsetgrating/2*contrast).*GaussFilt+.5;
end
gabor(1:100,1:100,:)=angle/90;

imwrite(gabor, [num2str(1000+n) '_gabor_ang' num2str(angle) '_pxcycle' num2str(GratingPixels) '.jpg'])

% clf
% image(gabor2)
% pause(.01)

end
end


