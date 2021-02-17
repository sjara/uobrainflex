xdim=1024                   % total horizontal pixels
ydim=768                    % total vertical pixels
fSize=140                   % size of gaussian filter
fX=512                      % horizontal center of filter
fY=384                      % vertical center of filter
GratingPixels=120           % pixels per cycle of grating. determine based on screen pixel density and distance from eye
contrast=1                  % weight of blend between gray&% grating
angle=0                     % rotation degrees
%% example test gabor iterations
for contrast=0:.1:1
image(generategabor(xdim,ydim,fSize,fX,fY,GratingPixels,contrast,angle))
daspect([1 1 1])
pause(.01)
end
%% generate gabors at defined contratsts and angles
cd '\\mammatus2.uoregon.edu\home\brainflex\gabor_patches'
savefolder=['\\mammatus2.uoregon.edu\home\brainflex\gabor_patches\'];
for contrast=[.2 .4 .6 .8 1] % iterate contrasts
for angle=0:45:135          %iterate angles
    mkdir([savefolder num2str(angle)])  %create folder for each angle
    cd([savefolder num2str(angle)])
%%SCREEN RESOLUTION AND DISTANCE FROM EYE AND PIXELS PER CYCLE WILL DETERMINE CYCLES PER DEGREE OF STIMULUS
for GratingPixels=[120]     %iterate grating size. Best to keep one per run
g=generategabor(xdim,ydim,fSize,fX,fY,GratingPixels,contrast,angle);
if angle==0 %make square indicator in top left with differing intensities
    g(1:100,1:100,:)=1;
elseif angle==45
    g(1:100,1:100,:)=.75;
elseif angle==90
    g(1:100,1:100,:)=.5;
elseif angle==135
    g(1:100,1:100,:)=.25;
end
image(g)
% imwrite(g, ['gabor_ang' num2str(angle) '_con' num2str(contrast*10) '_px' num2str(GratingPixels) '.tif'])
imwrite(g, ['gabor_ang' num2str(angle) '_con' num2str(contrast*10) '_px' num2str(GratingPixels) '.jpg'])
end
end
end

% make gray patches for background
contrast=0
mkdir([savefolder 'gray'])
cd([savefolder 'gray'])
for i=1:5
    g=generategabor(xdim,ydim,fSize,fX,fY,GratingPixels,contrast,angle);
    g(1:100,1:100,:)=0;
    imwrite(g, ['gray_' num2str(i) '.jpg'])
end