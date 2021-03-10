function plot_pupil_data_analysis(avi_folder)

cd(avi_folder)

files=dir('*.avi');
load('pupil_filt.mat');
load('xroi');
load('yroi');
load('pupil_long_axis.mat');
load('pupil_limits.mat');
plmod=pupil_long_axis;
plmod(find(plmod>pupil_limits(2)))=nan;
plmod(find(plmod<pupil_limits(1)))=nan;

clear ind
files(find([files.bytes]==0))=[];
for i=1:length(files)
    ind(i)=str2num(files(i).name(max(find(files(i).name=='_'))+1:end-4));
end
[temp,ind]=sort(ind);

videoframes = 10000;

max_frame = find(pupil_filt==max(pupil_filt));
max_frame=max_frame(1);
max_vid_ind = ind(ceil(max_frame(1)/videoframes));
max_frame_ind = max_frame(1)-(ceil(max_frame(1)/videoframes)-1)*videoframes;
this_vid = VideoReader(files(max_vid_ind).name);
max_frame_image=read(this_vid,max_frame_ind);
max_total_frames=this_vid.NumFrames;

min_frame = find(pupil_filt==min(pupil_filt));
min_frame=min_frame(1);
min_vid_ind = ind(ceil(min_frame(1)/videoframes));
min_frame_ind = min_frame(1)-(ceil(min_frame(1)/videoframes)-1)*videoframes;
this_vid = VideoReader(files(min_vid_ind).name);
min_total_frames=this_vid.NumFrames;
min_frame_image=read(this_vid,min_frame_ind);

%

figure,hold on,
subplot('position',[0 .5 .5 .5])
image(min_frame_image(xroi,yroi,:))
set(gca,'xtick',[],'ytick',[])
daspect([1 1 1])

subplot('position',[.5 .5 .5 .5])
image(max_frame_image(xroi,yroi,:))
set(gca,'xtick',[],'ytick',[])
daspect([1 1 1])

subplot('position',[.1 0 .8 .25]), hold on
plot(pupil_long_axis,'r')
plot(plmod,'k')
plot(pupil_filt,'g','linewidth',2)
plot(min_frame,pupil_filt(min_frame),'kv')
plot(max_frame,pupil_filt(max_frame),'k^')
ylim([pupil_filt(min_frame(1))-1 pupil_filt(max_frame(1))+1])
title(avi_folder)


subplot('position',[.1 .3 .25 .2]), hold on
if min_frame<300
    t1=1;
else
    t1=min_frame-300;
end

if min_frame+300>length(plmod)
    t2=length(plmod);
else
    t2=min_frame+300;
end
plot(pupil_filt(t1:t2),'g','linewidth',2)
plot(round((t2-t1)/2),pupil_filt(min_frame),'kv')
set(gca,'xtick',[])

subplot('position',[.7 .3 .25 .2]), hold on
if max_frame<300
    t1=1;
else
    t1=max_frame-300;
end
if max_frame+300>length(plmod)
    t2=length(plmod);
else
    t2=max_frame+300;
end

plot(pupil_filt(t1:t2),'g','linewidth',2)
plot(round((t2-t1)/2),pupil_filt(max_frame),'k^')
set(gca,'xtick',[])

subplot('position',[.4 .3 .2 .2]), hold on
hist(pupil_filt/max(pupil_filt))
xlim([0 1])
set(gca,'xtick',[0:.25:1])

saveas(gcf,'pupil_analysis_overview.fig')
close all

end

