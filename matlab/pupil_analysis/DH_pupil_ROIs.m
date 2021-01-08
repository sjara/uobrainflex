function DH_pupil_ROIs(avi_folder)

cd(avi_folder)

files=dir('*.avi');
files(find([files.bytes]==0))=[];

% set up videoreader, image first frame, and get input for eye roi
videoFReader = vision.VideoFileReader(files(1).name);
this_frame=videoFReader();
image(this_frame);
[x, y]=ginput(2);

xroi=round(min(y):max(y));
yroi=round(min(x):max(x));

% image eye ROI & get input for first pupil mask
this_frame=this_frame(xroi,yroi,:);
image(this_frame)
title('click and drag to draw circled WITHIN the pupil')
h=drawcircle;
first_pupil=createMask(h);
emptyframe=zeros(size(this_frame));


%% display range of binning/filtering options for selection
n=0
this_frame = this_frame(:,:,1)
figure,
filt_ind=0.01:0.01:0.2
for i=filt_ind
n=n+1
this_frame_bin = uint8(imbinarize(this_frame, i)); %was 0.1
%         this_frame_filt =  ordfilt2(this_frame_bin,1,ones(5,5)); %original
this_frame_filt =  ordfilt2(this_frame_bin,1,ones(5,5));
this_frame_filt(this_frame_filt == 0) = 255;
this_frame_filt(this_frame_filt == 1) = 0;
subplot(5,4,n), image(this_frame_filt)
xlabel(num2str(n))
set(gca,'xtick',[],'ytick',[])
end

select_ind=inputdlg('Indicate best filtered image')
            
filt_ind=filt_ind(str2num(select_ind{1}))
%%

save('filt_ind','filt_ind')
save('emptyframe','emptyframe')
save('first_pupil','first_pupil')
save('xroi','xroi')
save('yroi','yroi')

end