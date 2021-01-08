function pupil_video_analysis(folder_path)

cd(folder_path)
% folder_content=dir('processed')
% if isempty(folder_content)
%     mkdir('processed')
% end

files=dir('*.avi');
    
tic

% sort video files in folder & load parameters set prior
clear ind
files(find([files.bytes]==0))=[];
for i=1:length(files)
    ind(i)=str2num(files(i).name(max(find(files(i).name=='_'))+1:end-4));
end
[temp,ind]=sort(ind);

load('first_pupil');
load('xroi');
load('yroi');
load('emptyframe');
load('filt_ind');

[mx my]=find(first_pupil);

fig_position=[0 200 length(yroi) length(xroi)];

% set variables for expansion of pupil mask
xjump=1*size(xroi,2);
yjump=1;
mx=nan;

%preallocate variables
a=zeros(1,length(files)*10000); %area of found convex hull
a2=zeros(1,length(files)*10000); % long axis of fitted ellipse
a3=zeros(1,length(files)*10000); % short axis of fitted ellipse
a4=zeros(1,length(files)*10000); %largest distance between found points
mask=emptyframe;
mask(find(first_pupil))=1;
first_mask=mask;
maxd=20;


i=0;
for ii=1:length(files)
% % % % % % % %     this_video_filt = VideoWriter(char(strcat('processed\', files(ind(ii)).name(1:end-4), '_processed_', datestr(date,29), '.avi')), 'Motion JPEG AVI');
% % % % % % % %     this_video_filt.FrameRate = 30;
% % % % % % % %     open(this_video_filt)
    
%     tic
    disp(['Video file ' num2str(ii) ' of ' num2str(length(files))])
    videoFReader = vision.VideoFileReader(files(ind(ii)).name);
    %step through video frames and analyze each
    while ~isDone(videoFReader)
%         tic
        i=i+1;
        temp=videoFReader();
        % crop and filter frame
        this_frame=temp(xroi,yroi,1);
        this_frame_bin = uint8(imbinarize(this_frame, filt_ind)); %was 0.1
        this_frame_filt =  ordfilt2(this_frame_bin,1,ones(5,5));
        this_frame_filt(this_frame_filt == 0) = 255;
        this_frame_filt(this_frame_filt == 1) = 0;
        
        
        this_frame_analyze=this_frame_filt;
        this_frame_analyze([1:2 end-1:end],:)=0;
        this_frame_analyze(:,[1:2 end-1:end])=0;
        %                 image(this_frame_analyze)
        
        if isnan(mx)
            [y,x]=find(first_mask); %test this
            mx=round(mean(x));
            my=round(mean(y));
        end

        
        maskind=find(mask);
        jump_mult = round(maxd/10); % set pixels to jump on prior pupil size
        % create mask based on prior pupil found, and shrink it from the edges
        for zzz=1:2
            maskind=intersect(maskind,maskind+yjump*jump_mult);
            maskind=intersect(maskind,maskind-yjump*jump_mult);
            maskind=intersect(maskind,maskind-xjump*jump_mult);
            maskind=intersect(maskind,maskind+xjump*jump_mult);
            maskind(~ismember(maskind,find(mask)))=[];
        end
        mask=emptyframe;
        mask(maskind)=1;
        
        % if prior pupil is not found, use first pupil as default start
        if isempty(maskind)
            [y,x]=find(first_mask); %test this
%             mx=round(mean(x));
%             my=round(mean(y));
            mask(y,x)=1;
            for z=1:10
                mask(find(mask)+yjump)=1;
                mask(find(mask)-yjump)=1;
                mask(find(mask)-xjump)=1;
                mask(find(mask)+xjump)=1;
            end
        end
        
        
        % expand mask in x and y, and test if more pupil pixels are found.
        % expand mask untill no new pixels are found
        z=0; z2=1;
        while z<z2
            z=z2;
            mask(find(mask)+yjump)=1;
            mask(find(mask)-yjump)=1;% Huge bug potential
            mask(find(mask)-xjump)=1;
            mask(find(mask)+xjump)=1;
            maskpoints=find(mask);
            pupil=find(this_frame_analyze(maskpoints));
            z2=length(pupil);
            mask=emptyframe;
            mask(maskpoints(pupil))=1;
        end
        %                 image(mask)
        %                 image(this_frame_analyze)
        
        %%%%%%%% commented out all plotting
% % % % % % % %         if ismember(i,1:200:200000)
% % % % % % % %             close(gcf)
% % % % % % % %             figure(1), set(gcf,'Position',fig_position)
% % % % % % % %         end
% % % % % % % %         figure(1),clf,hold on
% % % % % % % %         subplot('position',[0 0 1 1]),image(temp(xroi,yroi,:)), daspect([1 1 1]), hold on
        
%         tic
        % analyze pupil size based on found pupil pixels
        [y,x]=find(mask);
        if ~isempty(x)
            maxd=0;
            nn=0;
%             tic
            % calculate longest dustance between two pupil pixels
            for z=1:length(x)
                for zz=1:length(x)-z
                    nn=nn+1;
                    d=sqrt( (x(z)-x(z+zz))^2+(y(z)-y(z+zz))^2 );
                    if d>maxd
                        maxd=d;
                        coord=[z z+zz];
                    end
                end
            end
            a4(i)=maxd;
%             toc          
            if length(x)>1
% % % % % % % %                 plot(x(coord),y(coord),'b','linewidth',3)
% % % % % % % %                 plot(x,y,'g')
% % % % % % % %                 this_ellipse=fit_ellipse(x,y,gca);
            else
            end
            
            % find area of convex hull of found pupil points 
            % And fit ellipse to points of convex hull
            if length(unique(x))>1 & length(unique(y))>1
                [k,a(i)]=convhull(x,y);
                x=x(k);y=y(k);
                this_ellipse=fit_ellipse(x,y);
                if ~isempty(this_ellipse.long_axis)
                a2(i)=this_ellipse.long_axis;
                a3(i)=this_ellipse.short_axis;
                end
            else
                x=mx;y=my;
                a(i)=0;
                a2(i)=0;
                a3(i)=0;
                a4(i)=0;
            end
%             toc
            mx=round(mean(x));
            my=round(mean(y));
        end
% % % % % % % % %         writeVideo(this_video_filt,getframe(gcf));
%         toc
    if ismember(i,1:1000:length(a))
        disp(['Frame ' num2str(i) ' of ~' num2str(length(a))])
    end
    end
%     disp(['completed in ' num2str(toc) ' seconds'])
% % % % % % % %     close(this_video_filt)
end

% pupil_area=a(1:i);
pupil_long_axis=a2(1:i);
pupil_short_axis=a3(1:i);
pupil_largest_dist=a4(1:i);

%         cd([sessions(jjjj).folder '\' sessions(jjjj).name])
% save('pupil_area','pupil_area')
save('pupil_long_axis','pupil_long_axis')
save('pupil_short_axis','pupil_short_axis')
save('pupil_largest_dist','pupil_largest_dist')
toc
end
