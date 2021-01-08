function filter_pupils(folder_path)
tic
cd(folder_path)
load('pupil_long_axis.mat')
load('pupil_limits.mat')


plmod=pupil_long_axis;
ind=find(plmod>pupil_limits(2));
ind=[ind find(plmod<pupil_limits(1))];
ind=sort(ind);

% while length(ind)>1
% %     length(ind)
%     temp=diff(ind);
%     temp2=min(find(temp>1));
%     ind(1:temp2);
%     if isempty(temp2)
%         temp2=2;
%     end
%     if ind(temp2)==length(plmod)
%         plmod(ind(1):ind(temp2))=nan;
%     elseif ind(1)==1
%         plmod(ind(1):ind(temp2))=linspace(plmod(ind(1)),plmod(ind(temp2)+1),temp2);
%     else
%         plmod(ind(1):ind(temp2))=linspace(plmod(ind(1)-1),plmod(ind(temp2)+1),temp2);
%     end
%     ind=find(plmod>pupil_limits(2));
%     ind=[ind find(plmod<pupil_limits(1))];
%     ind=sort(ind);
% end
plmod(ind)=nan; % make values above and below limits nan

pupil_filt=smooth(plmod,15/length(plmod),'rloess'); % filter with 15 frame window ( .5 seconds as 30fps is standard. Should this be a variable instead?)
ind=find(pupil_filt>pupil_limits(2));
ind=[ind; find(pupil_filt<pupil_limits(1))];
ind=sort(ind);
pupil_filt(ind)=nan;

save('pupil_filt','pupil_filt')
disp(['Completed in ' num2str(toc) ' seconds'])
end