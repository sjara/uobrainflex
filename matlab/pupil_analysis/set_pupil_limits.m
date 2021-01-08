function set_pupil_limits(folder_path)

cd(folder_path)
load('pupil_long_axis')
% re run code until criteria is met
happy{1}='n'
while happy{1}~='y' % plot pupil trace, and get input on max & min cutoffs
figure(1),clf, hold on
plot(pupil_long_axis,'r')
title('click twice: once for max, once for min')
pupil_limits=ginput(2)
pupil_limits=sort(pupil_limits(:,2))

plmod=pupil_long_axis;
plmod(find(plmod>pupil_limits(2)))=nan % make values above/below max/min nan
plmod(find(plmod<pupil_limits(1)))=nan
plot(plmod,'k')
title('zoom and pan to observe cutoffs, then press any key')

pause

happy=inputdlg('happy with results? y/n')
end

save('pupil_limits','pupil_limits')

end
