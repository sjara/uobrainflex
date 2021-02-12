%% session selection
main_folder='\\mammatus2.uoregon.edu\home\brainflex\Behavior_Suite';
cd(main_folder)
mice = dir();
mice(1:2)=[];
mice=mice(find([mice.isdir]));

n=0;
for i = 1:length(mice)
    cd([mice(i).folder '\' mice(i).name])
    mouse_sessions = dir ([mice(i).name '*']);
    for j = 1:length(mouse_sessions)
        n=n+1;
        all_session_folders{n}=[mouse_sessions(j).folder '\' mouse_sessions(j).name];
        mouse_session_dates{n} = mouse_sessions(j).date(1:11);
    end
end

upload_dates = datestr(sort(datenum(unique(mouse_session_dates)),'descend'));
user_selection = inputdlg(['Which upload date do you want to analyze?' sprintf('\n') ...
    '1: ' upload_dates(1,:) sprintf('\n') ...
    '2: ' upload_dates(2,:) sprintf('\n')...
    '3: ' upload_dates(3,:) sprintf('\n')...
    '4: Other']);

if str2num(user_selection{1})<4
    folder_date = upload_dates(str2num(user_selection{1}),:);
    folders_for_analysis = all_session_folders(find(contains(mouse_session_dates, folder_date)));
else
    cd(main_folder)
    sessions_file=uigetfile('*.txt', 'pick a file to indicate sessions to use!');
    this_file=fopen(sessions_file);
    folders_for_analysis=textscan(this_file,'%s')
    folders_for_analysis=folders_for_analysis{1};
end

clearvars -except folders_for_analysis folder_date main_folder sessions_file user_selection
%% set ROI, initial pupil, and filter settings
for i=1:length(folders_for_analysis)
    avi_folder=[folders_for_analysis{i} '\AVI_Files'];
    cd(avi_folder)
    if isempty(dir('first_pupil.mat'))
        DH_pupil_ROIs(avi_folder);
    else
        overwrite = inputdlg([folders_for_analysis{i} sprintf('\n') ...
            'This session already has pupil ROIs set. Overwrite? y/n'])
        if overwrite{1} == 'y'
            DH_pupil_ROIs(avi_folder);
        end
    end
    close all
end
%% pupil analysis
for i=1:length(folders_for_analysis)
    avi_folder=[folders_for_analysis{i} '\AVI_Files'];
    cd(avi_folder)
    if isempty(dir('pupil_long_axis.mat'))
        analyze_ind{i,1}='y'
    else
        analyze_ind{i} = inputdlg([folders_for_analysis{i} sprintf('\n') ...
            'This session already has pupil analysis data. Overwrite? y/n'])
    end
end

for i=1:length(folders_for_analysis)
    avi_folder=[folders_for_analysis{i} '\AVI_Files'];
    if ismember( analyze_ind{i}, 'y')
        disp(' ')
        disp(['Session ' num2str(i) ' of ' num2str(length(folders_for_analysis))])
        disp(['Analyzing video in ' avi_folder])
        pupil_video_analysis(avi_folder)
    else
        disp(' ')
        disp(['Session ' num2str(i) ' of ' num2str(length(folders_for_analysis))])
        disp(['Session already analyzed. Not analyzing this folder: ' avi_folder])
    end
end
%% set artifact cutoff limits 
for i=1:length(folders_for_analysis)
    avi_folder=[folders_for_analysis{i} '\AVI_Files'];
    cd(avi_folder)
    if isempty(dir('pupil_limits.mat'))
        set_pupil_limits(avi_folder);
    else
        overwrite = inputdlg([folders_for_analysis{i} sprintf('\n') ...
            'This session already has pupil limits set. Overwrite? y/n'])
        if overwrite{1} == 'y'
            set_pupil_limits(avi_folder);
        end
    end
    close all
end
%% filter pupil traces & make overview plots

for i=1:length(folders_for_analysis)
    avi_folder=[folders_for_analysis{i} '\AVI_Files'];
    cd(avi_folder)
    if isempty(dir('pupil_filt.mat'))
        analyze_ind{i,1}='y'
    else
        analyze_ind{i} = char(inputdlg([folders_for_analysis{i} sprintf('\n') ...
            'This session already has filtered pupil data. Overwrite? y/n']))
    end
end

for i=1:length(folders_for_analysis)
    avi_folder=[folders_for_analysis{i} '\AVI_Files'];
    if analyze_ind{i} == 'y'
        disp(' ')
        disp(['Session ' num2str(i) ' of ' num2str(length(folders_for_analysis))])
        disp(['Filtering pupils in ' avi_folder])
        filter_pupils(avi_folder)
        plot_pupil_data_analysis(avi_folder)
    else
        disp(' ')
        disp(['Session ' num2str(i) ' of ' num2str(length(folders_for_analysis))])
        disp(['Session already analyzed. Not analyzing this folder: ' avi_folder])
    end
end

%% confirm frame clock and frame lengths match, ok for upload

good_pupil_sessions={};
bad_pupil_sessions={};
for i=1:length(folders_for_analysis)
    avi_folder=[folders_for_analysis{i} '\AVI_Files'];
    session_folder=folders_for_analysis{i};
    cd(session_folder)
    FC=dir('Time_LV_FrameCLock.txt')
    if isempty(FC)
        FC=dir('LV_FrameCLock.txt')
    end
    this_frame_clock=load(FC(1).name)
    cd(avi_folder)
    load('pupil_filt.mat')
    open('pupil_analysis_overview.fig')
    disp('inspect figure for pupil fit parameters, then press any button')
    pause
    happy=inputdlg('Happy? y/n')
    close all
    if happy{1}=='y' & length(pupil_filt)==length(this_frame_clock)
        good_pupil_sessions{length(good_pupil_sessions)+1}=session_folder;
        cd(session_folder)
        writematrix(this_frame_clock,'Pupil_Time.txt')
        writematrix(pupil_filt/max(pupil_filt),'Pupil_Size.txt')
    else % if pupil trace does not look good, or frame clock and frame count are off
         % then write nans for pupil data 
        bad_pupil_sessions{length(bad_pupil_sessions)+1}=session_folder;
        cd(session_folder)
        writematrix(this_frame_clock,'Pupil_Time.txt')
        writematrix(NaN(length(this_frame_clock),1),'Pupil_Size.txt')
    end
end

if str2num(user_selection{1})<4
    cd(main_folder)
    writecell(good_pupil_sessions',[ folder_date '_good_pupil_sessions.txt'])
    writecell(bad_pupil_sessions',[folder_date '_bad_pupil_sessions.txt'])
else
    cd(main_folder)
    writecell(good_pupil_sessions',[sessions_file(1:end-4) '_renalaysis_on_' datestr(today) '_good_pupil_sessions.txt'])
    writecell(bad_pupil_sessions',[sessions_file(1:end-4) '_renalaysis_on_' datestr(today) '_bad_pupil_sessions.txt'])
end
    
