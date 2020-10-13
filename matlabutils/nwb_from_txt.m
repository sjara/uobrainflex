function nwb = nwb_from_txt(inputDir, outputDir)
% nwb_from_txt  Creates an NWB file given data as text files.
%
%   nwb_from_txt(inputDir, outputDir) will take .txt files from
%   inputDir and produce a single NWB file saved to outputDir.
%
%   The input text files should contain...


  subject = 'testMouse'; % Get this from a txt file
  sessionStartTime = datetime('now'); % Get this from a txt file
  outputFilename = sprintf('%s_%s.nwb',subject,datestr(sessionStartTime,'yyyymmddTHHMM'));
  fullOutputFile = fullfile(outputDir, outputFilename);
  fileID = outputFilename;  % This needs to be unique

  if(isfile(fullOutputFile))
    fprintf('A file associated with these data already exists and will be overwritten.\n');
  end

  % -- Load all text data files --
  % Find relevant files and load data

  % -- Store trial data --
  %trials = types.core.TimeIntervals(...)
  %nwb.intervals_trials = trials;
  
  % -- Store TimeSeries data --
  %pupil = types.core.TimeSeries(...)
  %nwb.acquisition.set('pupil', pupil);

  % -- Set up NWB file --
  nwb = NwbFile('session_description', 'Behavior data',...
		'identifier', fileID, ...
		'session_start_time', sessionStartTime);

  nwbExport(nwb, fullOutputFile);
  fprintf('Saved data to %s\n', fullOutputFile);

