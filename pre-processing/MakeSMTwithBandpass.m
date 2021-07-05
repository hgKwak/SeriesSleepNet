function out = MakeSMTwithBandpass(EEG, varargin)

if isempty(varargin)
    out.dat = EEG.data;
else
    low_pass = varargin{1};
    high_pass = varargin{2};
    
    if low_pass > high_pass
        EEG = pop_eegfiltnew(EEG, high_pass, low_pass, 660, 1, [], 1);
    else
        EEG = pop_eegfiltnew(EEG, low_pass, high_pass, 660, 0, [], 1);
    end
    out.dat = EEG.data;
end