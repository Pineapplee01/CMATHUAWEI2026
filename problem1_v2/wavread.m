function varargout = wavread(filename, varargin)
%WAVREAD  Compatibility shim for pre-R2015b (COVAREP backcompat).
%  Avoids recursion with external/backcompatibility_2015/audioread.m by
%  temporarily removing that folder before calling MATLAB audioread.
if nargin >= 2 && ischar(varargin{1}) && strcmpi(varargin{1}, 'size')
    info = audioinfo(filename);
    varargout{1} = [info.TotalSamples, info.NumChannels];
    return;
end

bc_dirs = {};
cands = which('audioread', '-all');
if ischar(cands)
    cands = {cands};
end
for i = 1:numel(cands)
    if contains(cands{i}, 'backcompatibility')
        d = fileparts(cands{i});
        bc_dirs{end+1} = d; %#ok<AGROW>
        rmpath(d);
    end
end
cleanup = onCleanup(@() restore_bc(bc_dirs));

[y, Fs] = audioread(filename);
if nargin >= 2
    range = varargin{1};
    if isscalar(range)
        y = y(1:min(range, size(y, 1)), :);
    else
        y = y(range(1):min(range(2), size(y, 1)), :);
    end
end
varargout{1} = y;
if nargout >= 2
    varargout{2} = Fs;
end
if nargout >= 3
    info = audioinfo(filename);
    varargout{3} = info.BitsPerSample;
end
end

function restore_bc(dirs)
for i = 1:numel(dirs)
    if exist(dirs{i}, 'dir')
        addpath(dirs{i});
    end
end
end
