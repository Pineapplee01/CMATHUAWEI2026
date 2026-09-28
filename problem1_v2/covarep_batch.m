function covarep_batch(covarep_root, wav_dir, hop_sec)
%COVAREP_BATCH  Batch-extract COVAREP 74-d features for all wavs in wav_dir.
%  Writes <stem>.mat next to each wav (fields: features [T x 74], names).
if nargin < 3 || isempty(hop_sec)
    hop_sec = 0.01;
end
addpath(genpath(covarep_root));
% Prefer MATLAB builtin audioread; COVAREP backcompat shim calls removed wavread.
bc = fullfile(covarep_root, 'external', 'backcompatibility_2015');
if exist(bc, 'dir')
    rmpath(genpath(bc));
end
% Keep Q1 dir on path so local wavread.m shim is available if needed.
q1 = fileparts(mfilename('fullpath'));
addpath(q1);
COVAREP_feature_extraction(wav_dir, hop_sec);
end
