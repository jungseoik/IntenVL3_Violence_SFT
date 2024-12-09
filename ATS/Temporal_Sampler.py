import numpy as np
from scipy import interpolate

def reshape(arr, num_frames):
    x = np.arange(0, arr.shape[0])
    f = interpolate.interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.linspace(0, arr.shape[0], num_frames)
    up_scale = f(scale_x)
    return up_scale

def density_awared_sample(num_frames, anomaly_score, sample_len=16, tau=0.1):
    if num_frames <= sample_len or sum(anomaly_score) < 1:
        sampled_idxs = list(np.rint(np.linspace(0, num_frames-1, sample_len)))
        return sampled_idxs
    else:
        scores = [score+tau for score in anomaly_score]
        score_cumsum = np.concatenate((np.zeros((1,), dtype=float), np.cumsum(scores)), axis=0)
        max_score_cumsum = np.round(score_cumsum[-1]).astype(int)
        f_upsample = interpolate.interp1d(score_cumsum, np.arange(num_frames+1), kind='linear', axis=0, fill_value='extrapolate')
        scale_x = np.linspace(1, max_score_cumsum, sample_len)
        sampled_idxs = f_upsample(scale_x)
        sampled_idxs = [min(num_frames-1, max(0, int(idx))) for idx in sampled_idxs]
        return sampled_idxs
    