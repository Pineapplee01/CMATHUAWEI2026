"""Frame-wise RPRM, explicitly separate block-missing and whole-modality probes."""
import torch


def corrupt(features, valid, rate, mode="random", seed=None, generator=None,
            modalities=("text", "audio", "vision")):
    if not 0 <= rate <= 1:
        raise ValueError("missing rate must lie in [0,1]")
    if mode not in {"random", "block", "whole"}:
        raise ValueError("mode must be random, block or whole")
    if not set(modalities) <= set(features):
        raise ValueError("unknown modality")
    if generator is None:
        generator = torch.Generator().manual_seed(2026 if seed is None else seed)
    result, observed = {}, {}
    for m, x in features.items():
        keep = valid[m].clone()
        if m in modalities:
            if mode == "random":
                keep &= torch.rand(keep.shape, generator=generator).to(keep.device) >= rate
            elif mode == "whole":
                keep &= (torch.rand((keep.shape[0], 1), generator=generator).to(keep.device) >= rate)
            else:
                for i in range(len(keep)):
                    positions = torch.where(valid[m][i])[0]
                    count = int(round(len(positions) * rate))
                    if count:
                        start = int(torch.randint(len(positions) - count + 1, (1,), generator=generator))
                        keep[i, positions[start:start + count]] = False
        result[m] = x.masked_fill(~keep[..., None], 0)
        observed[m] = keep
    return result, observed
