import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.backend.sox_io_backend import info as info_sox
from typing import Any, Dict, Optional, Union, Tuple


def time_to_samples(t: float, sample_rate: int) -> int:
    return int(t * sample_rate)


def time_to_frames(t: float, hop_time: float) -> int:
    return int(t / hop_time)


def sample_to_time(n_samples: int, sample_rate: int) -> float:
    return n_samples / sample_rate


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    info = info_sox(audio_path)
    return {
        "name": audio_path,
        "duration": sample_to_time(info.num_frames, info.sample_rate),
        "sample_rate": info.sample_rate,
        "num_frames": info.num_frames,
        "bits_per_sample": info.bits_per_sample,
        "num_channels": info.bits_per_sample,
        "encoding": info.encoding,
    }


def load_waveform(
    path: str,
    sample_rate: Optional[int] = 16000,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    normalize: bool = True,
    mono: bool = False,
    audio_normalize_threshold: float = 0.05,
) -> Tuple[torch.Tensor, int]:
    if start_time is None and end_time is None:
        x, sr = torchaudio.load(path)
    else:
        info = get_audio_info(path)

        start_frame = 0
        if start_time is not None:
            start_frame = time_to_samples(start_time, info["sample_rate"])

        end_frame = info["num_frames"]
        if end_time is not None:
            end_frame = time_to_samples(end_time, info["sample_rate"])

        num_frames = end_frame - start_frame
        x, sr = torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)

    if mono and x.shape[0] > 1:
        x = x.mean(dim=0).unsqueeze(0)
        if normalize:
            if x.abs().max() > audio_normalize_threshold:
                x /= x.abs().max()

    if sample_rate is not None:
        if sr != sample_rate:
            x = AF.resample(x, orig_freq=sr, new_freq=sample_rate)
            sr = sample_rate
    return x, sr


if __name__ == "__main__":

    path = "example/student_long_female_en-US-Wavenet-G.wav"
    x, sr = load_waveform(path)

    info = get_audio_info(path)