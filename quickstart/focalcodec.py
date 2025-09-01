#!/usr/bin/env python

from pathlib import Path

import torch
import torchaudio


DATA_DIR = Path(__file__).parent / "data"

# Load FocalCodec model
config = "lucadellalib/focalcodec_50hz"
codec = torch.hub.load("lucadellalib/focalcodec", "focalcodec", config=config, force_reload=True)
codec.eval().requires_grad_(False)

# Load and preprocess the input audio
audio_file = DATA_DIR / "mls_train_100_2315_000000.flac"
sig, sample_rate = torchaudio.load(audio_file)
sig = torchaudio.functional.resample(sig, sample_rate, codec.sample_rate)

# Encode audio into tokens
toks = codec.sig_to_toks(sig)  # Shape: (batch, time)
print(toks.shape)
print(toks)

# Convert tokens to their corresponding binary spherical codes
codes = codec.toks_to_codes(toks)  # Shape: (batch, time, log2 codebook_size)
print(codes.shape)
print(codes)

# Decode tokens back into a waveform
rec_sig = codec.toks_to_sig(toks)

# Save the reconstructed audio
rec_sig = torchaudio.functional.resample(rec_sig, codec.sample_rate, sample_rate)
torchaudio.save(DATA_DIR / "reconstruction.wav", rec_sig, sample_rate)
