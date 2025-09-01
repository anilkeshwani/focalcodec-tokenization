#!/usr/bin/env python

"""
Encode the Multilingual LibriSpeech (MLS) dataset using FocalCodec.

Speed of  tokenization:
- GPU: ~27 samples/s
"""

import json
import logging
import os
import sys
import warnings
from argparse import ArgumentParser, Namespace
from math import ceil
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


# Logging configuration
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATEFMT: str = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

# Constants
MLS_SPLIT_SIZES = {"train": 10_808_037, "dev": 3_807, "test": 3_769}
FC_CONFIG = "lucadellalib/focalcodec_50hz"

# Local MLS Dataset Paths
_MLS_SEGMENTS_PATH = "/mnt/scratch-artemis/shared/datasets/MLS/{}/segments.txt"
_MLS_AUDIO_DIR = "/mnt/scratch-artemis/shared/datasets/MLS/{}/audio"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "focalcodec_mls"

# Reconstruction path
RECONSTRUCTION_PATH = None  # Placeholder for reconstruction path; not used


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Encode MLS dataset with FocalCodec.")
    # Required
    parser.add_argument("idx_block", type=int, help="Block index to process (0-based)")
    parser.add_argument("--split", type=str, required=True, choices=["train", "dev", "test"])
    # Optional
    parser.add_argument("--block_size", type=int, default=300_000)  # <= 4 hours at 27 samples/s on a NVIDIA RTX A6000
    parser.add_argument("--output_jsonl", type=Path)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        dest="device_str",  # avoid overwriting variable with different type downstream
        choices=["cpu", "cuda"],
        help="Device to use for inference",
    )
    parser.add_argument("--force_reload", action="store_true")
    args = parser.parse_args()

    if args.idx_block < 0 or args.idx_block * args.block_size >= MLS_SPLIT_SIZES[args.split]:
        raise ValueError(
            f"Invalid block index {args.idx_block} for split '{args.split}' and block size {args.block_size}."
        )

    return args


def mls_id_to_path(mls_id: str, audio_dir: Path, suffix: str = ".flac") -> Path:
    """Infer path of the audio file from the MLS ID and audio directory.

    Args:
        mls_id (str): ID as found in transcripts.txt file e.g. 10214_10108_000000
        audio_dir (Path): "audio" directory e.g. /mnt/scratch-artemis/anilkeshwani/data/MLS/mls_english/dev/audio
        suffix (str, optional): File extension. Defaults to ".flac".

    Returns:
        Path: Resolved path pointing to audio file
    """
    speaker_id, book_id, file_specifier = mls_id.removesuffix(suffix).split("_")
    return (audio_dir / speaker_id / book_id / mls_id).with_suffix(suffix)


@torch.inference_mode()
def focalcodec_encode_mls(
    idx_block: int,
    block_size: int,
    split: str,
    output_jsonl: Path | None,
    device_str: str,
    force_reload: bool,
):
    device = torch.device(device_str)
    # Load FocalCodec model
    codec = torch.hub.load("lucadellalib/focalcodec", "focalcodec", config=FC_CONFIG, force_reload=force_reload)
    FC_SAMPLE_RATE = codec.sample_rate
    if FC_SAMPLE_RATE != 16_000:
        raise AssertionError("FocalCodec audio input sample rate is 16kHz: https://arxiv.org/pdf/2502.04465v1")
    codec.eval().requires_grad_(False)
    codec.to(device)

    mls_split_size = MLS_SPLIT_SIZES[split]
    mls_segments = _MLS_SEGMENTS_PATH.format(split)
    mls_audio_dir = Path(_MLS_AUDIO_DIR.format(split))
    n_blocks = ceil(mls_split_size / block_size)

    with open(mls_segments, "r") as f:
        mls_ids: list[str] = [line.strip().split(None, 1)[0] for line in f]

    if len(mls_ids) != mls_split_size:
        raise ValueError(f"Expected {mls_split_size} MLS IDs in {mls_segments}, but found {len(mls_ids)}.")

    # Get the block of MLS IDs to process
    start_idx = idx_block * block_size
    end_idx = min((idx_block + 1) * block_size, mls_split_size)
    mls_ids = mls_ids[start_idx:end_idx]

    if output_jsonl is None:
        idx_block_label = str(idx_block + 1).zfill(len(str(n_blocks)))  # NOTE 1-indexed block label
        jsonl_filename = f"{split}-mls-focalcodec-{idx_block_label}-of-{n_blocks}.jsonl"
        output_jsonl = OUTPUT_DIR / split / jsonl_filename

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "x") as f:
        for mls_id in tqdm(mls_ids, desc="Processing MLS with FocalCodec"):
            # Load and pre-process speech waveform
            wav, sr = torchaudio.load(mls_id_to_path(mls_id, mls_audio_dir))

            # monophonic checking
            if wav.size(0) > 1:
                warnings.warn(f"Audio {mls_id} is not monophonic. Shape: {wav.shape}. Taking the first channel.")
                wav = wav[:1, :]  # take first dimension whilst maintaining second - channel - dimension

            # sample rate checking
            if sr != FC_SAMPLE_RATE:
                LOGGER.debug(f"Audio {mls_id} has sample rate {sr}, expected {FC_SAMPLE_RATE}. Resampling...")
                wav = torchaudio.functional.resample(wav, sr, FC_SAMPLE_RATE)

            wav = wav.to(device)

            # Encode audio into tokens
            fc_tokens = codec.sig_to_toks(wav)  # Shape: (batch, time)

            # Convert tokens to their corresponding binary spherical codes
            # codes = codec.toks_to_codes(toks)  # Shape: (batch, time, log2 codebook_size)

            # Decode tokens back into a waveform
            # rec_sig = codec.toks_to_sig(toks)

            # Save the reconstructed audio
            # rec_sig = torchaudio.functional.resample(rec_sig, codec.sample_rate, sample_rate)
            # torchaudio.save(RECONSTRUCTION_PATH, rec_sig, sample_rate)

            focalcodec_sample: dict[str, list[int]] = {"ID": mls_id, "focalcodec_tokens": fc_tokens.squeeze(0).tolist()}
            f.write(json.dumps(focalcodec_sample) + "\n")
            # f.flush()

    print(f"Completed. Encoded block {idx_block} to {output_jsonl}.")


if __name__ == "__main__":
    focalcodec_encode_mls(**vars(parse_args()))
