import torch
import torchaudio
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests
from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")

#torchaudio.info() fetches audio metadata. takes path-like/file-like object as input
metadata = torchaudio.info(SAMPLE_WAV)
print(metadata)

# Where

# sample_rate is the sampling rate of the audio
# num_channels is the number of channels
# num_frames is the number of frames per channel
# bits_per_sample is bit depth
# encoding is the sample coding format
# encoding can take on one of the following values:

# "PCM_S": Signed integer linear PCM
# "PCM_U": Unsigned integer linear PCM
# "PCM_F": Floating point linear PCM
# "FLAC": Flac, Free Lossless Audio Codec
# "ULAW": Mu-law, [wikipedia]
# "ALAW": A-law [wikipedia]
# "MP3" : MP3, MPEG-1 Audio Layer III
# "VORBIS": OGG Vorbis [xiph.org]
# "AMR_NB": Adaptive Multi-Rate [wikipedia]
# "AMR_WB": Adaptive Multi-Rate Wideband [wikipedia]
# "OPUS": Opus [opus-codec.org]
# "GSM": GSM-FR [wikipedia]
# "HTK": Single channel 16-bit PCM
# "UNKNOWN" None of above

#LOADING AUDIO FILES
#torchaudio.load() loads audio files into a tensor. takes path-like/file-like object as input,
#and returns a tuple of waveform(Tensor) and sample_rate(int)
#by default the resulting tensor object has dtype=torch.float32 and its value range is [-1.0, 1.0]

waveform, sample_rate = torchaudio.load(SAMPLE_WAV)


def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1, figsize=(15, 10))
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block = False)
    

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show(block=False)
    

# plot_waveform(waveform, sample_rate)          
# plot_specgram(waveform, sample_rate)
Audio(waveform.numpy()[0], rate=sample_rate)

#