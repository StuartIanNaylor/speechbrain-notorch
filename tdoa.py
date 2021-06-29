from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import GccPhat
from speechbrain.processing.multi_mic import DelaySum

import torch
import sounddevice as sd
import soundfile as sf
import numpy as np

fs = 16000
rec_duration = 0.20
duration = 10
sample_rate = 16000
num_channels = 2

stft = STFT(sample_rate=fs)
cov = Covariance()
gccphat = GccPhat()
delaysum = DelaySum()
istft = ISTFT(sample_rate=fs)
beam = np.empty([1, 3200, 1])


def sd_callback(rec, frames, time, status):
    global beam
    sd.wait()
    torch_rec = torch.Tensor(rec)
    torch_rec = torch_rec.unsqueeze(0)
    Xs = stft(torch_rec)
    print(Xs.shape)
    XXs = cov(Xs)
    tdoas = gccphat(XXs) 

    print(tdoas)
    Ys_ds = delaysum(Xs, tdoas)
    ys_ds = istft(Ys_ds)
    print(ys_ds.shape)
    beam = np.append(beam, ys_ds, 1)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=fs,
                    blocksize=int(fs * rec_duration),
                    callback=sd_callback):
    sd.sleep(int(duration * 1000))
print(beam.shape)    
sf.write('test.wav', beam[0, :, 0], fs)
    


