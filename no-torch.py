from speechbrain.processing.features import STFT, ISTFT
from speechbrain.processing.multi_mic import Covariance
from speechbrain.processing.multi_mic import SrpPhat
from speechbrain.processing.multi_mic import Mvdr
from scipy.signal import stft as Stft


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
istft = ISTFT(sample_rate=fs)
mvdr = Mvdr()
mics = torch.zeros((2,3), dtype=torch.float)
mics[0,:] = torch.FloatTensor([-0.05, -0.00, +0.00])
mics[1,:] = torch.FloatTensor([+0.05, +0.00, +0.00])
srpphat = SrpPhat(mics=mics)
beam = np.empty([1, 3200, 1])


def sd_callback(rec, frames, time, status):
    global beam
    sd.wait()
    torch_rec = torch.Tensor(rec)
    torch_rec = torch_rec.unsqueeze(0)
    Xs = stft(torch_rec)
    print(Xs)
    test_Xs = Stft(torch_rec, fs)
    print(test_Xs)
    XXs = cov(Xs)
    doas = srpphat(XXs)
    print(doas)
    Ys_mvdr = mvdr(Xs, XXs, doas, doa_mode=True, mics=mics, fs=fs)
    ys_mvdr = istft(Ys_mvdr)
    print(ys_mvdr.shape)
    beam = np.append(beam, ys_mvdr, 1)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=fs,
                    blocksize=int(fs * rec_duration),
                    callback=sd_callback):
    sd.sleep(int(duration * 1000))
print(beam.shape)    
sf.write('test.wav', beam[0, :, 0], fs)
    
