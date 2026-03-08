
import librosa
import numpy as np
import soundfile as sf
import os

path = r"backend/mashup_backend/.sessions/23181abe-b6c9-4316-ae00-7b21f254fcca/preview.wav"

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

try:
    # Load with soundfile first to check raw integrity
    data, sr = sf.read(path)
    print(f"File info: {len(data)} samples, {sr} Hz, {len(data)/sr:.2f} seconds")
    
    # Check for NaNs or Infs
    nans = np.isnan(data).sum()
    infs = np.isinf(data).sum()
    print(f"NaNs: {nans}, Infs: {infs}")

    # Check RMS in 5 second blocks
    block_size = 5 * sr
    for i in range(0, len(data), block_size):
        block = data[i : i + block_size]
        rms = np.sqrt(np.mean(block**2))
        print(f"Block {i//sr}-{(i+len(block))//sr}s RMS: {rms:.6f}")

    # Check for exact zeros (silence)
    zeros = (data == 0).sum()
    print(f"Zero samples: {zeros} ({zeros/len(data)*100:.1f}%)")

except Exception as e:
    print(f"Error reading file: {e}")
