import numpy as np
import soundfile as sf
from pathlib import Path

npz_path = Path("flies.npz")
out_dir = Path("audio")
out_dir.mkdir(exist_ok=True)

data = np.load(npz_path)
recording = data["recording"].squeeze()
samplerate = int(data["samplerate"])

out_path = out_dir / "sample_002.wav"
sf.write(out_path, recording, samplerate)

print("Saved:", out_path)
print("Samplerate:", samplerate)
print("Duration seconds:", len(recording) / samplerate)
print("Pulse annotations:", len(data["pulsetimes"]))
print("Sine annotations:", len(data["sinetimes"]))