import numpy as np
import soundfile as sf

# Generate 3 seconds of white noise
sr = 16000
duration = 3
audio = np.random.uniform(-0.1, 0.1, int(sr * duration)).astype(np.float32)

# Save to wav
sf.write('d:\\project\\mydegree\\streamllm\\test_audio.wav', audio, sr)
print("Created d:\\project\\mydegree\\streamllm\\test_audio.wav")
