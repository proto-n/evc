from inference import SeedVCInference
import torch
import time
import librosa
import os
import torchaudio

"""Main entry point for voice conversion"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load audio files
try:
    source_audio = librosa.load("test2/bence2.wav", sr=22050)[0]
    ref_audio = librosa.load("test2/out.wav", sr=22050)[0]
except Exception as e:
    print(f"Error loading audio files: {e}")

# Preprocess audio
source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)
ref_audio = ref_audio[:, : 22050 * SeedVCInference.MAX_REFERENCE_DURATION]

# Initialize and run conversion
# try:
vc = SeedVCInference(
    "/mnt/idms/kdomokos/workspace/discc/seed-vc/models/config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    "/mnt/idms/kdomokos/workspace/discc/seed-vc/runs/tune-run-3-target/DiT_epoch_00000_step_00100.pth",
    30,
    1,
    0.97,
)
vc.load_models()

time_start = time.time()
vc_wave = vc.convert_audio(source_audio, ref_audio)
time_end = time.time()

rtf = (time_end - time_start) / vc_wave.size(-1) * vc.sr
print(f"RTF: {rtf:.3f}")

output_path = "ex.wav"
torchaudio.save(output_path, vc_wave.cpu(), vc.sr)

# except Exception as e:
#     print(f"Error during voice conversion: {e}")
