import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "seedvc"))
sys.path.insert(0, repo_root)

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
import warnings
import argparse
import torch
import yaml

warnings.simplefilter("ignore")

from seedvc.modules.commons import *
import time

import torchaudio
import librosa

from seedvc.hf_utils import load_custom_model_from_hf


class SeedVCInference:
    """Voice conversion model using SeedVC architecture"""

    # Audio processing constants
    SAMPLE_RATE_16K = 16000
    MAX_REFERENCE_DURATION = 20  # seconds
    WINDOW_SIZE_SECONDS = 25  # processing window in seconds
    OVERLAP_TIME = 5  # overlapping time in seconds
    OVERLAP_FRAMES = 16
    NUM_MEL_BINS = 80

    def __init__(
        self,
        config_path,
        checkpoint_path,
        diffusion_steps,
        length_adjust,
        inference_cfg_rate,
    ):
        """Initialize voice conversion model

        Args:
            args: Command line arguments containing model configuration
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.diffusion_steps = diffusion_steps
        self.length_adjust = length_adjust
        self.inference_cfg_rate = inference_cfg_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = True
        self.window_size_samples = self.WINDOW_SIZE_SECONDS * self.SAMPLE_RATE_16K
        self.overlap_samples = self.OVERLAP_TIME * self.SAMPLE_RATE_16K

        # Model components initialized in load_models()
        self.sr: int = None
        self.hop_length: int = None
        self.model = None
        self.semantic_fn = None
        self.vocoder_fn = None
        self.campplus_model = None
        self.mel_fn = None

    def _setup_whisper(self, whisper_name):
        from transformers import AutoFeatureExtractor, WhisperModel

        whisper_model = WhisperModel.from_pretrained(
            whisper_name, torch_dtype=torch.float16
        ).to(self.device)
        del whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
            whisper_name
        )

        def semantic_fn(waves_16k):
            ori_inputs = self.whisper_feature_extractor(
                [waves_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
            )
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
            ).to(self.device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
            return S_ori

        return semantic_fn

    def load_models(self) -> None:
        """Load and initialize all model components"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )

        config = yaml.safe_load(open(self.config_path, "r"))
        model_params = recursive_munch(config["model_params"])
        self.sr = config["preprocess_params"]["sr"]
        self.hop_length = config["preprocess_params"]["spect_params"]["hop_length"]

        # Initialize models
        model_params.dit_type = "DiT"
        self.model = build_model(model_params, stage="DiT")

        # Load checkpoints
        self.model, _, _, _ = load_checkpoint(
            self.model,
            None,
            self.checkpoint_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        for key in self.model:
            self.model[key].eval()
            self.model[key].to(self.device)
        self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

        # Setup components
        self.semantic_fn = self._setup_whisper(model_params.speech_tokenizer.name)
        self.vocoder_fn = self._setup_vocoder(model_params.vocoder)
        self.campplus_model = self._setup_campplus()
        self.mel_fn = self._setup_mel_fn(config["preprocess_params"]["spect_params"])

        return self.model

    @torch.no_grad()
    def convert_audio(
        self, source_audio: torch.Tensor, ref_audio: torch.Tensor
    ) -> torch.Tensor:
        """Convert source audio to target voice

        Args:
            source_audio: Source audio tensor [1, T]
            ref_audio: Reference audio tensor [1, T]

        Returns:
            Converted audio tensor [1, T]
        """
        if self.model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # Preprocess audio
        converted_waves_16k = self._preprocess_audio(source_audio)
        ref_waves_16k = self._preprocess_audio(ref_audio)

        # Extract features
        source_tokens = self._process_semantic_tokens(converted_waves_16k)
        ref_tokens = self._process_semantic_tokens(ref_waves_16k)

        mel_source = self.mel_fn(source_audio)
        mel_ref = self.mel_fn(ref_audio)

        style = self._extract_style(ref_waves_16k)

        # Generate conversion
        source_cond = self._regulate_length(source_tokens, mel_source, is_source=True)
        ref_cond = self._regulate_length(ref_tokens, mel_ref, is_source=False)

        return self._generate_audio(source_cond, ref_cond, mel_ref, style)

    def _preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Resample audio to 16kHz for feature extraction"""
        return torchaudio.functional.resample(audio, self.sr, self.SAMPLE_RATE_16K)

    def _extract_style(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract style embedding from reference audio"""
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k,
            num_mel_bins=self.NUM_MEL_BINS,
            dither=0,
            sample_frequency=self.SAMPLE_RATE_16K,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return self.campplus_model(feat.unsqueeze(0))

    def _regulate_length(
        self, tokens: torch.Tensor, mel: torch.Tensor, is_source: bool
    ) -> torch.Tensor:
        """Adjust sequence length using length regulator"""
        target_len = mel.size(2)
        if is_source:
            target_len = int(target_len * self.length_adjust)
        target_lengths = torch.LongTensor([target_len]).to(mel.device)

        condition, *_ = self.model.length_regulator(
            tokens, ylens=target_lengths, n_quantizers=3, f0=None
        )
        return condition

    def _setup_vocoder(self, vocoder_params):
        from seedvc.modules.bigvgan import bigvgan

        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            vocoder_params.name, use_cuda_kernel=False
        )
        bigvgan_model.remove_weight_norm()
        return bigvgan_model.eval().to(self.device)

    def _setup_campplus(self):
        from seedvc.modules.campplus.DTDNN import CAMPPlus

        campplus_ckpt_path = load_custom_model_from_hf(
            "funasr/campplus", "campplus_cn_common.bin", config_filename=None
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(
            torch.load(campplus_ckpt_path, map_location="cpu")
        )
        return campplus_model.eval().to(self.device)

    def _setup_mel_fn(self, spect_params):
        mel_fn_args = {
            "n_fft": spect_params["n_fft"],
            "win_size": spect_params["win_length"],
            "hop_size": spect_params["hop_length"],
            "num_mels": spect_params["n_mels"],
            "sampling_rate": self.sr,
            "fmin": spect_params.get("fmin", 0),
            "fmax": None if spect_params.get("fmax", "None") == "None" else 8000,
            "center": False,
        }
        from seedvc.modules.audio import mel_spectrogram

        return lambda x: mel_spectrogram(x, **mel_fn_args)

    def _process_semantic_tokens(self, waves_16k):
        if waves_16k.size(-1) <= self.window_size_samples:
            return self.semantic_fn(waves_16k)

        S_alt_list = []
        buffer = None
        traversed_time = 0

        while traversed_time < waves_16k.size(-1):
            if buffer is None:
                chunk = waves_16k[
                    :, traversed_time : traversed_time + self.window_size_samples
                ]
            else:
                chunk = torch.cat(
                    [
                        buffer,
                        waves_16k[
                            :,
                            traversed_time : traversed_time
                            + (self.window_size_samples - self.overlap_samples),
                        ],
                    ],
                    dim=-1,
                )

            S_alt = self.semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * self.OVERLAP_TIME :])

            buffer = chunk[:, -self.overlap_samples :]
            traversed_time += (
                self.window_size_samples
                if traversed_time == 0
                else chunk.size(-1) - self.overlap_samples
            )

        return torch.cat(S_alt_list, dim=1)

    def _process_frame(self, chunk_cond, prompt_condition, mel2, style2):
        """Process a single frame of audio"""
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16 if self.fp16 else torch.float32,
        ):
            vc_target = self.model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2,
                style2,
                None,
                self.diffusion_steps,
                inference_cfg_rate=self.inference_cfg_rate,
                temperature=1.0,
            )

        vc_target = vc_target[:, :, mel2.size(-1) :]
        vc_wave = self.vocoder_fn(
            vc_target.float()
        ).squeeze()  # Match original inference2.py
        return vc_wave[None, :]  # Ensure 2D tensor [1, T]

    def _crossfade(
        self, chunk1: np.ndarray, chunk2: np.ndarray, overlap: int
    ) -> np.ndarray:
        """Apply crossfade between two audio chunks

        Args:
            chunk1: First audio chunk
            chunk2: Second audio chunk
            overlap: Number of samples to overlap

        Returns:
            Crossfaded audio chunk
        """
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2

        if len(chunk2) < overlap:
            chunk2[:overlap] = (
                chunk2[:overlap] * fade_in[: len(chunk2)]
                + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
            )
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    def _generate_audio(self, cond, prompt_condition, mel2, style2):
        """Generate audio in chunks with crossfade"""
        overlap_frame_len = self.OVERLAP_FRAMES
        overlap_wave_len = overlap_frame_len * self.hop_length
        max_context_window = self.sr // self.hop_length * self.WINDOW_SIZE_SECONDS
        max_source_window = max_context_window - mel2.size(2)

        processed_frames = 0
        generated_wave_chunks = []
        previous_chunk = None

        while processed_frames < cond.size(1):
            chunk_cond = cond[
                :, processed_frames : processed_frames + max_source_window
            ]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)

            # Process frame and ensure we get [1, T] shaped output
            vc_wave = self._process_frame(chunk_cond, prompt_condition, mel2, style2)

            if len(vc_wave.shape) == 1:
                vc_wave = vc_wave.unsqueeze(0)

            if processed_frames == 0:
                if is_last_chunk:
                    generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                    break
                output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]
            elif is_last_chunk:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                break
            else:
                output_wave = self._crossfade(
                    previous_chunk.cpu().numpy(),
                    vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                    overlap_wave_len,
                )
                generated_wave_chunks.append(output_wave)
                previous_chunk = vc_wave[0, -overlap_wave_len:]

            processed_frames += chunk_cond.size(1) - overlap_frame_len

        final_wave = np.concatenate(generated_wave_chunks)
        if len(final_wave.shape) == 1:
            final_wave = final_wave[None, :]

        return torch.tensor(final_wave).float()


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    """Main entry point for voice conversion"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load audio files
    try:
        source_audio = librosa.load(args.source, sr=22050)[0]
        ref_audio = librosa.load(args.target, sr=22050)[0]
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    # Preprocess audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)
    ref_audio = ref_audio[:, : 22050 * SeedVCInference.MAX_REFERENCE_DURATION]

    # Initialize and run conversion
    try:
        vc = SeedVCInference(
            args.config_path,
            args.checkpoint_path,
            args.diffusion_steps,
            args.length_adjust,
            args.inference_cfg_rate,
        )
        vc.load_models()

        time_start = time.time()
        vc_wave = vc.convert_audio(source_audio, ref_audio)
        time_end = time.time()

        rtf = (time_end - time_start) / vc_wave.size(-1) * vc.sr
        print(f"RTF: {rtf:.3f}")

        # Save output
        source_name = os.path.basename(args.source).split(".")[0]
        target_name = os.path.basename(args.target).split(".")[0]
        output_path = os.path.join(
            args.output,
            f"vc_{source_name}_{target_name}_{args.length_adjust}_{args.diffusion_steps}_{args.inference_cfg_rate}.wav",
        )

        os.makedirs(args.output, exist_ok=True)
        torchaudio.save(output_path, vc_wave.cpu(), vc.sr)

    except Exception as e:
        print(f"Error during voice conversion: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SeedVC voice conversion")
    # Required arguments
    parser.add_argument(
        "--checkpoint-path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the config file"
    )
    # Input/output paths
    parser.add_argument(
        "--source", type=str, required=True, help="Source audio file path"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Target/reference audio file path"
    )
    parser.add_argument(
        "--output", type=str, default="./reconstructed", help="Output directory path"
    )
    # Model parameters
    parser.add_argument(
        "--diffusion-steps", type=int, default=50, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--length-adjust", type=float, default=1.0, help="Length adjustment factor"
    )
    parser.add_argument(
        "--inference-cfg-rate",
        type=float,
        default=1,
        help="Classifier-free guidance rate",
    )
    parser.add_argument(
        "--fp16", type=str2bool, default=True, help="Use FP16 precision"
    )

    args = parser.parse_args()
    main(args)
