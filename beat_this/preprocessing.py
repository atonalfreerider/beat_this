import numpy as np
import torch
import torchaudio
import tempfile
import os


def load_audio(path, dtype="float64"):
    # Check if the file is a video format
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    if any(str(path).lower().endswith(ext) for ext in video_extensions):
        try:
            from pydub import AudioSegment
            # Extract audio from video to temporary WAV file
            audio = AudioSegment.from_file(path)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                audio.export(tmp_path, format='wav')
            
            try:
                # Load the temporary audio file
                waveform, samplerate = torchaudio.load(tmp_path, channels_first=False)
                waveform = np.asanyarray(waveform.squeeze().numpy(), dtype=dtype)
                return waveform, samplerate
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            print(f"Failed to extract audio from video: {e}")
            # Fall through to try other methods
    
    try:
        waveform, samplerate = torchaudio.load(path, channels_first=False)
        waveform = np.asanyarray(waveform.squeeze().numpy(), dtype=dtype)
        return waveform, samplerate
    except Exception:
        # in case torchaudio fails, try soundfile
        try:
            import soundfile as sf

            return sf.read(path, dtype=dtype)
        except Exception:
            # some files are not readable by soundfile, try madmom
            try:
                import madmom

                return madmom.io.load_audio_file(str(path), dtype=dtype)
            except Exception:
                raise RuntimeError(f'Could not load audio from "{path}".')


class LogMelSpect(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=441,
        f_min=30,
        f_max=11000,
        n_mels=128,
        mel_scale="slaney",
        normalized="frame_length",
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        ).to(device)
        self.log_multiplier = log_multiplier

    def forward(self, x):
        """Input is a waveform as a monodimensional array of shape T,
        output is a 2D log mel spectrogram of shape (F,128)."""
        return torch.log1p(self.log_multiplier * self.spect_class(x).T)
