import os
import time
import warnings

import librosa
import numpy as np
import whisper

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="librosa.core.audio"
)


class OpenaiWhisperModelOffline:
    def __init__(self):

        start_time = time.time()
        self.model = whisper.load_model("medium")
        self.device = "cpu"
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Offline whisper model loaded time : {elapsed_time:.2f} seconds."
        )

    def process_audio_file(self, audio_file_path):
        audio, sr = librosa.load(audio_file_path, sr=None)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)

        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        result = self.model.transcribe(audio)
        detected_text = result["text"]

        return {
            "audio_sampling_rate": sr,
            "detected_lang": detected_lang,
            "detected_text": detected_text,
        }
