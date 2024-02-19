import io
import os

import langdetect
import librosa
import numpy as np
from google.cloud import speech_v1p1beta1 as speech
from pydub import AudioSegment


class GoogleASRModel:
    def __init__(self):

        pass

    def process_audio_file(self, audio_file_path):
        try:
            client = speech.SpeechClient()
            first_lang = "en"  # -IN
            second_lang = "hi"

            audio, sr = librosa.load(audio_file_path, sr=None)

            wav_file_path = self.convert_to_wav(audio, sr)

            with io.open(wav_file_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sr,
                language_code=first_lang,
                alternative_language_codes=[second_lang],
            )

            response = client.recognize(config=config, audio=audio)

            if response.results:
                detected_text = response.results[0].alternatives[0].transcript

            else:
                detected_text = ""

            return {
                "audio_sampling_rate": sr,
                "detected_lang": response.results[0].language_code,
                "detected_text": detected_text,
            }

        except Exception as e:

            print(f"Error in process_audio_file: {str(e)}")
            return {"error": str(e)}

    def convert_to_wav(self, audio, sr):
        try:
            # Convert audio to WAV format using pydub with 16-bit samples
            audio_16bit = (audio * 32767).astype(np.int16)
            audio_segment = AudioSegment(
                audio_16bit.tobytes(), frame_rate=sr, sample_width=2, channels=1
            )
            wav_file_path = "converted_audio.wav"
            audio_segment.export(wav_file_path, format="wav")
            return wav_file_path
        except Exception as e:
            print(f"Error in convert_to_wav: {str(e)}")
            return None
