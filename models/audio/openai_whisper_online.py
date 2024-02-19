from openai import OpenAI


class OpenaiWhisperModelOnline:
    def __init__(self, api_key):
        self.api_key = api_key
        self.organization = "REPLCE WITH YOUR ORG_ID"

    def process_audio_file(self, audio_file_path):
        try:
            with open(audio_file_path, "rb") as audio_file:
                client = OpenAI(api_key=self.api_key)
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                )

                return {
                    # The online model of OpenAI Whisper doesn't returns lang
                    "detected_text": transcript.text,
                }
        except Exception as e:
            print(f"Error processing audio file: {e}\n\n")
            if "insufficient_quota" in str(e):
                print("Error: Insufficient quota. Aborting script.")
                exit(1)
            else:
                print(f"Error processing audio file: {e}")
                return None
