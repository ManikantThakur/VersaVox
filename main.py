import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime

import langid
from docopt import docopt
from google.cloud import speech_v1p1beta1 as speech


import traceback


from models.audio.google_asr import GoogleASRModel
from models.audio.openai_whisper_offline import OpenaiWhisperModelOffline
from models.audio.openai_whisper_online import OpenaiWhisperModelOnline
from models.text.ollama_offline import LanguageDetectionOllamaAPI

TEXT_LIMIT = 10
AUDIO_LIMIT = 5

audio_model = None


warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging configuration
logging.basicConfig(
    filename="language_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Supported models
TEXT_MODELS = ["gpt3.5", "gpt4", "llama2"]
AUDIO_MODELS = ["google_asr", "whisper"]

doc = """
Language Detection Tool

Usage:
  main.py --data-type <type> --data-path <path> --model <model> --mode <mode>

Arguments:
  --data-type: Specify 'text' or 'audio'.
  --data-path: Path to the text file or audio directory/file.
  --model: Specify the model for language detection.
    Text: {text_models}
    Audio: {audio_models}
  --mode: Specify the mode, 'offline' or 'online'
""".format(
    text_models=", ".join(TEXT_MODELS), audio_models=", ".join(AUDIO_MODELS)
)


def detect_google_asr(audio_path, mode):
    try:
        if mode == "online":
            return audio_model.process_audio_file(audio_path)
    except Exception as e:
        logging.error(f"Error in detect_google_asr: {e}")
        return None


def detect_openai_whisper(audio_path, mode):
    try:
        if mode == "offline":
            whisper_model = OpenaiWhisperModelOffline()
            return whisper_model.process_audio_file(audio_path)
        elif mode == "online":
            return audio_model.process_audio_file(audio_path)
        else:
            logging.warning(f"Unsupported Whisper model type: {mode}")
            return None
    except Exception as e:
        logging.error(f"Error in detect_openai_whisper: {e}")
        return None


def detect_gpt35_language(text):
    print("The function detect_gpt35_language has not been implemented yet.")
    return None


def detect_gpt4_language(text):
    print("The function detect_gpt4_language has not been implemented yet.")
    return None


def detect_llama2_language(text):
    try:
        api_instance = LanguageDetectionOllamaAPI()
        result = api_instance.detect_language(text)
        if result is not None:
            return result
        else:
            print("Error: Language detection result is None.")
            return None
    except Exception as e:
        # print(result)
        traceback.print_exc()
        print(f"Error in detect_llama2_language: {e}")

        return None


def detect_audio_language(audio_path, model, mode):
    global audio_model
    if audio_model is None:
        audio_model_functions = {
            "google_asr": detect_google_asr,
            "whisper": detect_openai_whisper,
        }

        audio_model = audio_model_functions.get(model)

    model_function = audio_model
    if model_function:
        result = model_function.process_audio_file(audio_path)
        return result
    else:
        logging.warning(f"Unsupported audio model: {model}")
        return None


def detect_text_language(text, model):
    text_model_functions = {
        "gpt3.5": detect_gpt35_language,
        "gpt4": detect_gpt4_language,
        "llama2": detect_llama2_language,
    }

    text_model_function = text_model_functions.get(model.lower())

    if text_model_function:
        return text_model_function(text)
    else:
        logging.warning(f"Unsupported text model: {model}")
        return None


def process_audio_directory(directory_path, model, mode):
    global audio_model
    if audio_model is None:
        logging.error("Audio model is not defined.")
        return
    processed_files = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_path = os.path.join(directory_path, filename)
            detected_lang = detect_audio_language(audio_path, model, mode)
            if detected_lang:
                print(f"{filename}: {detected_lang}")
            else:
                logging.error(
                    f"Unable to detect language for {filename} using audio model: {model}"
                )

            processed_files += 1


def process_text_file(file_path, text_model, limit=None):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines_processed = 0
            for line in file:
                line = line.strip()
                if line:
                    detected_lang = detect_text_language(line, text_model)
                    if detected_lang:
                        text = line.strip('"').strip("'").strip(",")
                        print(f"{text}, {detected_lang}")
                    else:
                        logging.error(
                            f"Language detection failed using text model: {text_model}"
                        )
                    lines_processed += 1

                    if limit is not None and lines_processed >= limit:
                        print(
                            f"Processed {lines_processed} lines. Limit reached. Aborting."
                        )
                        exit(1)

    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")


def print_audio_summary(directory_path, audio_models, mode):
    header = f"ASR_CLASSIFICATION_AUDIO_FILE, {', '.join(audio_models)}"
    print(header)

    for filename in os.listdir(directory_path):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_path = os.path.join(directory_path, filename)
            detected_languages = [
                detect_audio_language(audio_path, model, mode)
                for model in audio_models
            ]
            values = f"{audio_path}, {', '.join(detected_languages)}"
            print(values)


def print_text_summary(text_data, text_models):
    header = f"ASR_CLASSIFICATION_TEXT, {', '.join(text_models)}"
    print(header)

    detected_languages = [
        detect_text_language(text_data, model) for model in text_models
    ]
    values = f"{text_data}, {', '.join(detected_languages)}"
    print(values)


def print_summary(data_path, models, data_type, mode):
    print(f"Print Summary")
    if data_type == "audio":
        print_audio_summary(data_path, models, mode=mode)
    elif data_type == "text":
        with open(data_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        print_text_summary(text_data, models)


def main():
    # print(f"IN main...")
    args = docopt(doc)
    data_type = args["<type>"]
    data_path = args["<path>"].strip()
    model = args["<model>"]
    mode = args["<mode>"]

    global audio_model

    audio_model = None

    print(f"\n", "." * 100)
    print(
        f"\nData Type : {data_type}, Data Path : {data_path}, Model : {model}, Mode : {mode}\n"
    )
    print(f"." * 100, "\n")

    if data_type == "text":
        if model in TEXT_MODELS:
            if model == "llama2" and mode == "online":
                print(
                    f"Online mode for llama2 in not available. Aborting execution."
                )
                exit(1)
            if model == "llama2" and mode == "offline":
                if os.path.exists(data_path):
                    process_text_file(data_path, model, limit=TEXT_LIMIT)
        else:
            print(f"Invalid text model. Choose from {', '.join(TEXT_MODELS)}.")
    elif data_type == "audio":
        if model in AUDIO_MODELS:
            if model == "google_asr" and mode == "online":
                print(f"Performing Google ASR online")
                credentials_path = os.environ.get(
                    "GOOGLE_APPLICATION_CREDENTIALS"
                )
                if not credentials_path or not os.path.isfile(credentials_path):
                    raise EnvironmentError(
                        "GOOGLE_APPLICATION_CREDENTIALS is not set or is not a valid file. Aborting execution."
                    )
                try:
                    with open(credentials_path, "r") as credentials_file:
                        json.load(credentials_file)
                except json.JSONDecodeError:
                    raise ValueError(
                        f"The file specified in GOOGLE_APPLICATION_CREDENTIALS is not a valid JSON file. Aborting execution."
                    )

                # audio_model = OpenaiWhisperModelOnline(api_key)
                audio_model = GoogleASRModel()
            elif model == "google_asr" and mode == "offline":
                print(
                    f"Google ASR is not available in offline mode. Aborting execution."
                )
                exit(1)
            elif model == "whisper" and mode == "online":
                api_key = os.environ.get("OPENAI_API_KEY")
                # print(f"api_key : {api_key}")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY environment variable is not set."
                    )
                audio_model = OpenaiWhisperModelOnline(api_key)
            elif model == "whisper" and mode == "offline":
                audio_model = OpenaiWhisperModelOffline()

            if os.path.exists(data_path):
                if os.path.isdir(data_path):
                    process_audio_directory(data_path, model, mode)
                else:
                    detected_lang = detect_audio_language(
                        data_path, model, mode
                    )
                    if detected_lang:
                        print(f"Detected Language: {detected_lang}")
                        print_summary(data_path, [model], "audio", mode=mode)
                    else:
                        print("Unable to detect language from audio.")
            else:
                print(
                    "Invalid audio file or directory path. Try using absolute path."
                )
        else:
            print(
                f"Invalid audio model. Choose from {', '.join(AUDIO_MODELS)}."
            )
    else:
        print("Invalid data type. Choose either 'text' or 'audio.'")


if __name__ == "__main__":
    main()
