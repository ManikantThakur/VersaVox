import json
import requests
import logging
from typing import Optional, Union
from os.path import join, dirname, abspath


class LanguageDetectionError(Exception):
    pass


DEFAULT_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 10
ALLOWED_LANGUAGES = {"en", "hi"}
MAX_LANGUAGE_RETRIES = 3


class LanguageDetectionOllamaAPI:
    CONFIG_DIR = "config"
    CONFIG_FILE = "config.json"
    ERROR_MESSAGE = "Error"
    MAX_LANGUAGE_RETRIES = 10

    def __init__(self):
        self.config = self._load_config()
        self._initialize_api_config()
        self.logger = self._setup_logger()

    def _initialize_api_config(self):
        ollama_config = self.config.get("ollama", {})
        self.api_url = ollama_config.get("api_url", DEFAULT_API_URL)
        self.max_retries = ollama_config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.timeout_seconds = ollama_config.get(
            "timeout_seconds", DEFAULT_TIMEOUT_SECONDS
        )

    def _load_config(self) -> dict:
        config_path = join(
            dirname(dirname(dirname(abspath(__file__)))),
            "config",
            self.CONFIG_FILE,
        )
        default_config = {
            "ollama": {
                "api_url": DEFAULT_API_URL,
                "max_retries": DEFAULT_MAX_RETRIES,
                "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            },
            "log_level": logging.INFO,
        }
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                user_config = json.load(file)
                default_config.update(user_config)
                return default_config
        except FileNotFoundError:
            self.logger.warning("Config file not found. Using default configuration.")
            return default_config
        except json.JSONDecodeError as e:
            self.logger.error("Error decoding JSON in config file: %s", e)
            return default_config

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=self.config.get("log_level", logging.INFO))
        return logger

    def detect_language(self, prompt: str) -> str:
        retries = 0
        while retries < self.MAX_LANGUAGE_RETRIES:
            try:
                li_prompt = self._prepare_prompt(prompt)
                response_data = self._send_request(li_prompt)

                if response_data is not None:
                    lang_value = self._extract_lang_value(response_data, prompt)
                    if lang_value is not None:
                        return lang_value
                    else:
                        return self.ERROR_MESSAGE

                return self.ERROR_MESSAGE

            except LanguageDetectionError as e:
                if "Invalid language code" in str(e):
                    invalid_code = self._extract_invalid_lang_code(e)
                    self.logger.warning(
                        "Invalid language code '%s' detected. Retrying language detection (attempt %s)",
                        invalid_code,
                        retries + 1,
                    )

                    retries += 1
                else:
                    self.logger.error("Language detection failed: %s", e)
                    return self.ERROR_MESSAGE
            except requests.RequestException as e:
                self.logger.error("Error during API request: %s", e)
                return self.ERROR_MESSAGE
            except Exception as e:
                self.logger.error("An unexpected error occurred: %s", e)
                return self.ERROR_MESSAGE

        self.logger.warning(
            "Maximum retries (%s) reached. Unable to detect a valid language code.",
            self.MAX_LANGUAGE_RETRIES,
        )
        return self.ERROR_MESSAGE

    def _prepare_prompt(self, prompt: str) -> str:
        prepared_prompt = f"Identify only the lang code in: {prompt}"
        self.logger.debug("Prepared prompt: %s", prepared_prompt)
        return prepared_prompt

    def _send_request(self, li_prompt: str, retries: int = 0) -> dict:
        data = {
            "model": "llama2",
            "prompt": li_prompt,
            "stream": False,
            "format": "json",
        }

        try:
            response = requests.post(
                self.api_url, json=data, timeout=self.timeout_seconds
            )

            response.raise_for_status()

            response_data = response.json().get("response", {})
            self.logger.debug("API response: %s", response_data)
            return response_data

        except requests.RequestException as e:
            if retries < self.max_retries:
                self.logger.warning(
                    "Retrying API request (attempt %s): %s", retries + 1, e
                )
                return self._send_request(li_prompt, retries + 1)
            else:
                raise LanguageDetectionError(f"Error during API request: {e}") from e

    def _extract_lang_value(
        self, response_data: Union[dict, str], prompt: str
    ) -> Union[str, None]:
        if isinstance(response_data, str):
            try:
                response_data_dict = json.loads(response_data)
            except json.JSONDecodeError as e:
                raise LanguageDetectionError(
                    f"{prompt} Error decoding JSON: {e}\n{response_data}\n"
                ) from e
        else:
            response_data_dict = response_data

        lang_value = response_data_dict.get("lang") or response_data_dict.get(
            "lang_code"
        )

        if lang_value is None:
            raise LanguageDetectionError(
                f"{prompt} Error: Invalid or missing 'lang' key in the response.\n{response_data}\n"
            )

        # Check if lang_value is in the set of allowed languages
        if lang_value not in ALLOWED_LANGUAGES:
            raise LanguageDetectionError(
                f"{prompt} Error: Invalid language code '{lang_value}' in the response.\n{response_data}\n"
            )
        self.logger.info("Detected language value: %s", lang_value)
        return lang_value

    def _extract_invalid_lang_code(self, error: LanguageDetectionError) -> str:
        # Extracting the invalid language code from the error message
        error_msg = str(error)
        start_pos = error_msg.find("'") + 1
        end_pos = error_msg.find("'", start_pos)
        return error_msg[start_pos:end_pos]
