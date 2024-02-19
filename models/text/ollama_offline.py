import json

import requests


class LanguageDetectionOllamaAPI:
    def __init__(self, api_url="http://localhost:11434/api/generate"):
        self.api_url = api_url

    def detect_language(self, prompt):
        LI_prompt = "Identify only the lang code in:" + prompt

        data = {
            "model": "llama2",
            "prompt": LI_prompt,
            "stream": False,
            "format": "json",
        }

        response = requests.post(self.api_url, json=data)

        if response.status_code == 200:
            try:
                response_data = json.loads(
                    response.json().get("response", "{}")
                )
                lang_value = next(
                    (
                        value
                        for key, value in response_data.items()
                        if key.startswith("lang")
                    ),
                    None,
                )

                if lang_value is None:
                    print(
                        f"{prompt} Error: 'lang' key not found in the response.\n{response_data}\n"
                    )
                return lang_value
            except json.JSONDecodeError:
                print("Error decoding JSON:", response.text)
                return "Error (JSON Decode)"
        else:
            print("Error:", response.text)
            return "Error"
