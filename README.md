# VersaVox
VersaVox, a flexible language detection tool, handles audio/text inputs using top ASR models. It operates in online/offline modes and provides free/premium options. Suitable for transcription and language detection applications.

This tool is designed to accurately detect and identify the language of
a given text. It utilizes advanced machine learning algorithms to
provide highly accurate results, supporting a wide range of languages.
This tool can be beneficial for various applications such as translation
services, multilingual content management, language tutoring, and more.

The Language Detection Tool is easy to use and requires minimal setup.
It has a user-friendly interface and provides results quickly and
efficiently.

This tool is WIP, open source and contributions to improve its
functionality or expand its features are always welcome. Please confirm
to the standard contribution process and do refer some of the open
source contribution guidelines for some motivation and guidance.

## Installation:

Please install the required pip packages using the command:

pip install -r requirements.txt

## Usage:

main.py --data-type `<type>` --data-path `<path>` --model
`<model>` --mode `<mode>`

## Arguments:
- `--data-type`: Specify 'text' or 'audio'
- `--data-path`: Path to the text file or audio directory/file.
- `--model`: Specify the model for language detection.
  - Text: gpt3.5, gpt4, llama2
  - Audio: google_asr, whisper
- `--mode`: Specify the mode, 'offline' or 'online'


## Running Instructions:
The program is run using main.py and requires several arguments to function correctly.

Here is a breakdown of the arguments:

`--data-type` This argument is used to specify the type of data that you
are working with. It can either be 'text' or 'audio'.

`--data-path` This argument is used to specify the path to the text file
or audio directory/file that you want to use.

`--model` This argument is used to specify the model for language detection.
- If you are working with text, you can choose between 'gpt3.5', 'gpt4', and 'llama2'.
- If you are working with audio, you can choose between 'google_asr' and 'whisper'.

`--mode` This argument is used to specify the mode in which you want to
operate. It can either be 'offline' or 'online'.

For example, if you have a text file located at /home/user/documents/text.txt and you want to use the 'llama2' model in 'offline' mode, you would run the following command:

For text: 
main.py --data-type text --data-path /home/user/documents/text.txt --model llama2 --mode offline

For audio: 
main.py --data-type audio --data-path /home/user/documents/audio_samples --model whisper --mode offline
main.py --data-type audio --data-path /home/user/documents/audio_samples --model google_ars --mode online

Before running the script, set the API Key environment variables,
preferably in .rc files of your choice, based on the online models
passed as parameters to the script.

Set the environment variables:

- `export GOOGLE_APPLICATION_CREDENTIALS=<FILE_PATH>` //Google ASR
- `export OPENAI_API_KEY=<KEY>` //OpenAI


This script can also be used to leverage local models that you can setup
to run you local device using, please refer to official document:
- [Ollama](https://ollama.com)
- [Huggingface](https://huggingface.co)
