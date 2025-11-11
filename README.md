# Local-Translator-Pro

## Overview

Offline Translator Pro is a fully offline, privacy-focused translator that can translate:

Text

Images (OCR)

Audio (MP3)


## Features:

Text translation between multiple languages

Image OCR and translation

Audio transcription and translation (Whisper or Pocketsphinx)

Dynamic language model download

Translation history saved locally

CLI and GUI support

Fully offline, no cloud storage



---

## Installation

1. Clone the repository or download the source code.


2. Run the installer:
```
python setup.py
```
3. Follow prompts to:

Create a virtual environment (recommended)

Install system packages (Tesseract OCR, FFmpeg, build tools)

Choose CPU or GPU PyTorch installation

Install Python dependencies



4. Activate your virtual environment (if created):



Linux/macOS:

```
source venv/bin/activate
```
Windows:

```
venv\Scripts\activate
```

---

## Usage

CLI
```
python cli.py
```
GUI
```
python gui.py
```

---

CLI Options

Translate text interactively

Translate images via OCR

Translate audio files (MP3)

Dynamic language download

Translation history

Select ASR engine: Whisper (accurate) or Pocketsphinx (light)



---

GUI Features

Text, Image, and Audio translation

ASR engine selection

Search & install new language models

Translation history viewer and clear history

Fully offline and privacy-focused



---

Supported Languages

Default languages include:

English

Russian

Chinese

French

German

Polish

Spanish

Italian

Japanese

Korean

Arabic

Hindi


You can search and install additional languages dynamically.


---

Notes

Ensure Tesseract OCR and FFmpeg are installed on your system.

Whisper ASR provides higher accuracy but requires more resources.

Pocketsphinx is lightweight but less accurate.

PyTorch GPU installation depends on your CUDA drivers; CPU fallback is available.

Translation history is saved locally in translation_history.txt.



---

Credits

Aryan Giri — Developer & Maintainer

Hugging Face Transformers, Whisper models, MarianMT models

PyQt6, Pytesseract, Pydub, SpeechRecognition



---

License

MIT License

