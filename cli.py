import os
import torch
import pytesseract
import warnings
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from colorama import Fore, init
from termcolor import colored
import pyfiglet
from pydub import AudioSegment
import torchaudio

warnings.filterwarnings("ignore")
init(autoreset=True)

loaded_mt_models = {}
loaded_asr_model = None
loaded_asr_processor = None
DEVICE = None
ASR_ENGINE = None

LANG_CODE_MAP = {
    "english": "en", "russian": "ru", "chinese": "zh", "french": "fr", "german": "de",
    "polish": "pl", "spanish": "es", "italian": "it", "japanese": "ja",
    "korean": "ko", "arabic": "ar", "hindi": "hi"
}

def get_lang_code(lang_name):
    code = LANG_CODE_MAP.get(lang_name.lower())
    if code:
        return code
    return lang_name[:2].lower()

def banner():
    print(colored(pyfiglet.figlet_format("Local Translator Pro", font="slant"), "cyan"))
    print(Fore.GREEN + "🌍 Made by Aryan Giri | 100% Offline | 0% Cloud | Full Privacy\n")

def select_device():
    global DEVICE
    print(Fore.MAGENTA + "\n⚙️  System Setup")
    choice = input(Fore.CYAN + "Do you want to connect with GPU for faster translation? (y/n): ").strip().lower()
    if choice == "y":
        if torch.cuda.is_available():
            DEVICE = "cuda"
            print(Fore.GREEN + "✅ GPU mode enabled (CUDA detected).")
        elif torch.backends.mps.is_available():
            DEVICE = "mps"
            print(Fore.GREEN + "✅ GPU mode enabled (Apple Metal detected).")
        else:
            DEVICE = "cpu"
            print(Fore.YELLOW + "⚠️ No GPU detected. Using CPU instead.")
    else:
        DEVICE = "cpu"
        print(Fore.YELLOW + "🧠 CPU mode selected (privacy-safe & portable).")

def select_asr_engine():
    global ASR_ENGINE
    print(Fore.MAGENTA + "\n🎙️ Choose Audio Transcription Engine")
    print("1️⃣ Whisper (high accuracy, heavier, GPU recommended)")
    print("2️⃣ SpeechRecognition + pocketsphinx (lighter, faster, less accurate)")
    choice = input(Fore.CYAN + "Enter choice (1 or 2): ").strip()
    if choice == "1":
        ASR_ENGINE = "whisper"
        print(Fore.GREEN + "✅ Whisper selected.")
    elif choice == "2":
        ASR_ENGINE = "pocketsphinx"
        print(Fore.GREEN + "✅ SpeechRecognition + pocketsphinx selected.")
    else:
        ASR_ENGINE = "whisper"
        print(Fore.YELLOW + "⚠️ Invalid choice. Defaulting to Whisper.")

def load_translation_model(src_lang, tgt_lang):
    lang_pair = f"{src_lang}-{tgt_lang}"
    if lang_pair in loaded_mt_models:
        return loaded_mt_models[lang_pair]

    model_name = f"Helsinki-NLP/opus-mt-{lang_pair}"
    print(Fore.YELLOW + f"\n📦 Loading translation model: {model_name} ...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        loaded_mt_models[lang_pair] = (tokenizer, model)
        print(Fore.GREEN + "✅ Translation model loaded.\n")
        return tokenizer, model
    except Exception as e:
        print(Fore.RED + f"❌ Model not found for {src_lang}-{tgt_lang}. Try another language pair.")
        raise e

def translate_text(text, src_lang, tgt_lang):
    try:
        tokenizer, model = load_translation_model(src_lang, tgt_lang)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        return f"❌ Translation Error: {str(e)}"

def translate_image(image_path, src_lang, tgt_lang):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        print(Fore.CYAN + f"\n📝 Extracted text:\n{text.strip()}\n")
        return translate_text(text, src_lang, tgt_lang)
    except Exception as e:
        return f"❌ Image Error: {str(e)}"

def convert_to_wav(mp3_path):
    audio = AudioSegment.from_file(mp3_path, format="mp3")
    wav_path = "temp_audio.wav"
    audio.export(wav_path, format="wav")
    return wav_path

def whisper_transcribe(mp3_path):
    global loaded_asr_model, loaded_asr_processor
    if not loaded_asr_model or not loaded_asr_processor:
        print(Fore.YELLOW + "\n🎧 Loading Whisper small model...")
        model_name = "openai/whisper-small"
        loaded_asr_processor = WhisperProcessor.from_pretrained(model_name)
        loaded_asr_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)
        print(Fore.GREEN + "✅ Whisper loaded.\n")

    wav_path = convert_to_wav(mp3_path)
    speech, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)
    os.remove(wav_path)

    input_features = loaded_asr_processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    predicted_ids = loaded_asr_model.generate(input_features)
    transcription = loaded_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    detected_lang_code = "en"  # fallback
    return transcription, detected_lang_code

def pocketsphinx_transcribe(mp3_path):
    import speech_recognition as sr
    wav_path = convert_to_wav(mp3_path)
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)
    os.remove(wav_path)
    try:
        return r.recognize_sphinx(audio_data), "en"
    except Exception:
        return "❌ Could not transcribe audio.", "en"

def translate_audio(mp3_path, target_lang):
    if ASR_ENGINE == "whisper":
        transcription, src_lang = whisper_transcribe(mp3_path)
    else:
        transcription, src_lang = pocketsphinx_transcribe(mp3_path)
    print(Fore.CYAN + f"\n🎙️ Transcribed text:\n{transcription}\n")
    return translate_text(transcription, src_lang, target_lang)

def search_and_install_language(lang_name):
    lang_code = get_lang_code(lang_name)
    print(Fore.YELLOW + f"\n🔍 Searching for MarianMT model for {lang_name} ({lang_code}) ...")
    try:
        # English -> target
        model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        loaded_mt_models[f"en-{lang_code}"] = (tokenizer, model)
        # Target -> English
        model_name_rev = f"Helsinki-NLP/opus-mt-{lang_code}-en"
        tokenizer_rev = MarianTokenizer.from_pretrained(model_name_rev)
        model_rev = MarianMTModel.from_pretrained(model_name_rev).to(DEVICE)
        loaded_mt_models[f"{lang_code}-en"] = (tokenizer_rev, model_rev)
        print(Fore.GREEN + f"✅ {lang_name} models installed (forward & reverse).")
        return lang_code
    except Exception:
        print(Fore.RED + f"❌ No MarianMT model found for {lang_name}.")
        return None

def main():
    banner()
    select_device()
    select_asr_engine()
    while True:
        print(Fore.MAGENTA + "\n📘 Choose Option:")
        print("1️⃣ Text Translation")
        print("2️⃣ Image Translation (OCR)")
        print("3️⃣ Audio Translation (MP3)")
        print("4️⃣ Exit")
        print("5️⃣ Search & Install New Language")

        choice = input(Fore.YELLOW + "\nEnter choice (1-5): ").strip()
        if choice == "4":
            print(Fore.CYAN + "\n👋 Exiting Translator Pro. Stay private, stay legendary.")
            break

        elif choice == "1":
            src_name = input(Fore.CYAN + "\nEnter source language (English): ").strip()
            tgt_name = input(Fore.CYAN + "Enter target language (e.g., Polish): ").strip()
            src_lang = get_lang_code(src_name)
            tgt_lang = get_lang_code(tgt_name)
            text = input(Fore.WHITE + "\nEnter text: ").strip()
            print(Fore.GREEN + f"\n✅ Translation:\n{translate_text(text, src_lang, tgt_lang)}\n")

        elif choice == "2":
            img_path = input(Fore.WHITE + "\nEnter image path: ").strip()
            src_name = input(Fore.CYAN + "Enter image text language: ").strip()
            tgt_name = input(Fore.CYAN + "Enter target language: ").strip()
            src_lang = get_lang_code(src_name)
            tgt_lang = get_lang_code(tgt_name)
            print(Fore.GREEN + f"\n✅ Translation:\n{translate_image(img_path, src_lang, tgt_lang)}\n")

        elif choice == "3":
            mp3_path = input(Fore.WHITE + "\nEnter MP3 path: ").strip()
            tgt_name = input(Fore.CYAN + "Enter target language name in English (e.g., Polish): ").strip()
            tgt_lang = get_lang_code(tgt_name)
            print(Fore.GREEN + f"\n✅ Translation:\n{translate_audio(mp3_path, tgt_lang)}\n")

        elif choice == "5":
            lang_name = input(Fore.CYAN + "\nEnter language name to search/install (e.g., Finnish): ").strip()
            installed_code = search_and_install_language(lang_name)
            if installed_code:
                print(Fore.GREEN + f"\n✅ {lang_name} is ready for translation!\n")

        else:
            print(Fore.RED + "❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
