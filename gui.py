import sys, os
import torch
import pytesseract
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QComboBox, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from pydub import AudioSegment
import torchaudio
from datetime import datetime

loaded_mt_models = {}
loaded_asr_model = None
loaded_asr_processor = None
DEVICE = "cpu"
ASR_ENGINE = "whisper"
LANG_CODE_MAP = {
    "english": "en", "russian": "ru", "chinese": "zh", "french": "fr", "german": "de",
    "polish": "pl", "spanish": "es", "italian": "it", "japanese": "ja",
    "korean": "ko", "arabic": "ar", "hindi": "hi"
}

HISTORY_FILE = "translation_history.txt"

def get_lang_code(lang_name):
    code = LANG_CODE_MAP.get(lang_name.lower())
    if code:
        return code
    return lang_name[:2].lower()

def load_translation_model(src_lang, tgt_lang):
    lang_pair = f"{src_lang}-{tgt_lang}"
    if lang_pair in loaded_mt_models:
        return loaded_mt_models[lang_pair]

    model_name = f"Helsinki-NLP/opus-mt-{lang_pair}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        loaded_mt_models[lang_pair] = (tokenizer, model)
        return tokenizer, model
    except Exception:
        return None, None

def translate_text(text, src_lang, tgt_lang):
    tokenizer, model = load_translation_model(src_lang, tgt_lang)
    if not tokenizer:
        return "❌ Model not found for this language pair."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    translated = model.generate(**inputs)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)
    save_translation_to_history(text, src_lang, tgt_lang, result)
    return result

def translate_image(image_path, src_lang, tgt_lang):
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        result = translate_text(text, src_lang, tgt_lang)
        return result
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
        model_name = "openai/whisper-small"
        loaded_asr_processor = WhisperProcessor.from_pretrained(model_name)
        loaded_asr_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(DEVICE)

    wav_path = convert_to_wav(mp3_path)
    speech, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        speech = resampler(speech)
    os.remove(wav_path)

    input_features = loaded_asr_processor(speech.squeeze(), sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
    predicted_ids = loaded_asr_model.generate(input_features)
    transcription = loaded_asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def pocketsphinx_transcribe(mp3_path):
    import speech_recognition as sr
    wav_path = convert_to_wav(mp3_path)
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = r.record(source)
    os.remove(wav_path)
    try:
        return r.recognize_sphinx(audio_data)
    except:
        return "❌ Could not transcribe audio."

def translate_audio(mp3_path, tgt_lang):
    if ASR_ENGINE == "whisper":
        transcription = whisper_transcribe(mp3_path)
    else:
        transcription = pocketsphinx_transcribe(mp3_path)
    result = translate_text(transcription, "en", tgt_lang)
    return result

def search_and_install_language(lang_name):
    lang_code = get_lang_code(lang_name)
    try:
        model_name = f"Helsinki-NLP/opus-mt-en-{lang_code}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(DEVICE)
        loaded_mt_models[f"en-{lang_code}"] = (tokenizer, model)
        model_name_rev = f"Helsinki-NLP/opus-mt-{lang_code}-en"
        tokenizer_rev = MarianTokenizer.from_pretrained(model_name_rev)
        model_rev = MarianMTModel.from_pretrained(model_name_rev).to(DEVICE)
        loaded_mt_models[f"{lang_code}-en"] = (tokenizer_rev, model_rev)
        return True
    except Exception:
        return False

def save_translation_to_history(original, src_lang, tgt_lang, translated):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {src_lang} -> {tgt_lang}\nOriginal: {original}\nTranslated: {translated}\n\n")

def read_translation_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def clear_translation_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

class TranslatorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Translator Pro | Made by Aryan Giri")
        self.setGeometry(100, 100, 850, 700)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.src_lang_input = QLineEdit()
        self.src_lang_input.setPlaceholderText("Source Language (English)")
        self.tgt_lang_input = QLineEdit()
        self.tgt_lang_input.setPlaceholderText("Target Language (Polish)")

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text to translate...")

        self.translate_button = QPushButton("Translate Text")
        self.translate_button.clicked.connect(self.translate_text_action)

        self.result_output = QTextEdit()
        self.result_output.setReadOnly(True)

        self.image_button = QPushButton("Select Image for OCR Translation")
        self.image_button.clicked.connect(self.translate_image_action)

        self.audio_button = QPushButton("Select MP3 for Translation")
        self.audio_button.clicked.connect(self.translate_audio_action)

        self.install_lang_button = QPushButton("Search & Install New Language")
        self.install_lang_button.clicked.connect(self.install_language_action)

        self.history_button = QPushButton("Show Translation History")
        self.history_button.clicked.connect(self.show_history)

        self.clear_history_button = QPushButton("Clear Translation History")
        self.clear_history_button.clicked.connect(self.clear_history)

        self.asr_engine_combo = QComboBox()
        self.asr_engine_combo.addItems(["Whisper (accurate)", "Pocketsphinx (light)"])
        self.asr_engine_combo.currentIndexChanged.connect(self.select_asr_engine)

        layout.addWidget(QLabel("Source Language:"))
        layout.addWidget(self.src_lang_input)
        layout.addWidget(QLabel("Target Language:"))
        layout.addWidget(self.tgt_lang_input)
        layout.addWidget(QLabel("Text Translation:"))
        layout.addWidget(self.text_input)
        layout.addWidget(self.translate_button)
        layout.addWidget(QLabel("Result:"))
        layout.addWidget(self.result_output)
        layout.addWidget(self.image_button)
        layout.addWidget(self.audio_button)
        layout.addWidget(self.install_lang_button)
        layout.addWidget(self.history_button)
        layout.addWidget(self.clear_history_button)
        layout.addWidget(QLabel("ASR Engine:"))
        layout.addWidget(self.asr_engine_combo)

        self.setLayout(layout)

    def select_asr_engine(self, index):
        global ASR_ENGINE
        ASR_ENGINE = "whisper" if index == 0 else "pocketsphinx"

    def translate_text_action(self):
        src = get_lang_code(self.src_lang_input.text())
        tgt = get_lang_code(self.tgt_lang_input.text())
        text = self.text_input.toPlainText()
        if not text:
            QMessageBox.warning(self, "Warning", "Enter some text to translate.")
            return
        result = translate_text(text, src, tgt)
        self.result_output.setText(result)

    def translate_image_action(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image")
        if not file_path:
            return
        src = get_lang_code(self.src_lang_input.text())
        tgt = get_lang_code(self.tgt_lang_input.text())
        result = translate_image(file_path, src, tgt)
        self.result_output.setText(result)

    def translate_audio_action(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select MP3", filter="MP3 Files (*.mp3)")
        if not file_path:
            return
        tgt = get_lang_code(self.tgt_lang_input.text())
        result = translate_audio(file_path, tgt)
        self.result_output.setText(result)

    def install_language_action(self):
        lang_name, ok = QFileDialog.getText(self, "Search & Install Language", "Enter language name (English):")
        if not ok or not lang_name:
            return
        success = search_and_install_language(lang_name)
        if success:
            QMessageBox.information(self, "Success", f"✅ {lang_name} models installed (forward & reverse).")
        else:
            QMessageBox.warning(self, "Error", f"❌ Could not find model for {lang_name}.")

    def show_history(self):
        history = read_translation_history()
        if not history:
            history = "No translations yet."
        QMessageBox.information(self, "Translation History", history)

    def clear_history(self):
        clear_translation_history()
        QMessageBox.information(self, "Translation History", "✅ Translation history cleared.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = TranslatorGUI()
    gui.show()
    sys.exit(app.exec())
