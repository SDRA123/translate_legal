import runpod
import torch
import re
from transformers import pipeline

# --- Configuration ---
MODEL_NAME = "shehryaraijaz/m2m100-legal-translation-en-ur"
DEVICE = 0 if torch.cuda.is_available() else -1

# --- Load Model ---
try:
    translator = pipeline(
        "translation",
        model=MODEL_NAME,
        src_lang="en",
        tgt_lang="ur",
        device=DEVICE
    )
    print(f"Model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    translator = None
    print(f"Error loading translation model: {e}")


# --- Helper Functions ---
def simple_sentence_splitter(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())


def translate_paragraph(text, translator_pipeline):
    if translator_pipeline is None:
        return "[Translation model unavailable]"

    sentences = simple_sentence_splitter(text.strip())
    translated_sentences = []

    for i, sentence in enumerate(sentences):
        try:
            result = translator_pipeline(sentence)
            translated_text = result[0]['translation_text']
            translated_sentences.append(translated_text)
        except Exception as e:
            translated_sentences.append(f"[Error translating sentence {i+1}]")

    return " ".join(translated_sentences)


# --- RunPod Handler ---
def handler(job):
    try:
        job_input = job["input"]
        english_text = job_input.get("prompt", "").strip()

        if not english_text:
            return {"error": "Missing 'text' in input."}

        translated_text = translate_paragraph(english_text, translator)

        return translated_text

    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless handler
runpod.serverless.start({"handler": handler})
