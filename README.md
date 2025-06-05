# 🎙️ auralislib

> Intelligent Python Library for Voice Cloning, Audio Synthesis, and Image Generation

# auralislib

![PyPI](https://img.shields.io/pypi/v/auralislib))
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
[![Downloads](https://img.shields.io/pypi/dm/auralislib.svg?label=Downloads&color=green)]
![License](https://img.shields.io/github/license/joshirugved11/auralis)
![Contributors](https://img.shields.io/github/contributors/joshirugved11/auralis)
![Last Commit](https://img.shields.io/github/last-commit/joshirugved11/auralis)
![Issues](https://img.shields.io/github/issues/joshirugved11/auralis)
![Pull Requests](https://img.shields.io/github/issues-pr/joshirugved11/auralis)
![Stars](https://img.shields.io/github/stars/joshirugved11/auralis?style=social)

---

## 🚀 What is auralislib?

**auralislib** is a Python library designed to make working with voice, audio, and image data creative and intelligent. It enables:
- Voice cloning & synthesis  
- Text-to-speech (TTS)  
- Speech-to-text (STT)  
- Image fetching & transformation  
- Multimodal features (text → image → audio) (upcoming features)
---

## 🛠️ Features

- ✅ Generate synthetic speech from text and text from speech
- ✅ Clone voices from audio samples (upcoming features)
- ✅ Fetch and process images using the Unsplash API (upcoming features)
- ✅ Basic image editing (resize, crop, filters) (upcoming features)
- ✅ Plan for future AI-powered voice & image fusion (upcoming features)
---

## 📦 Installation

<pre lang="markdown"> pip install auralislib </pre>
---

## Usage
1. Text-to-speech
<pre lang="markdown"> 
  from auralislib.TTS.synth import synthesize
  synthesize("Hello world!", "output.wav")
</pre>

2. Speech-to-text
<pre lang="markdown"> 
  from auralislib.STT.inference import transcribe
  text = transcribe("samples/audio.wav")
  print(text)
</pre>
---



