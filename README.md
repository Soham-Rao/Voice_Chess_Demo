# Wizard Chess

Run from this folder with the `techit` conda environment:

```powershell
$env:TCL_LIBRARY='C:\Users\soham\.conda\envs\techit\Library\lib\tcl8.6'; $env:TK_LIBRARY='C:\Users\soham\.conda\envs\techit\Library\lib\tk8.6'; conda run -n techit python app.py
```

This sets the Tcl/Tk paths that this Windows conda environment needs before opening the app.

Dependencies used by the app:

- `customtkinter`
- `Pillow`
- `python-chess`
- `SpeechRecognition`
- `PyAudio`
- `openai-whisper` + `torch` for offline Whisper transcription
- `openai` (optional, for API-backed speech transcription)

The app stays playable with clicks and typed commands if microphone recognition fails.

By default the wand uses offline Whisper with `tiny.en`, so no API key is needed. It automatically uses CUDA when PyTorch can see your GPU. To try a larger local model:

```powershell
$env:LOCAL_WHISPER_MODEL='base.en'
```

Force a device if needed:

```powershell
$env:LOCAL_WHISPER_DEVICE='cuda'
```

To use OpenAI API transcription instead of local Whisper:

```powershell
$env:USE_OPENAI_TRANSCRIBE='1'
$env:OPENAI_API_KEY='your_api_key_here'
```
