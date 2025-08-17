import base64
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from audioseal import AudioSeal

# === FastAPI app ===
app = FastAPI(title="Audio Watermarking API", version="1.0.0")

# === Models ===
class WatermarkRequest(BaseModel):
    audioUrl: str
    watermarkMessage: Optional[str] = ""  # Must be 16-bit binary string

class WatermarkResponse(BaseModel):
    status: str
    base64_audio: Optional[str] = None
    decoded_message: Optional[str] = None
    error: Optional[str] = None

# === Globals ===
watermark_model = None
detector_model = None
device = None

# === Startup ===
@app.on_event("startup")
async def load_models():
    global watermark_model, detector_model, device
    print("🔄 Loading AudioSeal models...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    try:
        watermark_model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
        detector_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise e

# === Helpers ===
def load_audio_from_url(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return sf.read(BytesIO(response.content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")

def audio_to_base64_string(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert audio to base64: {str(e)}")

def preprocess_audio(data, sr):
    try:
        if data.ndim == 1:
            data = data[None, None, :]
        else:
            data = data.T
            data = data[None]

        wav = torch.tensor(data, dtype=torch.float32)

        if wav.shape[1] > 1:
            wav = wav.mean(dim=1, keepdim=True)

        if sr != 16000:
            import torchaudio
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
            sr = 16000

        return wav.to(device), sr
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {str(e)}")

# === Routes ===
@app.get("/")
async def root():
    return {"message": "Audio Watermarking API is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": watermark_model is not None and detector_model is not None,
        "device": str(device) if device else "not set"
    }

@app.post("/add-watermark-url", response_model=WatermarkResponse)
async def add_watermark_from_url(request: WatermarkRequest):
    if not watermark_model or not detector_model:
        raise HTTPException(status_code=503, detail="Models not loaded. Please try again later.")

    temp_file = None

    try:
        print(f"🔄 Processing audio from URL: {request.audioUrl}")
        print(f"💬 Watermark message: {request.watermarkMessage}")

        data, sr = load_audio_from_url(request.audioUrl)
        wav, sr = preprocess_audio(data, sr)

        # === Embed 16-bit binary string
        binary_msg = request.watermarkMessage.strip()
        if not all(bit in '01' for bit in binary_msg) or len(binary_msg) != 16:
            raise HTTPException(status_code=400, detail="Watermark must be a 16-bit binary string (e.g., '0101010101010101')")

        bit_tensor = torch.tensor([[int(bit) for bit in binary_msg]], dtype=torch.float32).to(device)
        print("🔄 Generating watermark...")
        watermark = watermark_model.get_watermark(wav, sr, bit_tensor)
        watermarked_audio = wav + watermark

        # === Save watermarked file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(temp_file.name, watermarked_audio.squeeze().detach().cpu().numpy(), sr)
        print(f"✅ Watermarked audio saved to: {temp_file.name}")

        # === Detect to verify
        print("🔄 Verifying watermark...")
        result, message_tensor = detector_model.detect_watermark(watermarked_audio, sr)

        if result:
            bit_list = message_tensor.flatten().tolist()
            binary_string = ''.join(str(int(round(bit))) for bit in bit_list[:16])

            print(f"✅ Watermark detected: {binary_string}")
            base64_audio = audio_to_base64_string(temp_file.name)

            return WatermarkResponse(
                status="success",
                base64_audio=base64_audio,
                decoded_message=binary_string
            )
        else:
            print("❌ Watermark detection failed.")
            return WatermarkResponse(
                status="failed",
                error="Watermark detection failed after embedding"
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                print(f"🧹 Cleaned up temporary file: {temp_file.name}")
            except:
                pass

@app.post("/detect-watermark")
async def detect_watermark_from_url(request: WatermarkRequest):
    if not detector_model:
        raise HTTPException(status_code=503, detail="Detector model not loaded. Please try again later.")

    try:
        print(f"🔄 Detecting watermark in audio from URL: {request.audioUrl}")
        data, sr = load_audio_from_url(request.audioUrl)
        wav, sr = preprocess_audio(data, sr)

        print("🔄 Detecting watermark...")
        result, message_tensor = detector_model.detect_watermark(wav, sr)

        if result:
            bit_list = message_tensor.flatten().tolist()
            binary_string = ''.join(str(int(round(bit))) for bit in bit_list[:16])

            print(f"✅ Watermark detected! Binary: {binary_string}")
            return {
                "status": "done",
                "watermark_detected": True,
                "confidence": float(result),
                "decoded_message": binary_string
            }
        else:
            print("❌ No watermark detected.")
            return {
                "status": "done",
                "watermark_detected": False,
                "confidence": 0.0,
                "decoded_message": None
            }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error detecting watermark: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# === Run locally ===
if __name__ == "__main__":
    print("🚀 Starting Audio Watermarking API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )