import base64
import os
import tempfile
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import requests
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from audioseal import AudioSeal

# === Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === FastAPI app ===
app = FastAPI(title="Audio Watermarking API", version="3.0.0")

# === Add CORS middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    logger.info("🔄 Loading AudioSeal models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"📱 Using device: {device}")
    
    try:
        watermark_model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
        detector_model = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
        
        # Set to eval mode
        watermark_model.eval()
        detector_model.eval()
        
        logger.info("✅ Models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise e

# === Helpers ===
def load_audio_from_url(url):
    try:
        # Simple download with reasonable timeout
        response = requests.get(url, timeout=60)  # Increased from 30s
        response.raise_for_status()
        
        # Use torchaudio for better format support
        audio_data = BytesIO(response.content)
        wav, sr = torchaudio.load(audio_data)
        
        return wav, sr
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load audio: {str(e)}")

def audio_to_base64_string(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to convert audio to base64: {str(e)}")

def preprocess_audio(wav, sr):
    try:
        # Ensure proper shape for AudioSeal: [batch, channels, samples]
        if wav.ndim == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.ndim == 2:
            if wav.shape[0] > wav.shape[1]:  # If channels > samples, transpose
                wav = wav.T
            wav = wav.unsqueeze(0)  # Add batch dimension

        # Convert to mono if stereo
        if wav.shape[1] > 1:
            wav = wav.mean(dim=1, keepdim=True)

        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            wav = resampler(wav)
            sr = 16000

        return wav.to(device), sr
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio preprocessing failed: {str(e)}")

# === Routes ===
@app.get("/")
async def root():
    return {"message": "Audio Watermarking API v3 is running!", "status": "healthy"}

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
        logger.info(f"🔄 Processing audio from URL: {request.audioUrl[:100]}...")
        logger.info(f"💬 Watermark message: {request.watermarkMessage}")

        # Load and preprocess audio
        wav, sr = load_audio_from_url(request.audioUrl)
        wav, sr = preprocess_audio(wav, sr)
        logger.info(f"📊 Audio shape: {wav.shape}, Sample rate: {sr}")

        # Handle watermark message
        if request.watermarkMessage and request.watermarkMessage.strip():
            binary_msg = request.watermarkMessage.strip()
            if not all(bit in '01' for bit in binary_msg) or len(binary_msg) != 16:
                raise HTTPException(status_code=400, detail="Watermark must be a 16-bit binary string (e.g., '0101010101010101')")
            
            message_tensor = torch.tensor([[int(bit) for bit in binary_msg]], dtype=torch.int32).to(device)
            logger.info(f"🔄 Using provided message: {binary_msg}")
        else:
            # Generate random message if none provided
            message_tensor = torch.randint(0, 2, (1, 16), dtype=torch.int32).to(device)
            binary_msg = ''.join(str(int(bit)) for bit in message_tensor.squeeze().cpu().numpy())
            logger.info(f"🔄 Generated random message: {binary_msg}")

        # Generate watermarked audio
        logger.info("🔄 Generating watermarked audio...")
        with torch.no_grad():
            watermarked_audio = watermark_model(wav, sample_rate=sr, message=message_tensor, alpha=1.0)

        # Save watermarked file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_np = watermarked_audio.squeeze().detach().cpu().numpy()
        sf.write(temp_file.name, audio_np, sr)
        logger.info(f"✅ Watermarked audio saved to: {temp_file.name}")

        # Verify watermark
        logger.info("🔄 Verifying watermark...")
        with torch.no_grad():
            results, detected_message = detector_model.detect_watermark(watermarked_audio, sample_rate=sr)

        # Check detection results
        if hasattr(results, 'mean'):
            confidence = results.mean().item()
            is_detected = torch.all(results > 0.5).item()
        else:
            confidence = float(results)
            is_detected = results > 0.5

        if is_detected:
            detected_bits = detected_message.squeeze().detach().cpu().numpy()
            detected_binary = ''.join(str(int(round(bit))) for bit in detected_bits[:16])

            logger.info(f"✅ Watermark detected: {detected_binary} (confidence: {confidence:.4f})")
            base64_audio = audio_to_base64_string(temp_file.name)

            return WatermarkResponse(
                status="success",
                base64_audio=base64_audio,
                decoded_message=detected_binary
            )
        else:
            logger.warning(f"❌ Watermark detection failed (confidence: {confidence:.4f})")
            return WatermarkResponse(
                status="failed",
                error=f"Watermark detection failed (confidence: {confidence:.4f})"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Cleanup
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
                logger.info(f"🧹 Cleaned up temporary file")
            except:
                pass
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.post("/detect-watermark")
async def detect_watermark_from_url(request: WatermarkRequest):
    if not detector_model:
        raise HTTPException(status_code=503, detail="Detector model not loaded.")

    try:
        logger.info(f"🔄 Detecting watermark in audio from URL: {request.audioUrl[:100]}...")
        
        # Load and preprocess audio
        wav, sr = load_audio_from_url(request.audioUrl)
        wav, sr = preprocess_audio(wav, sr)

        # Detect watermark
        logger.info("🔄 Detecting watermark...")
        with torch.no_grad():
            results, message_tensor = detector_model.detect_watermark(wav, sample_rate=sr)

        # Process results
        if hasattr(results, 'mean'):
            confidence = results.mean().item()
            is_detected = torch.all(results > 0.5).item()
        else:
            confidence = float(results)
            is_detected = results > 0.5

        if is_detected:
            detected_bits = message_tensor.squeeze().detach().cpu().numpy()
            binary_string = ''.join(str(int(round(bit))) for bit in detected_bits[:16])

            logger.info(f"✅ Watermark detected! Binary: {binary_string} (confidence: {confidence:.4f})")
            
            return {
                "status": "done",
                "watermark_detected": True,
                "confidence": float(confidence),
                "decoded_message": binary_string
            }
        else:
            logger.info(f"❌ No watermark detected (confidence: {confidence:.4f})")
            return {
                "status": "done",
                "watermark_detected": False,
                "confidence": float(confidence),
                "decoded_message": None
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error detecting watermark: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# === Run locally ===
if __name__ == "__main__":
    logger.info("🚀 Starting Audio Watermarking API v3...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info"
    )