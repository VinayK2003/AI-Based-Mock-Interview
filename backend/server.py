from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import logging
import numpy as np
import struct
import io
from voice_rater import VoiceRatingAnalyzer

app = FastAPI()

# Configuration
FRONTEND_URL = "http://localhost:3000"
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleVoiceAnalyzer:
    """A simple voice analyzer using basic signal processing"""
    
    def analyze_audio(self, audio_data: bytes) -> dict:
        try:
            # Convert bytes to numpy array of floating point values
            # Assuming 16-bit audio
            audio_array = np.frombuffer(audio_data, dtype=np.int8)
            
            if len(audio_array) == 0:
                return {"error": "Empty audio data"}
            
            # Normalize the audio data to [-1, 1]
            audio_float = audio_array.astype(np.float32) / np.iinfo(np.int8).max
            
            # Calculate metrics
            metrics = self._calculate_metrics(audio_float)
            
            return {
                "status": "success",
                "metrics": metrics,
                "feedback": self._generate_feedback(metrics["overall_score"])
            }
            
        except Exception as e:
            logger.error(f"Error in analyze_audio: {str(e)}", exc_info=True)
            return {"error": f"Analysis failed: {str(e)}"}

    def _calculate_metrics(self, audio_data: np.ndarray) -> dict:
        """Calculate various audio metrics"""
        try:
            # Volume (RMS energy)
            rms = np.sqrt(np.mean(np.square(audio_data)))
            volume_score = min(100, rms * 100)

            # Clarity (peak-to-RMS ratio)
            peak = np.max(np.abs(audio_data))
            clarity_score = min(100, (peak / (rms + 1e-6)) * 50)

            # Rhythm (zero crossings rate)
            zero_crossings = np.sum(np.diff(np.signbit(audio_data).astype(int)))
            rhythm_score = min(100, zero_crossings / len(audio_data) * 1000)

            # Overall score
            overall_score = np.mean([volume_score, clarity_score, rhythm_score])

            return {
                "overall_score": round(float(overall_score), 2),
                "volume_score": round(float(volume_score), 2),
                "clarity_score": round(float(clarity_score), 2),
                "rhythm_score": round(float(rhythm_score), 2)
            }

        except Exception as e:
            logger.error(f"Error in _calculate_metrics: {str(e)}", exc_info=True)
            raise

    def _generate_feedback(self, score: float) -> str:
        """Generate feedback based on the overall score"""
        if score > 85:
            return "Excellent voice quality! Your speech is clear and well-modulated."
        elif score > 70:
            return "Good voice quality. Minor improvements could be made in clarity and modulation."
        elif score > 50:
            return "Average voice quality. Try speaking more clearly and varying your tone."
        else:
            return "Voice quality needs improvement. Focus on speaking clearly and maintaining consistent volume."

@app.post("/upload-audio/{userEmail}/{questionId}")
async def upload_audio(userEmail: str, questionId: int, audio: UploadFile):
    try:
        # Read the audio data
        contents = await audio.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Create analyzer and process audio
        analyzer = SimpleVoiceAnalyzer()
        result = analyzer.analyze_audio(contents)
        
        if "error" in result:
            logger.error(f"Analysis error: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
            
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process audio: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)