from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import numpy as np
import io
import librosa
from pydub import AudioSegment  # Requires: pip install pydub and ffmpeg installed
from voice_rater import VoiceRatingAnalyzer  # Ensure this is defined or remove if not used

app = FastAPI()

# Configuration
FRONTEND_URL = "http://localhost:3000"

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
    """A simple voice analyzer using basic signal processing."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def load_audio(self, audio_bytes: bytes) -> (np.ndarray, int):
        """
        Convert the input WebM audio (as bytes) to a WAV waveform using PyDub
        and load it with librosa.
        """
        try:
            audio_stream = io.BytesIO(audio_bytes)
            # Load the WebM file using PyDub (ensure the file is indeed in WebM format)
            audio_segment = AudioSegment.from_file(audio_stream, format="webm")
            # Export to WAV format in-memory
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            # Load the WAV data using librosa
            audio, sr = librosa.load(wav_io, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            logger.error(f"Error in load_audio: {str(e)}", exc_info=True)
            raise

    def analyze_audio(self, audio_data: bytes) -> dict:
        """
        Perform basic analysis (volume, clarity, rhythm) on the raw bytes.
        (This method assumes a raw PCM interpretation; adjust dtype if needed.)
        """
        try:
            # Convert bytes to numpy array of 8-bit integers.
            audio_array = np.frombuffer(audio_data, dtype=np.int8)
            if len(audio_array) == 0:
                return {"error": "Empty audio data"}
            
            # Normalize the audio data to [-1, 1]
            audio_float = audio_array.astype(np.float32) / np.iinfo(np.int8).max
            
            # Calculate basic metrics
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
        """Calculate various basic audio metrics (volume, clarity, rhythm)."""
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

    def analyze_pitch(self, audio_bytes: bytes) -> float:
        """
        Analyze pitch characteristics of the audio using librosa.yin.
        Computes the fundamental frequency (F0) over time and calculates
        a stability score based on the coefficient of variation.
        """
        try:
            audio, sr = self.load_audio(audio_bytes)
            # Set minimum and maximum frequencies for pitch detection.
            fmin = 50   # Adjust as needed for your voice recordings.
            fmax = 300
            # Use librosa.yin to estimate the fundamental frequency over time.
            pitch_values = librosa.yin(audio, fmin=fmin, fmax=fmax, sr=sr)
            valid_pitches = pitch_values[pitch_values > 0]
            if valid_pitches.size == 0:
                return 0.0
            pitch_mean = np.mean(valid_pitches)
            pitch_std = np.std(valid_pitches)
            # Compute stability as a percentage: lower variation yields higher score.
            stability_score = max(0, 100 - (pitch_std / pitch_mean * 100))
            return stability_score
        except Exception as e:
            logger.error(f"Error in analyze_pitch: {str(e)}", exc_info=True)
            return 0.0

    def analyze_tone(self, audio_wave: np.ndarray) -> float:
        """Analyze tone quality based on spectral characteristics."""
        try:
            stft = librosa.stft(audio_wave)
            db = librosa.amplitude_to_db(np.abs(stft))
            tone_score = min(100, max(0, np.mean(db) + 100))
            return tone_score
        except Exception as e:
            logger.error(f"Error in analyze_tone: {str(e)}", exc_info=True)
            return 0.0

    def analyze_frequency_variation(self, audio_wave: np.ndarray) -> float:
        """Analyze frequency variation."""
        try:
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_wave, sr=self.sample_rate)[0]
            variation_score = min(100, max(0, 100 - (np.std(spectral_centroids) * 0.1)))
            return variation_score
        except Exception as e:
            logger.error(f"Error in analyze_frequency_variation: {str(e)}", exc_info=True)
            return 0.0

    def analyze_pauses(self, audio_wave: np.ndarray) -> float:
        """Analyze speech pauses based on RMS energy."""
        try:
            rms = librosa.feature.rms(y=audio_wave)[0]
            silence_threshold = np.mean(rms) * 0.5
            pauses = np.sum(rms < silence_threshold) / len(rms)
            pause_score = min(100, max(0, 100 - abs(pauses - 0.2) * 200))
            return pause_score
        except Exception as e:
            logger.error(f"Error in analyze_pauses: {str(e)}", exc_info=True)
            return 0.0

    def analyze_additional_features(self, audio_bytes: bytes) -> dict:
        """
        Analyze additional features (tone, frequency variation, and pauses)
        from the audio waveform.
        """
        try:
            audio_wave, sr = self.load_audio(audio_bytes)
            tone_score = self.analyze_tone(audio_wave)
            freq_variation = self.analyze_frequency_variation(audio_wave)
            pause_score = self.analyze_pauses(audio_wave)
            return {
                "tone_score": round(float(tone_score), 2),
                "frequency_variation": round(float(freq_variation), 2),
                "pause_score": round(float(pause_score), 2)
            }
        except Exception as e:
            logger.error(f"Error in analyze_additional_features: {str(e)}", exc_info=True)
            return {
                "tone_score": 0.0,
                "frequency_variation": 0.0,
                "pause_score": 0.0
            }

    def _generate_feedback(self, score: float) -> str:
        """Generate feedback based on the overall score."""
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
        # Read the audio data from the uploaded file (in-memory, no file saving)
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Create analyzer and process audio
        analyzer = SimpleVoiceAnalyzer()
        
        # Basic analysis (volume, clarity, rhythm)
        result = analyzer.analyze_audio(contents)
        
        # Compute pitch score using the updated method
        pitch_score = analyzer.analyze_pitch(contents)
        
        # Compute additional features (tone, frequency variation, pauses)
        additional = analyzer.analyze_additional_features(contents)
        
        # Merge all scores into the metrics dictionary
        result["metrics"]["pitch_score"] = round(float(pitch_score), 2)
        result["metrics"]["tone_score"] = additional["tone_score"]
        result["metrics"]["frequency_variation"] = additional["frequency_variation"]
        result["metrics"]["pause_score"] = additional["pause_score"]
            
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
