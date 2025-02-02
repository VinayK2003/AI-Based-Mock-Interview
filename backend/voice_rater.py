from typing import Dict
import numpy as np
import librosa

class VoiceRatingAnalyzer:
    """
    A class for analyzing voice recordings and providing feedback.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def analyze_voice(self, file_path: str) -> Dict:
        """ Analyze the voice file and return an analysis result """
        try:
            # Load audio file using librosa
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            if audio is None or len(audio) == 0:
                return {"error": "Failed to load audio file"}

            # Analyze various parameters
            duration = librosa.get_duration(y=audio, sr=sr)
            pitch_score = self.analyze_pitch(audio)
            tone_score = self.analyze_tone(audio)
            variation_score = self.analyze_frequency_variation(audio)
            pause_score = self.analyze_pauses(audio)

            # Calculate overall score
            overall_score = np.mean([pitch_score, tone_score, variation_score, pause_score])

            feedback = self.generate_feedback(overall_score)

            return {
                "overall_score": round(overall_score, 2),
                "pitch_score": round(pitch_score, 2),
                "tone_score": round(tone_score, 2),
                "variation_score": round(variation_score, 2),
                "pause_score": round(pause_score, 2),
                "feedback": feedback,
                "duration": round(duration, 2)
            }

        except Exception as e:
            return {"error": str(e)}

    def analyze_pitch(self, audio: np.ndarray) -> float:
        """ Analyze pitch characteristics of the audio. """
        pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate)
        pitch_data = pitches[pitches > 0]

        if len(pitch_data) == 0:
            return 0.0

        pitch_mean = np.mean(pitch_data)
        pitch_std = np.std(pitch_data)

        stability_score = 100 - min(100, (pitch_std / pitch_mean) * 100)
        range_score = 100 - min(100, abs(pitch_mean - 200) / 2)

        return np.mean([stability_score, range_score])

    def analyze_tone(self, audio: np.ndarray) -> float:
        """ Analyze tone quality using spectral characteristics. """
        stft = librosa.stft(audio)
        db = librosa.amplitude_to_db(abs(stft))

        harmonic, percussive = librosa.decompose.hpss(stft)
        harmonic_ratio = np.sum(abs(harmonic)) / (np.sum(abs(percussive)) + 1e-6)

        spectral_score = min(100, max(0, np.mean(db) + 100))
        harmonic_score = min(100, harmonic_ratio * 50)

        return np.mean([spectral_score, harmonic_score])

    def analyze_frequency_variation(self, audio: np.ndarray) -> float:
        """ Analyze frequency variation and speaking dynamics. """
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        contrast_score = min(100, np.mean(contrast) * 20 + 50)

        centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        centroid_var = np.std(centroids)
        variation_score = min(100, max(0, 100 - (centroid_var * 0.1)))

        return np.mean([contrast_score, variation_score])

    def analyze_pauses(self, audio: np.ndarray) -> float:
        """ Analyze speech pauses and rhythm. """
        rms = librosa.feature.rms(y=audio)[0]
        silence_threshold = np.mean(rms) * 0.5

        pauses = np.sum(rms < silence_threshold) / len(rms)

        pause_segments = np.where(rms < silence_threshold)[0]
        if len(pause_segments) > 1:
            pause_spacing = np.diff(pause_segments)
            spacing_score = min(100, max(0, 100 - np.std(pause_spacing) * 0.1))
        else:
            spacing_score = 50

        return np.mean([(1 - pauses) * 100, spacing_score])

    def generate_feedback(self, score: float) -> str:
        """ Generate feedback based on the overall score. """
        if score > 85:
            return "Excellent voice quality and clarity!"
        elif score > 70:
            return "Good voice quality, but some improvements needed."
        elif score > 50:
            return "Average voice quality. Try improving pitch and tone."
        else:
            return "Poor voice quality. Work on clarity and variation."
