import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { Button } from "../../../../../../components/ui/button";
import { useUser } from "@clerk/nextjs";
import { db } from "../../../../../../utils/db";
import { UserAnswer } from "../../../../../../utils/schema";
import { useParams } from "next/navigation";
import useSpeechToText from "react-hook-speech-to-text";
import { Mic, StopCircle } from "lucide-react";
import { toast } from "sonner";
import { chatSession } from "../../../../../../utils/GeminiAIModal";
import moment from "moment";
import { eq, and } from "drizzle-orm";

function RecordQuestionSection({
  mockInterviewQuestion = [],
  activeQuestionIndex = 0,
  interviewData,
}) {
  const [recording, setRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [userAnswer, setUserAnswer] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const finalAnswerRef = useRef("");
  const { user } = useUser();
  const params = useParams();

  const {
    error: speechError,
    interimResult,
    isRecording,
    results,
    startSpeechToText,
    stopSpeechToText,
    setResults,
  } = useSpeechToText({
    continuous: true,
    useLegacyResults: false,
    speechRecognitionProperties: {
      lang: "en-IN",
    },
  });

  // Listen for tab changes using the Page Visibility API.
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden) {
        toast("Tab change detected. Please remain on this page during the interview.");
        // Optionally, you can automatically stop recording if desired.
        if (recording) {
          stopRecording();
        }
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [recording]);

  useEffect(() => {
    if (!isSubmitted) {
      const newTranscript = results.map(result => result?.transcript || "").join(" ");
      if (newTranscript) {
        setUserAnswer(prevAnswer => {
          const updatedAnswer = prevAnswer + " " + newTranscript;
          finalAnswerRef.current = updatedAnswer.trim();
          return updatedAnswer;
        });
      }
    }
  }, [results, isSubmitted]);

  const startRecording = async () => {
    try {
      setIsSubmitted(false);
      setError(null);
      setAnalysisResult(null);
      setUserAnswer("");
      finalAnswerRef.current = "";
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          const audioBlob = new Blob(chunksRef.current, {
            type: "audio/webm;codecs=opus",
          });
          await sendAudio(audioBlob);
        } finally {
          stream.getTracks().forEach((track) => track.stop());
        }
      };

      mediaRecorder.start();
      setRecording(true);
      startSpeechToText();
    } catch (error) {
      setError(`Error accessing microphone: ${error.message}`);
      console.error("Error accessing microphone", error);
    }
  };

  const stopRecording = async () => {
    if (mediaRecorderRef.current && recording) {
      setIsSubmitted(true);
      const capturedAnswer = finalAnswerRef.current || userAnswer;
      console.log("Final captured answer:", capturedAnswer);
      
      mediaRecorderRef.current.stop();
      setRecording(false);
      await stopSpeechToText();
      
      if (capturedAnswer) {
        finalAnswerRef.current = capturedAnswer;
      }
    }
  };

  const sendAudio = async (blob) => {
    setLoading(true);
    setError(null);

    try {
      if (!blob || blob.size === 0) {
        throw new Error("No audio data recorded");
      }

      const capturedAnswer = finalAnswerRef.current || userAnswer;
      console.log("Answer being sent to server:", capturedAnswer);

      if (!capturedAnswer) {
        throw new Error("No answer was recorded");
      }

      const formData = new FormData();
      formData.append("audio", blob, "recording.webm");

      const userEmail = user?.primaryEmailAddress?.emailAddress;
      if (!userEmail) {
        throw new Error("User email not found");
      }

      const response = await fetch(
        `http://localhost:8000/upload-audio/${encodeURIComponent(
          userEmail
        )}/${activeQuestionIndex}`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `Server error: ${response.status}`);
      }
      const data = await response.json();
      setAnalysisResult(data);

      await updateUserAnswer(data.metrics, capturedAnswer);
    } catch (error) {
      setError(`Failed to process recording: ${error.message}`);
      console.error("Error sending audio:", error);
    } finally {
      setLoading(false);
    }
  };

  const updateUserAnswer = async (voiceMetrics = null, capturedAnswer) => {
    if (!capturedAnswer) {
      setError("No answer to submit");
      return;
    }

    setLoading(true);
    try {
      if (!mockInterviewQuestion[activeQuestionIndex]) {
        throw new Error("Invalid question index");
      }

      console.log("Saving answer to DB:", capturedAnswer);

      const feedbackPrompt = `Question: ${mockInterviewQuestion[activeQuestionIndex]?.question}, 
        User Answer: ${capturedAnswer}. Please provide a rating out of 10 and feedback in JSON format with fields: "rating" and "feedback".Please dont look for punctuation mistakes!Go easy on ratings`;

      const result = await chatSession.sendMessage(feedbackPrompt);
      const mockJsonResp = result.response
        .text()
        .replace("```json", "")
        .replace("```", "");
      const JsonFeedbackResp = JSON.parse(mockJsonResp);
      console.log("answer hai ", mockInterviewQuestion);
      const correctAnswer =
        mockInterviewQuestion[activeQuestionIndex].answerExample ||
        mockInterviewQuestion[activeQuestionIndex].answer;

      const nlpResponse = await fetch('/api/process-text', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          reference: correctAnswer,
          candidate: capturedAnswer
        }),
      });
  
      const nlpData = await nlpResponse.json();
      if (!nlpData.success) {
        throw new Error(nlpData.error);
      }

      await db
        .update(UserAnswer)
        .set({
          question: mockInterviewQuestion[activeQuestionIndex]?.question,
          userAns: capturedAnswer,
          correctAns: correctAnswer,
          feedback: JsonFeedbackResp?.feedback,
          rating: JsonFeedbackResp?.rating,
          createdAt: moment().format("DD-MM-yyyy"),
          userEmail: user?.primaryEmailAddress.emailAddress,
          voiceMetrics: voiceMetrics || null,
        })
        .where(
          and(
            eq(UserAnswer.mockIdRef, interviewData?.mockId)
          )
        );

      toast("User Answer Recorded successfully.");

      setTimeout(() => {
        setResults([]);
        setUserAnswer("");
        finalAnswerRef.current = "";
        setIsSubmitted(false);
      }, 500);

    } catch (error) {
      setError(`Failed to update answer: ${error.message}`);
      console.error("Error updating answer:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center flex-col">
      <div className="flex flex-col justify-center items-center rounded-lg p-5">
        <img src="/webcam3.png" alt="WebCAM" width={140} height={140} />
      </div>
      <div className="flex flex-col justify-center items-center rounded-lg p-5 mt-5 bg-black">
        <Webcam mirrored={true} style={{ height: 300, width: "100%", zIndex: 100 }} />
      </div>

      {recording && (
        <div className="mt-4 p-4 bg-white rounded shadow w-full max-w-2xl">
          <h3 className="font-bold mb-2">Current Answer:</h3>
          <p>{userAnswer || "Recording..."}</p>
        </div>
      )}

      <Button 
        disabled={loading} 
        variant="outline" 
        className="my-10" 
        onClick={recording ? stopRecording : startRecording}
      >
        {loading ? (
          <h2 className="text-gray-500">Processing...</h2>
        ) : recording ? (
          <h2 className="text-red-500 flex animate-pulse items-center gap-2">
            <StopCircle /> Stop Recording...
          </h2>
        ) : (
          <h2 className="flex gap-2 items-center">
            <Mic /> Record Answer
          </h2>
        )}
      </Button>

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded">
          {error}
        </div>
      )}
    </div>
  );
}

export default RecordQuestionSection;
