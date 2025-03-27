"use client";
import React, { useEffect, useState } from "react";
import { db } from "../../../../../utils/db";
import { UserAnswer } from "../../../../../utils/schema";
import { eq } from "drizzle-orm";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "../../../../../components/ui/collapsible";
import { ChevronsUpDown } from "lucide-react";
import { Button } from "../../../../../components/ui/button";
import { useRouter } from "next/navigation";
import { MdOutlineDashboardCustomize } from "react-icons/md";
import { useParams } from "next/navigation";

// Emotion color mapping for better visualization
const EMOTION_COLORS = {
  Happy: "bg-green-100 text-green-900",
  Neutral: "bg-gray-100 text-gray-900",
  Sad: "bg-blue-100 text-blue-900",
  Angry: "bg-red-100 text-red-900",
  Disgust: "bg-purple-100 text-purple-900",
  Fear: "bg-yellow-100 text-yellow-900",
  Surprise: "bg-orange-100 text-orange-900"
};

// Movement direction color mapping
const MOVEMENT_COLORS = {
  left: "bg-blue-100 text-blue-900",
  right: "bg-green-100 text-green-900", 
  up: "bg-yellow-100 text-yellow-900",
  down: "bg-red-100 text-red-900"
};

function Feedback() {
  const [feedbackList, setFeedbackList] = useState([]);
  const router = useRouter();
  const params = useParams();
  const interviewId = params.interviewId;

  useEffect(() => {
    GetFeedback();
  }, []);

  const GetFeedback = async () => {
    const result = await db
      .select()
      .from(UserAnswer)
      .where(eq(UserAnswer.mockIdRef, interviewId))
      .orderBy(UserAnswer.id);

    console.log("result  ", result);
    setFeedbackList(result);
  };

  const renderAnalysisMetrics = (metrics) => {
    if (!metrics) return null;
    
    return (
      <>
        {/* Previous Voice Metrics Section Remains the Same */}
        <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-bold mb-2 text-blue-900">Voice Analysis Metrics:</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <p className="text-blue-800">
              <strong>Overall Confidence Score:</strong> {metrics.voiceMetrics.overall_score}
            </p>
            <p className="text-blue-800">
              <strong>Volume Score:</strong> {metrics.voiceMetrics.volume_score}
            </p>
            <p className="text-blue-800">
              <strong>Clarity Score:</strong> {metrics.voiceMetrics.clarity_score}
            </p>
            <p className="text-blue-800">
              <strong>Rhythm Score:</strong> {metrics.voiceMetrics.rhythm_score}
            </p>
          </div>
          <div className="space-y-2">
            <p className="text-blue-800">
              <strong>Pitch Score:</strong> {metrics.voiceMetrics.pitch_score}
            </p>
            <p className="text-blue-800">
              <strong>Tone Score:</strong> {metrics.voiceMetrics.tone_score}
            </p>
            <p className="text-blue-800">
              <strong>Pause Score:</strong> {metrics.voiceMetrics.pause_score}
            </p>
            <p className="text-blue-800">
              <strong>Frequency Variation:</strong> {metrics.voiceMetrics.frequency_variation}
            </p>
          </div>
        </div>
      </div>
  
        {metrics.videoMetrics && (
          <div className="mt-4 p-4 bg-green-50 rounded-lg">
            <h3 className="font-bold mb-2 text-green-900">Video Analysis Metrics:</h3>
            
            {/* Emotion Analysis with Enhanced Details */}
            <div className="mb-4">
              <h4 className="font-semibold text-green-800 mb-2">Comprehensive Emotion Analysis:</h4>
              <div className="grid grid-cols-2 gap-4 mb-4">
      <div className="bg-white border border-green-200 rounded-lg p-4 shadow-sm">
        <div className="flex justify-between items-center mb-2">
          <h4 className="font-semibold text-balck-800">Blink Rate</h4>
          <span className="text-sm font-bold text-black-600">
            {metrics.videoMetrics.blink_rate.toFixed(2)} blinks/min
          </span>
        </div>
      </div>
      
      <div className="bg-white border border-green-200 rounded-lg p-4 shadow-sm">
        <div className="flex justify-between items-center mb-2">
          <h4 className="font-semibold text-black-800">Positive Emotions</h4>
          <span className="text-sm font-bold text-black-600">
            {metrics.videoMetrics.positive_emotions_percentage}%
          </span>
        </div>
      </div>
    </div>
              <div className="grid grid-cols-3 gap-3">
                {metrics.videoMetrics.raw_metrics.emotion_count && 
                 Object.entries(metrics.videoMetrics.raw_metrics.emotion_count).map(([emotion, details]) => (
                  <div 
                    key={emotion} 
                    className={`${EMOTION_COLORS[emotion] || 'bg-gray-100'} p-3 rounded-lg shadow-sm`}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <h5 className="font-bold text-lg">{emotion}</h5>
                      <span className="text-sm font-semibold">
                        {details}%
                      </span>
                    </div>
                    <div className="grid grid-cols-2 gap-1 text-sm">
                      <p>
                        <strong>Count:</strong> {details}
                      </p>
                    </div>
                    <div className="mt-2 w-full bg-white/50 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full" 
                        style={{width: `${details}%`}}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Head Movement Analysis with Enhanced Details */}
            <div>
              <h4 className="font-semibold text-green-800 mb-2">Detailed Head Movement Analysis:</h4>
              <div >
                {/* Leaning Details */}
                <div className="bg-green-100 p-4 rounded-lg">
                  <h5 className="font-bold text-green-900 mb-3">Head Leaning Analysis</h5>
                  {metrics.videoMetrics.raw_metrics.head_movement && 
                   Object.entries(metrics.videoMetrics.raw_metrics.head_movement).map(([direction, details]) => (
                    <div 
                      key={direction} 
                      className={`${MOVEMENT_COLORS[direction] || 'bg-gray-100'} p-2 rounded mb-2`}
                    >
                      <div className="flex justify-between items-center">
                        <strong className="capitalize">{direction} Lean:</strong>
                        <span className="text-sm font-semibold">
                          {details}%
                        </span>
                      </div>
                      <div className="grid grid-cols-2 gap-1 text-sm mt-1">
                        <p>Count: {details}</p>
                      </div>
                      <div className="mt-2 w-full bg-white/50 rounded-full h-2">
                        <div 
                          className="bg-green-500 h-2 rounded-full" 
                          style={{width: `${details}%`}}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </>
    );
  };

  return (
    <div className="p-10">
      <div className={feedbackList.length === 0 ? "block" : "hidden"}>
        <h2 className="font-bold text-xl text-gray-500">
          No Interview Feedback Record Found
        </h2>
      </div>

      <div className={feedbackList.length > 0 ? "block" : "hidden"}>
        <h2 className="text-3xl font-bold text-green-600">
          Congratulations!
        </h2>
        <h2 className="font-bold text-2xl">
          Here is your interview feedback
        </h2>
        <h2 className="text-sm text-gray-500">
          Find below interview question with correct answer, your answer, voice analysis, and feedback for improvement.
        </h2>
        
        {feedbackList.map((item, index) => (
          <Collapsible key={index} className="mt-7">
            <CollapsibleTrigger className="p-2 bg-secondary rounded-lg my-2 flex justify-between text-left gap-7 w-full">
              {item.question}
              <ChevronsUpDown className="h-5 w-5" />
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="flex flex-col gap-2 w-full">
                {/* Find the existing rating section and replace it with this */}
<h2 className="p-2 border rounded-lg bg-blue-50 text-base text-blue-900 flex items-center gap-4">
  <strong className="text-blue-700 text-lg">Rating:</strong> 
  <span className="text-lg">{item.rating}</span>
  
  {item.bleuScore !== undefined && (
    <div className="flex items-center gap-2">
      <span className="text-blue-600 font-semibold text-base">Bleu Score:</span>
      <span className="bg-blue-100 px-2 py-1 rounded text-blue-800 text-lg font-bold">
        {item.bleuScore}
      </span>
    </div>
  )}
  
  {item.fillerWordsCount !== undefined && (
    <div className="flex items-center gap-2">
      <span className="text-blue-600 font-semibold text-base">Filler Words:</span>
      <span className="bg-blue-100 px-2 py-1 rounded text-blue-800 text-lg font-bold">
        {item.fillerWordsCount}
      </span>
    </div>
  )}
</h2>
                <h2 className="p-2 border rounded-lg bg-red-50 text-sm text-red-900">
                  <strong>Your Answer: </strong>
                  {item.userAns}
                </h2>
                <h2 className="p-2 border rounded-lg bg-green-50 text-sm text-green-900">
                  <strong>Correct Answer: </strong>
                  {item.correctAns}
                </h2>
                {renderAnalysisMetrics(item)}
                <h2 className="p-2 border rounded-lg bg-yellow-50 text-sm text-yellow-900">
                  <strong>Feedback: </strong>
                  {item.feedback}
                </h2>
              </div>
            </CollapsibleContent>
          </Collapsible>
        ))}
      </div>

      <Button
        onClick={() => router.replace("/dashboard")}
        className="flex gap-1 my-4 bg-green-600 hover:bg-green-700"
      >
        Return to Dashboard
        <MdOutlineDashboardCustomize className="w-4 h-4" />
      </Button>
    </div>
  );
}

export default Feedback;