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
      <div className="mt-4 p-4 bg-blue-50 rounded-lg">
        <h3 className="font-bold mb-2 text-blue-900">Voice Analysis Metrics:</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <p className="text-blue-800">
              <strong>Overall Confidence Score:</strong> {metrics.overall_score}
            </p>
            <p className="text-blue-800">
              <strong>Volume Score:</strong> {metrics.volume_score}
            </p>
            <p className="text-blue-800">
              <strong>Clarity Score:</strong> {metrics.clarity_score}
            </p>
            <p className="text-blue-800">
              <strong>Rhythm Score:</strong> {metrics.rhythm_score}
            </p>
          </div>
          <div className="space-y-2">
            <p className="text-blue-800">
              <strong>Pitch Score:</strong> {metrics.pitch_score}
            </p>
            <p className="text-blue-800">
              <strong>Tone Score:</strong> {metrics.tone_score}
            </p>
            <p className="text-blue-800">
              <strong>Pause Score:</strong> {metrics.pause_score}
            </p>
            <p className="text-blue-800">
              <strong>Frequency Variation:</strong> {metrics.frequency_variation}
            </p>
          </div>
        </div>
      </div>
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
                <h2 className="text-red-500 p-2 border rounded-lg">
                  <strong>Rating: </strong>
                  {item.rating}
                </h2>
                <h2 className="p-2 border rounded-lg bg-red-50 text-sm text-red-900">
                  <strong>Your Answer: </strong>
                  {item.userAns}
                </h2>
                <h2 className="p-2 border rounded-lg bg-green-50 text-sm text-green-900">
                  <strong>Correct Answer: </strong>
                  {item.correctAns}
                </h2>
                {renderAnalysisMetrics(item.voiceMetrics)}
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
