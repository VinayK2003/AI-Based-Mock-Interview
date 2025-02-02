// pages/api/voice-analysis.js
export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const analysisResult = req.body;
    console.log("Voice Feedback :-  ",analysisResult);
    // Here you can:
    // 1. Save the results to your database
    // 2. Process the results further
    // 3. Update your UI state
    // 4. etc.

    // For now, we'll just send back a success response
    res.status(200).json({
      message: 'Analysis results received successfully',
      data: analysisResult
    });
  } catch (error) {
    console.error('Error processing voice analysis:', error);
    res.status(500).json({ message: 'Error processing voice analysis' });
  }
}