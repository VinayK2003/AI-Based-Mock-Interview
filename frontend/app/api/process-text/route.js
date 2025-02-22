// app/api/process-text/route.js

// Simple BLEU score calculation
function calculateBLEU(reference, candidate) {
    // Tokenize the strings into words
    const tokenize = (text) => text.toLowerCase().match(/\b\w+\b/g) || [];
    const referenceTokens = tokenize(reference);
    const candidateTokens = tokenize(candidate);
  
    // Calculate word overlap
    const overlap = candidateTokens.filter(token => 
      referenceTokens.includes(token)
    ).length;
  
    // Calculate precision
    const precision = overlap / candidateTokens.length;
  
    // Calculate brevity penalty
    const brevityPenalty = Math.exp(
      Math.min(0, 1 - (referenceTokens.length / candidateTokens.length))
    );
  
    return precision * brevityPenalty;
  }
  
  // Calculate similarity score based on word overlap
  function calculateSimilarity(reference, candidate) {
    const tokenize = (text) => text.toLowerCase().match(/\b\w+\b/g) || [];
    const referenceTokens = new Set(tokenize(reference));
    const candidateTokens = new Set(tokenize(candidate));
  
    // Find common words
    const intersection = new Set(
      [...referenceTokens].filter(x => candidateTokens.has(x))
    );
  
    // Calculate Jaccard similarity
    const union = new Set([...referenceTokens, ...candidateTokens]);
    return intersection.size / union.size;
  }
  
  export async function POST(request) {
    try {
      const { reference, candidate } = await request.json();
  
      if (!reference || !candidate) {
        throw new Error('Both reference and candidate texts are required');
      }
  
      // Calculate metrics
      const bleuScore = calculateBLEU(reference, candidate);
      const similarity = calculateSimilarity(reference, candidate);
  
      // Calculate keyword coverage
      const getKeywords = (text) => {
        // Simple keyword extraction (words longer than 4 characters)
        return new Set(
          text.toLowerCase()
            .match(/\b\w{4,}\b/g)
            ?.filter(word => !['this', 'that', 'with', 'from'].includes(word)) || []
        );
      };
  
      const referenceKeywords = getKeywords(reference);
      const candidateKeywords = getKeywords(candidate);
      const matchedKeywords = [...referenceKeywords]
        .filter(keyword => candidateKeywords.has(keyword));
  
      const keywordCoverage = referenceKeywords.size > 0 
        ? matchedKeywords.length / referenceKeywords.size
        : 0;
        console.log("similarity :-",similarity*100)
        console.log("bleuScore :-",bleuScore)
  
      return Response.json({
        success: true,
        metrics: {
          bleuScore: Math.round(bleuScore * 100),
          similarity: Math.round(similarity * 100),
          keywordCoverage: Math.round(keywordCoverage * 100),
          matchedKeywords,
        }
      });
    } catch (error) {
      return Response.json({
        success: false,
        error: error.message
      }, { status: 500 });
    }
  }