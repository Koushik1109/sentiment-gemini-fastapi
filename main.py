from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gemini Sentiment Analysis API")

# Request model
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="Customer comment to analyze")

# Exact response model (this becomes the JSON schema)
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int = Field(..., ge=1, le=5, description="1=highly negative, 5=highly positive")

# Gemini client (automatically reads GEMINI_API_KEY from environment)
client = genai.Client()

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    if not request.comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    try:
        # System + User prompt (Gemini structured outputs style)
        contents = [
            {
                "role": "system",
                "parts": [{
                    "text": (
                        "You are a strict and accurate sentiment judge. "
                        "Analyze the comment and return ONLY the sentiment and rating. "
                        "Be consistent every single time. "
                        "5 = extremely positive, 1 = extremely negative, 3 = completely neutral."
                    )
                }]
            },
            {
                "role": "user",
                "parts": [{"text": request.comment}]
            }
        ]

        response = client.models.generate_content(
            model="gemini-2.5-flash",   # Best free model with excellent structured outputs (Feb 2026)
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": SentimentResponse.model_json_schema(),
            },
        )

        # Parse the guaranteed JSON
        result = SentimentResponse.model_validate_json(response.text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")


@app.get("/")
async def home():
    return {"message": "Gemini Sentiment API is running! Test at /docs"}
