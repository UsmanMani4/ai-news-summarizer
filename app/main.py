# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from extractive.textrank import summarize_text
from abstractive.t5_model import generate_summary

# -------------------------------
# FastAPI App
# -------------------------------
app = FastAPI(title="AI News Summarizer API")

# -------------------------------
# CORS Middleware
# -------------------------------
origins = [
    "http://localhost:5500",  # frontend server
    "http://127.0.0.1:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# -------------------------------
# Request / Response Models
# -------------------------------
class SummarizeRequest(BaseModel):
    text: str

class SummarizeResponse(BaseModel):
    extractive: str
    abstractive: str

# -------------------------------
# Pipeline Function
# -------------------------------
def run_pipeline(article_text: str):
    """Run extractive + abstractive summarization pipeline."""
    extractive_summary = summarize_text(article_text)
    abstractive_summary = generate_summary(extractive_summary)
    return extractive_summary, abstractive_summary

# -------------------------------
# FastAPI Endpoints
# -------------------------------
@app.get("/")
def root():
    """Check server status."""
    return {"message": "AI News Summarizer API is running"}

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_news(request: SummarizeRequest):
    """Return both extractive and abstractive summaries."""
    try:
        extractive_summary, abstractive_summary = run_pipeline(request.text)
        return {
            "extractive": extractive_summary,
            "abstractive": abstractive_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))