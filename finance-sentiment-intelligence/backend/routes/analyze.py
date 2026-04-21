from fastapi import APIRouter, Depends, HTTPException, status
from backend.schemas.models import AnalyzeRequest, AnalyzeResponse, SentimentScore
from backend.services.predictor import SentimentPredictor, get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analyze", tags=["Sentiment Analysis"])


@router.post(
    "",
    response_model   = AnalyzeResponse,
    summary          = "Analyze sentiment of finance text",
    description      = """
Submit any finance-related text and receive:
- **sentiment**: positive | neutral | negative
- **confidence**: probability score of predicted class
- **scores**: full probability breakdown across all 3 classes
- **cleaned_text**: text after preprocessing (transparency)
    """,
)
async def analyze_sentiment(
    request:   AnalyzeRequest,
    predictor: SentimentPredictor = Depends(get_predictor),
) -> AnalyzeResponse:
    """
    POST /analyze

    Example request body:
    {
        "text": "Reliance Industries reported record profits this quarter",
        "model": "best"
    }
    """
    try:
        logger.info(
            f"POST /analyze | model={request.model} | "
            f"text_len={len(request.text)}"
        )

        result = predictor.predict(
            text       = request.text,
            model_name = request.model,
        )

        return AnalyzeResponse(
            sentiment    = result["sentiment"],
            confidence   = result["confidence"],
            scores       = SentimentScore(**result["scores"]),
            model_used   = result["model_used"],
            cleaned_text = result["cleaned_text"],
            word_count   = result["word_count"],
            status       = "success",
        )

    except ValueError as e:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = str(e),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Prediction failed: {str(e)}",
        )


@router.get(
    "/models",
    summary     = "List available models",
    description = "Returns the names of all ML models currently loaded in memory.",
)
async def list_models(
    predictor: SentimentPredictor = Depends(get_predictor),
) -> dict:
    """GET /analyze/models — returns available model names."""
    return {
        "available_models": predictor.loaded_models,
        "default_model":    "best",
        "status":           "success",
    }