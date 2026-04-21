"""
Finance Sentiment Intelligence System — FastAPI Backend
Run with: uvicorn backend.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.routes.analyze import router as analyze_router
from backend.routes.stats   import router as stats_router
from backend.services.predictor import get_predictor
from backend.schemas.models import HealthResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)

PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-load the predictor singleton so the first
    request doesn't pay the model-loading cost.
    """
    logger.info("🚀 Starting Finance Sentiment Intelligence API...")
    get_predictor()   # warms up model loading
    logger.info("✅ API ready — all models loaded.")
    yield
    logger.info("🛑 Shutting down API.")


# ── App definition ────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Finance Sentiment Intelligence API",
    description = """
## 🏦 Finance Sentiment Intelligence System

Analyze the sentiment of finance-related text using multiple ML models
trained on Reddit data from r/stocks, r/IndianStockMarket, r/investing,
and r/wallstreetbets.

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Analyze text sentiment |
| `GET`  | `/analyze/models` | List available ML models |
| `GET`  | `/stats` | Dataset analytics report |
| `GET`  | `/stats/plot/{name}` | Serve a visualization plot |
| `GET`  | `/stats/plots` | List all available plots |
| `GET`  | `/health` | API health check |

### Models available
- **Logistic Regression** (tuned)
- **Naive Bayes** (tuned)
- **Random Forest** (tuned)
- **best** — automatically selected highest-F1 model
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── Middleware ─────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten this in production
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Static files (serve plots directly) ───────────────────────────────────────

app.mount(
    "/plots",
    StaticFiles(directory=str(PLOTS_DIR)),
    name="plots",
)


# ── Routers ────────────────────────────────────────────────────────────────────

app.include_router(analyze_router)
app.include_router(stats_router)


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": "Finance Sentiment Intelligence API",
        "docs":    "/docs",
        "redoc":   "/redoc",
        "health":  "/health",
    }


@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["System"],
    summary        = "API health check",
)
async def health_check() -> HealthResponse:
    """Returns API status and list of loaded models."""
    predictor = get_predictor()
    return HealthResponse(
        status        = "healthy",
        models_loaded = predictor.loaded_models,
        version       = "1.0.0",
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code = 404,
        content     = {
            "error":   "Endpoint not found",
            "docs":    "/docs",
            "status":  "error",
        },
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code = 500,
        content     = {
            "error":  "Internal server error",
            "status": "error",
        },
    )