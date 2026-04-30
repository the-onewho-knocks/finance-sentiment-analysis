"""
backend/main.py
───────────────
Finance Sentiment Intelligence System — FastAPI Backend

Changes from v1:
  • Prometheus /metrics endpoint via prometheus-fastapi-instrumentator
  • Request-level metrics (latency, status codes, endpoint labels)
  • Startup exports current dataset size to Prometheus gauge

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from backend.routes.analyze import router as analyze_router
from backend.routes.stats   import router as stats_router
from backend.services.predictor import get_predictor
from backend.schemas.models import HealthResponse
from src.utils.config import RAW_DATA_FILE
from src.utils.logger import get_logger
from src.metrics.prometheus_metrics import posts_in_raw_dataset

logger = get_logger(__name__)

PLOTS_DIR = Path("data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Finance Sentiment Intelligence API…")
    get_predictor()   # warms up model loading

    # Sync dataset size to Prometheus on startup
    try:
        import pandas as pd
        if RAW_DATA_FILE.exists():
            df = pd.read_csv(RAW_DATA_FILE, usecols=["id"])
            posts_in_raw_dataset.set(len(df))
            logger.info(f"📊 Prometheus gauge set: {len(df)} rows in raw dataset")
    except Exception as exc:
        logger.warning(f"Could not pre-populate dataset gauge: {exc}")

    logger.info("✅ API ready — all models loaded.")
    yield
    logger.info("🛑 Shutting down API.")


# ── App definition ─────────────────────────────────────────────────────────────

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
| `GET`  | `/metrics` | Prometheus metrics scrape endpoint |

### Models available
- **Logistic Regression** (tuned)
- **Naive Bayes** (tuned)
- **Random Forest** (tuned)
- **best** — automatically selected highest-F1 model
    """,
    version     = "2.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)


# ── Middleware ─────────────────────────────────────────────────────────────────
# FIX: allow_credentials=True is invalid with allow_origins=["*"].
# Use specific origins if you need credentials, otherwise set credentials to False.

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = False,   # was True — invalid with wildcard origins
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Prometheus instrumentation ─────────────────────────────────────────────────
# This adds /metrics endpoint automatically and instruments all routes.

Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=False,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    inprogress_name="fastapi_inprogress",
    inprogress_labels=True,
).instrument(app).expose(app, endpoint="/metrics", tags=["Monitoring"])


# ── Static files ───────────────────────────────────────────────────────────────

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
        "metrics": "/metrics",
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
        version       = "2.0.0",
    )


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code = 404,
        content     = {"error": "Endpoint not found", "docs": "/docs", "status": "error"},
    )


@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(
        status_code = 500,
        content     = {"error": "Internal server error", "status": "error"},
    )