from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path

from src.visualization.stats_reporter import load_stats_report
from src.utils.config import MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/stats", tags=["Dataset Statistics"])

PLOTS_DIR = MODELS_DIR.parent.parent / "data" / "plots"


@router.get(
    "",
    summary     = "Get dataset statistics",
    description = """
Returns full analytics report including:
- Sentiment distribution across all posts
- Subreddit-level breakdown
- Numeric statistics (score, comments, upvote ratio, etc.)
- Top words per sentiment class
- List of available plot filenames
    """,
)
async def get_stats() -> dict:
    """GET /stats — returns full dataset analytics report."""
    try:
        logger.info("GET /stats")
        report = load_stats_report()
        report["status"] = "success"
        return report

    except FileNotFoundError as e:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = str(e),
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Failed to load stats: {str(e)}",
        )


@router.get(
    "/plot/{plot_name}",
    summary     = "Serve a visualization plot",
    description = "Returns a PNG image by plot name. "
                  "Use GET /stats to get the list of available_plots.",
    response_class = FileResponse,
)
async def get_plot(plot_name: str) -> FileResponse:
    """
    GET /stats/plot/{plot_name}

    Examples:
        /stats/plot/sentiment_distribution
        /stats/plot/wordclouds
        /stats/plot/hourly_heatmap
    """
    clean_name = plot_name.replace(".png", "")
    path       = PLOTS_DIR / f"{clean_name}.png"

    if not path.exists():
        available = [p.stem for p in PLOTS_DIR.glob("*.png")]
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail      = {
                "error":     f"Plot '{clean_name}' not found.",
                "available": available,
            },
        )

    logger.info(f"GET /stats/plot/{clean_name}")
    return FileResponse(
        path       = str(path),
        media_type = "image/png",
        filename   = f"{clean_name}.png",
    )


@router.get(
    "/plots",
    summary = "List all available plots",
)
async def list_plots() -> dict:
    """GET /stats/plots — returns all available plot names."""
    plots = [p.stem for p in sorted(PLOTS_DIR.glob("*.png"))]
    return {
        "plots":  plots,
        "count":  len(plots),
        "status": "success",
    }

# print("🔥 stats router loading...")
# from fastapi import APIRouter, HTTPException, status
# from fastapi.responses import FileResponse
# from pathlib import Path

# from src.visualization.stats_reporter import load_stats_report
# from src.utils.config import MODELS_DIR
# from src.utils.logger import get_logger

# logger = get_logger(__name__)

# router = APIRouter(prefix="/stats", tags=["Dataset Statistics"])

# PLOTS_DIR = MODELS_DIR.parent.parent / "data" / "plots"


# @router.get(
#     "",
#     summary     = "Get dataset statistics",
#     description = """
# Returns full analytics report including:
# - Sentiment distribution across all posts
# - Subreddit-level breakdown
# - Numeric statistics (score, comments, upvote ratio, etc.)
# - Top words per sentiment class
# - List of available plot filenames
#     """,
# )
# async def get_stats() -> dict:
#     """GET /stats — returns full dataset analytics report."""
#     try:
#         logger.info("GET /stats")
#         report = load_stats_report()
#         report["status"] = "success"
#         return report

#     except FileNotFoundError as e:
#         raise HTTPException(
#             status_code = status.HTTP_404_NOT_FOUND,
#             detail      = str(e),
#         )
#     except Exception as e:
#         logger.error(f"Stats error: {e}")
#         raise HTTPException(
#             status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail      = f"Failed to load stats: {str(e)}",
#         )


# @router.get(
#     "/plot/{plot_name}",
#     summary     = "Serve a visualization plot",
#     description = "Returns a PNG image by plot name. "
#                   "Use GET /stats to get the list of available_plots.",
#     response_class = FileResponse,
# )
# async def get_plot(plot_name: str) -> FileResponse:
#     """
#     GET /stats/plot/{plot_name}

#     Examples:
#         /stats/plot/sentiment_distribution
#         /stats/plot/wordclouds
#         /stats/plot/radar_chart
#     """
#     # Strip .png extension if user includes it
#     clean_name = plot_name.replace(".png", "")
#     path       = PLOTS_DIR / f"{clean_name}.png"

#     if not path.exists():
#         available = [p.stem for p in PLOTS_DIR.glob("*.png")]
#         raise HTTPException(
#             status_code = status.HTTP_404_NOT_FOUND,
#             detail      = {
#                 "error":     f"Plot '{clean_name}' not found.",
#                 "available": available,
#             },
#         )

#     logger.info(f"GET /stats/plot/{clean_name}")
#     return FileResponse(
#         path         = str(path),
#         media_type   = "image/png",
#         filename     = f"{clean_name}.png",
#     )


# @router.get(
#     "/plots",
#     summary = "List all available plots",
# )
# async def list_plots() -> dict:
#     """GET /stats/plots — returns all available plot names."""
#     plots = [p.stem for p in sorted(PLOTS_DIR.glob("*.png"))]
#     return {
#         "plots":  plots,
#         "count":  len(plots),
#         "status": "success",
#     }