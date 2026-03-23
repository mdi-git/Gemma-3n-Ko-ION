import time
import logging
import os

from mcp.server.fastmcp import FastMCP

KOREAN_AGENT_NAME = "풍력 발전량 예측 에이전트"
AGENT_KEY = "wind_forecast"

logging.basicConfig(
    level=logging.DEBUG,
    format=f"[%(asctime)s][%(levelname)s][wind_forecast_agent|{KOREAN_AGENT_NAME}] %(message)s",
)
logger = logging.getLogger(__name__)

_selected = (os.getenv("MAGENT_LOG_AGENT") or "").strip().lower()
if _selected and _selected not in {"all", "*", AGENT_KEY}:
    logging.disable(logging.CRITICAL)

mcp = FastMCP(name=f"wind_forecast_agent|{KOREAN_AGENT_NAME}")


@mcp.tool()
def wind_forecast(location: str = "default") -> dict:
    start = time.monotonic()
    logger.info("tool called: wind_forecast(location=%s)", location)
    while time.monotonic() - start < 3.0:
        logger.debug("heartbeat: %.2fs", time.monotonic() - start)
        time.sleep(0.25)
    result = {
        "agent": "wind_forecast_agent",
        "agent_ko": KOREAN_AGENT_NAME,
        "location": location,
        "status": "completed",
        "duration_seconds": round(time.monotonic() - start, 3),
    }
    logger.info("tool completed: %s", result)
    return result


if __name__ == "__main__":
    logger.info("starting MCP stdio server")
    mcp.run(transport="stdio")
