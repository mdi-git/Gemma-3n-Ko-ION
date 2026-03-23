import time
import logging
import os

from mcp.server.fastmcp import FastMCP

KOREAN_AGENT_NAME = "태양광 이상 탐지 에이전트"
AGENT_KEY = "solar_anomaly"

logging.basicConfig(
    level=logging.DEBUG,
    format=f"[%(asctime)s][%(levelname)s][solar_anomaly_detection_agent|{KOREAN_AGENT_NAME}] %(message)s",
)
logger = logging.getLogger(__name__)

_selected = (os.getenv("MAGENT_LOG_AGENT") or "").strip().lower()
if _selected and _selected not in {"all", "*", AGENT_KEY}:
    logging.disable(logging.CRITICAL)

mcp = FastMCP(name=f"solar_anomaly_detection_agent|{KOREAN_AGENT_NAME}")


@mcp.tool()
def solar_anomaly_detect(asset_id: str = "solar_array_01") -> dict:
    start = time.monotonic()
    logger.info("tool called: solar_anomaly_detect(asset_id=%s)", asset_id)
    while time.monotonic() - start < 3.0:
        logger.debug("heartbeat: %.2fs", time.monotonic() - start)
        time.sleep(0.25)
    result = {
        "agent": "solar_anomaly_detection_agent",
        "agent_ko": KOREAN_AGENT_NAME,
        "asset_id": asset_id,
        "status": "completed",
        "duration_seconds": round(time.monotonic() - start, 3),
    }
    logger.info("tool completed: %s", result)
    return result


if __name__ == "__main__":
    logger.info("starting MCP stdio server")
    mcp.run(transport="stdio")
