import time
import logging
import os

from mcp.server.fastmcp import FastMCP

KOREAN_AGENT_NAME = "전력 소비 예측 에이전트"
AGENT_KEY = "consumption_forecast"

logging.basicConfig(
    level=logging.DEBUG,
    format=f"[%(asctime)s][%(levelname)s][consumption_forecast_agent|{KOREAN_AGENT_NAME}] %(message)s",
)
logger = logging.getLogger(__name__)

_selected = (os.getenv("MAGENT_LOG_AGENT") or "").strip().lower()
if _selected and _selected not in {"all", "*", AGENT_KEY}:
    logging.disable(logging.CRITICAL)

mcp = FastMCP(name=f"consumption_forecast_agent|{KOREAN_AGENT_NAME}")


@mcp.tool()
def consumption_forecast(customer_group: str = "residential") -> dict:
    start = time.monotonic()
    logger.info("tool called: consumption_forecast(customer_group=%s)", customer_group)
    while time.monotonic() - start < 3.0:
        logger.debug("heartbeat: %.2fs", time.monotonic() - start)
        time.sleep(0.25)
    result = {
        "agent": "consumption_forecast_agent",
        "agent_ko": KOREAN_AGENT_NAME,
        "customer_group": customer_group,
        "status": "completed",
        "duration_seconds": round(time.monotonic() - start, 3),
    }
    logger.info("tool completed: %s", result)
    return result


if __name__ == "__main__":
    logger.info("starting MCP stdio server")
    mcp.run(transport="stdio")
