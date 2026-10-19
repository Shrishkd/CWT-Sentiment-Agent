from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

ASSET = "BTC-USD"
LOOKBACK_DAYS = 180

END_DATE = datetime.now(timezone.utc).date()
START_DATE = END_DATE - timedelta(days=LOOKBACK_DAYS)

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)