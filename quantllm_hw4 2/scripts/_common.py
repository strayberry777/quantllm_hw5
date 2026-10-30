import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

RAW = ROOT / "raw"
OUT = ROOT / "outputs"
CXMT = "688825.SH"
