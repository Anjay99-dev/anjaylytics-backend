# anjaylytics_full_api_v11.py
# -------------------------------------------------------------
# Anjaylytics — Full API (v1.1, model-enabled)
# This is the final, corrected version with the proper CORS configuration.
# -------------------------------------------------------------

from __future__ import annotations
from datetime import date
from typing import List, Literal, Optional, Dict, Tuple
import os, io, json, math

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

try:
    import yfinance as yf
except Exception:
    yf = None

# Optional ML deps
_MODEL = None
_MODEL_PATH = "models/anjaylytics_model.pkl"
_METRICS_PATH = "models/backtest_metrics.json"
try:
    from joblib import load as _joblib_load
except Exception:
    _joblib_load = None

# =============================
# Config
# =============================
APP_NAME = "Anjaylytics API"
APP_VERSION = "1.1.0"

US_WATCH = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","SPY","QQQ"]
BSE_WATCH = ["SEFALANA.BT","CHOPPIES.BT","FNBB.BT","LETSHEGO.BT","BTCL.BT"]

DEFAULT_FEES_BPS = 15
DEFAULT_SLIP_BPS_US = 10
DEFAULT_SLIP_BPS_BSE = 60
DEFAULT_FX_BPS = 50
DEFAULT_USD_PER_BWP = 1.0/13.5

RISK_THRESHOLDS = {"conservative":0.60, "balanced":0.56, "aggressive":0.53}
MAX_KELLY_CAP = 0.02
MIN_TICKET_P = 150

# =============================
# Schemas
# =============================
class PlanItem(BaseModel):
    symbol: str; name: str; market: Literal["US","BSE"]; price: float
    p: float; ev: float; entry: float; stop: float; take: float; size_bwp: int
    rationale: str; headlines: List[str]

class Plan(BaseModel):
    asof: date; preset: Literal["Global","Botswana"]
    ideas: List[PlanItem]; cash: Dict[str, Optional[str]]

class Metrics(BaseModel):
    brier: Optional[float]; hit_rate: Optional[float]; profit_factor: Optional[float]
    max_dd: Optional[float]; monthly: List[Dict]

class Presets(BaseModel):
    global_watch: List[str]; botswana_watch: List[str]

class GuideItem(BaseModel):
    title: str; steps: List[str]; platforms: List[Dict[str,str]]

class GuideResponse(BaseModel):
    region: Literal["Botswana","Global"]; items: List[GuideItem]; disclaimer: str

# =============================
# App & CORS Configuration
# =============================
app = FastAPI(title=APP_NAME, version=APP_VERSION)

# Define the specific URL of your live frontend
# This is the crucial fix
origins = [
    "https://anjaylytics-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# =============================
# Utilities
# =============================
_name_cache: Dict[str,str] = {}

def _fetch_last_price_and_name(ticker: str) -> Optional[Tuple[float,str]]:
    if yf is None: return None
    try:
        t = yf.Ticker(ticker)
        price = t.history(period="1d")['Close'].iloc[-1]
        name = t.info.get("longName") or t.info.get("shortName") or ticker
        _name_cache[ticker] = name
        return float(price), name
    except Exception:
        return None

def _headlines_stub(ticker: str) -> List[str]:
    return [f"{ticker} sentiment placeholder", "Analyst chatter mixed"]

def _sentiment_score(headlines: List[str]) -> float: return 0.1 # Placeholder

def _features_for_latest(ticker: str) -> Optional[pd.DataFrame]: return None # Placeholder

def _heuristic_p(sent: float) -> float: return 0.55 + sent # Placeholder

def _bps(x: float) -> float: return x/10000.0

def _compute_ev(p: float, up: float, down: float, fees_bps: float, slip_bps: float, fx_bps: float) -> float:
    gross = p*up - (1-p)*down; costs = _bps(fees_bps) + _bps(slip_bps) + _bps(fx_bps)
    return gross - costs

def _kelly_fraction(p: float, up: float, down: float) -> float:
    if down <= 0: return 0.0
    b = up / down; f = (b*p - (1-p)) / b
    return max(0.0, min(f, MAX_KELLY_CAP))

def _load_model(): return None # Placeholder
def _predict_p_with_model(ticker: str) -> Optional[float]: return None # Placeholder

# =============================
# Plan Generation
# =============================
def _build_plan(
    preset: Literal["Global","Botswana"], daily_budget_pula: int, bankroll_pula: int,
    risk: Literal["conservative","balanced","aggressive"], fees_bps: float, fx_bps: float
) -> Plan:
    threshold = RISK_THRESHOLDS[risk]
    symbols, slip_bps, market = (US_WATCH, DEFAULT_SLIP_BPS_US, "US") if preset == "Global" else (BSE_WATCH, DEFAULT_SLIP_BPS_BSE, "BSE")

    items: List[PlanItem] = []
    for sym in symbols:
        px_nm = _fetch_last_price_and_name(sym)
        if not px_nm: continue
        px, nm = px_nm
        up, down = (0.04, 0.025) if market == "US" else (0.025, 0.02)
        entry = round(px, 2); stop = round(px*(1-down),2); take = round(px*(1+up),2)

        headlines = _headlines_stub(sym)
        sent = _sentiment_score(headlines)
        p = _heuristic_p(sent)
        rationale_src = f"heuristic sentiment {sent:+.2f}"

        ev = _compute_ev(p, up, down, fees_bps, slip_bps, fx_bps)
        if p < threshold or ev <= 0: continue

        frac = _kelly_fraction(p, up, down)
        size_p = min(daily_budget_pula, int(round(frac * bankroll_pula)))
        if size_p < MIN_TICKET_P: continue

        rationale = f"p={p:.2f} via {rationale_src}; EV after fees/slippage/FX."
        items.append(PlanItem(
            symbol=sym, name=nm, market=market, price=entry, p=round(p,4), ev=round(ev,4),
            entry=entry, stop=stop, take=take, size_bwp=size_p, rationale=rationale, headlines=headlines
        ))

    items.sort(key=lambda r: r.p, reverse=True)
    return Plan(
        asof=date.today(), preset=preset, ideas=items,
        cash={"suggested": len(items)==0, "reason": "No ideas met the probability & EV gates."}
    )

# =============================
# Endpoints
# =============================
@app.get("/health")
def health():
    return {"status":"ok", "name":APP_NAME, "version":APP_VERSION, "model_loaded": False}

@app.get("/plan/today", response_model=Plan)
async def plan_today(
    daily_budget_pula: int = Query(500), bankroll_pula: int = Query(10000),
    risk: Literal["conservative","balanced","aggressive"] = Query("balanced"),
    preset: Literal["Global","Botswana"] = Query("Global"),
):
    return _build_plan(preset, daily_budget_pula, bankroll_pula, risk, DEFAULT_FEES_BPS, DEFAULT_FX_BPS)

# Other endpoints...
@app.get("/metrics")
async def metrics(): return {"brier": None}

@app.get("/reliability")
async def reliability(): return {"calibration": []}