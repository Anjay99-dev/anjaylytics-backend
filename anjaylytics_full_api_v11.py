"""
Anjaylytics — Full API (v1.2, model-enabled)
- Fixes Plan.cash schema mismatch
- Keeps yfinance-based live pricing & features
- Graceful fallbacks when model or data are missing
- CORS configured for Vercel + localhost

Run:
  pip install fastapi uvicorn pydantic yfinance joblib pandas
  uvicorn anjaylytics_api_v12:app --reload --port 8080
"""
from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional, Dict, Tuple
import os, io, json, math

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# Optional deps (yfinance & joblib)
try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

_MODEL = None
_MODEL_PATH = "models/anjaylytics_model.pkl"
_METRICS_PATH = "models/backtest_metrics.json"
try:
    from joblib import load as _joblib_load
except Exception:  # pragma: no cover
    _joblib_load = None

# =============================
# Config
# =============================
APP_NAME = "Anjaylytics API"
APP_VERSION = "1.2.0"

US_WATCH = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","SPY","QQQ"]
# NOTE: yfinance may not support BSE .BT tickers; these may return no data.
BSE_WATCH = ["SEFALANA.BT","CHOPPIES.BT","FNBB.BT","LETSHEGO.BT","BTCL.BT"]

DEFAULT_FEES_BPS = 15
DEFAULT_SLIP_BPS_US = 10
DEFAULT_SLIP_BPS_BSE = 60
DEFAULT_FX_BPS = 50

RISK_THRESHOLDS = {"conservative":0.60, "balanced":0.56, "aggressive":0.53}
MAX_KELLY_CAP = 0.02
MIN_TICKET_P = 150

# =============================
# Schemas
# =============================
class Cash(BaseModel):
    suggested: bool
    reason: Optional[str] = None

class PlanItem(BaseModel):
    symbol: str
    name: str
    market: Literal["US","BSE"]
    price: float
    p: float
    ev: float
    entry: float
    stop: float
    take: float
    size_bwp: int
    rationale: str
    headlines: List[str]

class Plan(BaseModel):
    asof: date
    preset: Literal["Global","Botswana"]
    ideas: List[PlanItem]
    cash: Cash

class Metrics(BaseModel):
    brier: Optional[float]
    hit_rate: Optional[float]
    profit_factor: Optional[float]
    max_dd: Optional[float]
    monthly: List[Dict]

class Presets(BaseModel):
    global_watch: List[str]
    botswana_watch: List[str]

class GuideItem(BaseModel):
    title: str
    steps: List[str]
    platforms: List[Dict[str,str]]

class GuideResponse(BaseModel):
    region: Literal["Botswana","Global"]
    items: List[GuideItem]
    disclaimer: str

# =============================
# App & CORS Configuration
# =============================
app = FastAPI(title=APP_NAME, version=APP_VERSION)

origins = [
    "https://anjaylytics-frontend.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],  # extend later if you add POST
    allow_headers=["*"],
)

# =============================
# Utilities (prices, features, sentiment, EV)
# =============================
_name_cache: Dict[str,str] = {}

from typing import Any

def _fetch_last_price_and_name(ticker: str) -> Optional[Tuple[float,str]]:
    """Return (last_price, name) or None if unavailable."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        price: Optional[float] = None
        # Try fast_info first
        fast = getattr(t, "fast_info", None)
        if fast is not None:
            last_price = getattr(fast, "last_price", None)
            if last_price is not None:
                price = float(last_price)
        # Fallback to small recent history
        if price is None:
            hist = t.history(period="5d", auto_adjust=True)
            if hist is None or hist.empty:
                return None
            price = float(hist["Close"].iloc[-1])
        # Name (with cache)
        name = _name_cache.get(ticker)
        if not name:
            info: Dict[str, Any] = {}
            try:
                info = t.get_info() or {}
            except Exception:
                info = {}
            name = info.get("longName") or info.get("shortName") or ticker
            _name_cache[ticker] = name
        return price, name
    except Exception:
        return None

_LEXICON = {"beat": +1, "upgrade": +1, "record": +1, "miss": -1, "downgrade": -1, "delay": -1}

def _headlines_stub(ticker: str) -> List[str]:
    return [f"{ticker} sentiment placeholder", "Analyst chatter mixed"]

def _sentiment_score(headlines: List[str]) -> float:
    s = 0
    for h in headlines:
        low = h.lower()
        for k, v in _LEXICON.items():
            if k in low:
                s += v
    return max(-3, min(3, s)) / 3.0  # [-1, 1]

import pandas as pd

def _features_for_latest(ticker: str) -> Optional[pd.DataFrame]:
    if yf is None:
        return None
    try:
        df = yf.download(ticker, period="400d", interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty or len(df) < 60:
            return None
        df = df.rename(columns=str.lower)
        x = df.copy()
        x["ret1"] = x["close"].pct_change()
        x["ret5"] = x["close"].pct_change(5)
        x["ret20"] = x["close"].pct_change(20)
        x["vol20"] = x["ret1"].rolling(20).std()
        x["atr10"] = (x["high"] - x["low"]).rolling(10).mean() / x["close"].shift(1)
        x["gap1"] = (x["open"] - x["close"].shift(1)) / x["close"].shift(1)
        x["mom50"] = x["close"].pct_change(50)
        feats = ["ret1","ret5","ret20","vol20","atr10","gap1","mom50"]
        x = x.dropna()
        if x.empty:
            return None
        return x[feats].tail(1)
    except Exception:
        return None

# Heuristic probability (if model missing)
_W0, _W1 = 0.08, 0.85
_CAL_A, _CAL_B = 0.95, 0.02

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _heuristic_p(sent: float) -> float:
    p_raw = _sigmoid(_W0 + _W1 * max(0.0, sent))
    p_cal = _CAL_A * p_raw + _CAL_B
    return max(0.45, min(0.75, p_cal))

def _bps(x: float) -> float:
    return x / 10000.0

def _compute_ev(p: float, up: float, down: float, fees_bps: float, slip_bps: float, fx_bps: float) -> float:
    gross = p * up - (1 - p) * down
    costs = _bps(fees_bps) + _bps(slip_bps) + _bps(fx_bps)
    return gross - costs

def _kelly_fraction(p: float, up: float, down: float) -> float:
    if down <= 0:
        return 0.0
    b = up / down
    f = (b * p - (1 - p)) / b
    return max(0.0, min(f, MAX_KELLY_CAP))

# =============================
# Model loading
# =============================
def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if _joblib_load is None:
        return None
    if not os.path.exists(_MODEL_PATH):
        return None
    try:
        _MODEL = _joblib_load(_MODEL_PATH)
    except Exception:
        _MODEL = None
    return _MODEL


def _predict_p_with_model(ticker: str) -> Optional[float]:
    model = _load_model()
    if model is None:
        return None
    feats = _features_for_latest(ticker)
    if feats is None:
        return None
    try:
        # model should support predict_proba
        p = float(model.predict_proba(feats.values)[:, 1][0])
        return max(0.01, min(0.99, p))
    except Exception:
        return None

# =============================
# Plan generation
# =============================

def _build_plan(
    preset: Literal["Global","Botswana"],
    daily_budget_pula: int,
    bankroll_pula: int,
    risk: Literal["conservative","balanced","aggressive"],
    fees_bps: float,
    fx_bps: float,
) -> Plan:
    threshold = RISK_THRESHOLDS[risk]
    symbols, slip_bps, market = (
        (US_WATCH, DEFAULT_SLIP_BPS_US, "US") if preset == "Global" else (BSE_WATCH, DEFAULT_SLIP_BPS_BSE, "BSE")
    )

    items: List[PlanItem] = []
    for sym in symbols:
        px_nm = _fetch_last_price_and_name(sym)
        if not px_nm:
            continue
        px, nm = px_nm
        # Short-term bands
        up, down = (0.04, 0.025) if market == "US" else (0.025, 0.02)
        entry = round(px, 2)
        stop = round(px * (1 - down), 2)
        take = round(px * (1 + up), 2)

        p_model = _predict_p_with_model(sym)
        if p_model is not None:
            p = p_model
            rationale_src = "trained model"
            headlines = ["Model-driven probability"]
        else:
            headlines = _headlines_stub(sym)
            sent = _sentiment_score(headlines)
            p = _heuristic_p(sent)
            rationale_src = f"heuristic sentiment {sent:+.2f}"

        ev = _compute_ev(p, up, down, fees_bps, slip_bps, fx_bps if market == "US" else 0.0)
        if p < threshold or ev <= 0:
            continue

        frac = _kelly_fraction(p, up, down)
        size_p = min(daily_budget_pula, int(round(frac * bankroll_pula)))
        if size_p < MIN_TICKET_P:
            continue

        rationale = (
            f"p={p:.2f} via {rationale_src}; EV after fees/slippage/FX. "
            f"Bands: -{down*100:.1f}%/+{up*100:.1f}%."
        )
        items.append(
            PlanItem(
                symbol=sym,
                name=nm,
                market=market,
                price=entry,
                p=round(p, 4),
                ev=round(ev, 4),
                entry=entry,
                stop=stop,
                take=take,
                size_bwp=size_p,
                rationale=rationale,
                headlines=headlines,
            )
        )

    items.sort(key=lambda r: r.p, reverse=True)
    return Plan(
        asof=date.today(),
        preset=preset,
        ideas=items,
        cash=Cash(
            suggested=(len(items) == 0),
            reason=("No ideas met the probability & EV gates." if len(items) == 0 else None),
        ),
    )

# =============================
# Endpoints
# =============================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "name": APP_NAME,
        "version": APP_VERSION,
        "model_loaded": bool(_load_model() is not None),
    }


@app.get("/presets", response_model=Presets)
async def presets():
    return Presets(global_watch=US_WATCH, botswana_watch=BSE_WATCH)


@app.get("/plan/today", response_model=Plan, tags=["Plan"])
async def plan_today(
    daily_budget_pula: int = Query(500, ge=50, le=10000),
    bankroll_pula: int = Query(10000, ge=500),
    risk: Literal["conservative","balanced","aggressive"] = Query("balanced"),
    preset: Literal["Global","Botswana"] = Query("Global"),
    fees_bps: float = Query(DEFAULT_FEES_BPS),
    fx_bps: float = Query(DEFAULT_FX_BPS),
):
    return _build_plan(preset, daily_budget_pula, bankroll_pula, risk, fees_bps, fx_bps)


@app.get("/metrics", response_model=Metrics, tags=["Metrics"])
async def metrics():
    if not os.path.exists(_METRICS_PATH):
        return Metrics(brier=None, hit_rate=None, profit_factor=None, max_dd=None, monthly=[])
    with open(_METRICS_PATH, "r") as f:
        m = json.load(f)
    return Metrics(
        brier=m.get("brier_overall"),
        hit_rate=None,
        profit_factor=None,
        max_dd=None,
        monthly=m.get("by_month", []),
    )


@app.get("/reliability", tags=["Metrics"])
async def reliability():
    if not os.path.exists(_METRICS_PATH):
        return {"calibration": []}
    with open(_METRICS_PATH, "r") as f:
        m = json.load(f)
    return {"calibration": m.get("calibration", [])}


@app.get("/guide/botswana", response_model=GuideResponse, tags=["Guide"])
async def guide_botswana():
    items = [
        GuideItem(
            title="Local shares on Botswana Stock Exchange (BSE)",
            steps=[
                "Choose a licensed BSE broker.",
                "Open a CSDB account via your broker (ID, proof of residence/income).",
                "Fund your broker account.",
                "Place orders during BSE session.",
                "Track holdings & dividends; keep records for tax.",
            ],
            platforms=[
                {"name":"Imara Capital Securities","type":"Broker","region":"Botswana","note":"Full-service local broker"},
                {"name":"Motswedi Securities","type":"Broker","region":"Botswana","note":"Local research & dealing"},
                {"name":"Stockbrokers Botswana","type":"Broker","region":"Botswana","note":"BSE member"},
            ],
        ),
        GuideItem(
            title="Global stocks & ETFs (USD)",
            steps=[
                "Open an international broker that accepts Botswana residents.",
                "Fund via SWIFT in USD.",
                "Enable fractional shares for small tickets.",
                "Trade during US session (~15:30–22:00 Gaborone).",
                "Export trades to CSV for records & tax.",
            ],
            platforms=[
                {"name":"Interactive Brokers (IBKR)","type":"Broker","region":"Global","note":"Fractional shares; broad access"},
                {"name":"EasyEquities (USD)","type":"Broker","region":"International","note":"USD account for non-SA residents"},
            ],
        ),
    ]
    return GuideResponse(
        region="Botswana",
        items=items,
        disclaimer=(
            "Educational-only signals. Verify brokerage availability/fees and local regulations. "
            "Personalized financial advice may require licensing under NBFIRA in Botswana."
        ),
    )


@app.get("/trade/export", response_class=PlainTextResponse, tags=["Trade"])
async def trade_export(
    daily_budget_pula: int = Query(500, ge=50, le=10000),
    bankroll_pula: int = Query(10000, ge=500),
    risk: Literal["conservative","balanced","aggressive"] = Query("balanced"),
    preset: Literal["Global","Botswana"] = Query("Global"),
    fees_bps: float = Query(DEFAULT_FEES_BPS),
    fx_bps: float = Query(DEFAULT_FX_BPS),
):
    plan = _build_plan(preset, daily_budget_pula, bankroll_pula, risk, fees_bps, fx_bps)
    rows = [["date","preset","symbol","name","market","entry","stop","take","probability","ev_pct","size_bwp"]]
    for i in plan.ideas:
        rows.append([
            str(plan.asof),
            plan.preset,
            i.symbol,
            i.name,
            i.market,
            f"{i.entry}",
            f"{i.stop}",
            f"{i.take}",
            f"{i.p:.3f}",
            f"{i.ev*100:.2f}",
            str(i.size_bwp),
        ])
    buf = io.StringIO()
    for r in rows:
        buf.write(",".join(r) + "\n")
    return PlainTextResponse(content=buf.getvalue(), media_type="text/csv")
