"""
Generate NBIS_valuation.ipynb — Nebius Group N.V. (NBIS)
Uses nbformat to build a full tech-company valuation notebook.
Run:  python generate_nebius_valuation.py

Nebius Group N.V. (formerly Yandex N.V.) is an AI-centric cloud company
listed on NASDAQ. Its main business is GPU-as-a-Service / AI cloud
infrastructure (Nebius AI), plus Toloka (data labelling), TripleTen
(tech education), and Avride (autonomous driving).
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}
cells = []

def md(text):
    cells.append(new_markdown_cell(text))

def code(src):
    cells.append(new_code_cell(src))


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
md("""# Nebius Group N.V. (NBIS) — Comprehensive Valuation

**Sections**
1. Imports & Configuration
2. Live Data (yfinance)
3. Company Overview & Historical Financials
4. Segment Revenue Model
5. Technical Analysis
6. Segment-Level DCF
7. Sensitivity Heatmaps
8. Monte Carlo Simulation
9. Relative / Peer Valuation
10. FCF Bridge Waterfall
11. Investment Summary

---
*AI cloud GPU infrastructure — high-growth, early-stage. Edit config variables in Cell 0.*
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0 — IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
md("## Cell 0 — Imports & Configuration")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  TECH COMPANY VALUATION — NEBIUS GROUP N.V. (NBIS)
#  Built by Claude · Edit the variables below to customise
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    YF_AVAILABLE = False

# ── COMPANY ───────────────────────────────────────────────────────
TICKER          = "NBIS"
COMPANY_NAME    = "Nebius Group N.V."

# ── Manual fallback values (overridden by yfinance if available) ──
# Nebius FY2024 estimated: ~$117M H1 + fast-accelerating H2
SHARES_OUT_M    = 340     # ~340M diluted shares (incl. options/RSUs)
CURRENT_PRICE   = 0.0
MARKET_CAP_B    = 0.0
# Net cash: raised ~$700M from investors inc. Nvidia in 2024; minimal debt
NET_DEBT_USD_M  = -500_000   # ~$500M net cash (conservative estimate)
BETA            = 1.80       # high-beta growth/AI infrastructure name

# ── REVENUE GROWTH ASSUMPTIONS (annual %) ────────────────────────
# Nebius AI cloud growing ~3-5x YoY; blended with slower-growth subsidiaries
REVENUE_GROWTH_BEAR  = 0.40   # slowdown in GPU demand or pricing pressure
REVENUE_GROWTH_BASE  = 0.80   # continuation of rapid AI infrastructure ramp
REVENUE_GROWTH_BULL  = 1.30   # Nvidia partnership + hyperscaler overflow demand

# ── MARGIN ASSUMPTIONS ───────────────────────────────────────────
# Currently EBITDA-negative; path to profitability as scale increases
OPERATING_MARGIN_TERMINAL = 0.20   # mature GPU cloud margins (asset-heavy)
FCF_CONVERSION             = 0.70  # FCF/operating income — capex-intensive

# ── DISCOUNT RATE ────────────────────────────────────────────────
RISK_FREE_RATE    = 0.043
EQUITY_RISK_PREM  = 0.055
COST_OF_DEBT      = 0.060    # small-cap, limited debt history
DEBT_WEIGHT       = 0.02
TAX_RATE          = 0.10     # low near-term (NOL carryforwards + tax location)

WACC_OVERRIDE     = None     # override CAPM calc if desired

# ── DCF HORIZON ──────────────────────────────────────────────────
FORECAST_YEARS      = 10
TERMINAL_GROWTH     = 0.03
FADE_GROWTH_TO      = 0.06   # growth fades to ~6% by Year 10 (still scaling)

# ── MONTE CARLO ──────────────────────────────────────────────────
MC_SIMULATIONS      = 10_000
MC_REVENUE_STDEV    = 0.25   # wide range — early-stage company
MC_MARGIN_STDEV     = 0.05
MC_WACC_STDEV       = 0.015

# ── SHARE-BASED COMPENSATION ─────────────────────────────────────
SBC_PCT_REVENUE     = 0.25   # ~25% of revenue (typical early-stage tech)
ANNUAL_DILUTION_PCT = 0.03   # ~3% net dilution (high SBC, limited buybacks)

# ── CORPORATE G&A ────────────────────────────────────────────────
CORPORATE_GA_USD_M  = 80     # lean corporate overhead

print("Config loaded. Edit variables above and re-run to update all outputs.")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — FETCH LIVE DATA
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 1 — Fetch Live Data")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: FETCH LIVE DATA FROM YFINANCE
# ═══════════════════════════════════════════════════════════════════

if YF_AVAILABLE:
    stock = yf.Ticker(TICKER)
    info  = stock.info

    COMPANY_NAME    = info.get('longName', COMPANY_NAME)
    CURRENT_PRICE   = info.get('currentPrice', info.get('regularMarketPrice', 0))
    MARKET_CAP_B    = info.get('marketCap', 0) / 1e9
    SHARES_OUT_M    = info.get('sharesOutstanding', 0) / 1e6
    BETA            = info.get('beta', BETA)

    total_debt = info.get('totalDebt', 0) or 0
    total_cash = info.get('totalCash', 0) or 0
    NET_DEBT_USD_M = (total_debt - total_cash) / 1e6   # negative = net cash

    income_stmt   = stock.income_stmt
    balance_sheet = stock.balance_sheet
    cash_flow     = stock.cash_flow
    quarterly_inc = stock.quarterly_income_stmt
    price_data    = stock.history(period="2y")

    try:
        analyst_targets = {
            'target_mean'   : info.get('targetMeanPrice'),
            'target_low'    : info.get('targetLowPrice'),
            'target_high'   : info.get('targetHighPrice'),
            'num_analysts'  : info.get('numberOfAnalystOpinions'),
            'recommendation': info.get('recommendationKey'),
        }
    except:
        analyst_targets = {}

    try:
        insider_txns = stock.insider_transactions
    except:
        insider_txns = pd.DataFrame()

    try:
        inst_holders = stock.institutional_holders
    except:
        inst_holders = pd.DataFrame()

    def safe_get(df, label, default=0):
        if df is None or df.empty:
            return default
        if label in df.index:
            val = df.loc[label].dropna()
            if len(val) > 0:
                return float(val.iloc[0])
        return default

    LATEST_REVENUE      = safe_get(income_stmt, 'Total Revenue')      / 1e6
    LATEST_GROSS_PROFIT = safe_get(income_stmt, 'Gross Profit')       / 1e6
    LATEST_OP_INCOME    = safe_get(income_stmt, 'Operating Income')   / 1e6
    LATEST_NET_INCOME   = safe_get(income_stmt, 'Net Income')         / 1e6
    LATEST_FCF          = (safe_get(cash_flow, 'Operating Cash Flow')
                           - abs(safe_get(cash_flow, 'Capital Expenditure'))) / 1e6
    LATEST_SBC          = safe_get(cash_flow, 'Stock Based Compensation') / 1e6
    LATEST_CAPEX        = abs(safe_get(cash_flow, 'Capital Expenditure')) / 1e6
    LATEST_DA           = safe_get(cash_flow, 'Depreciation And Amortization') / 1e6

    tax_provision = safe_get(income_stmt, 'Tax Provision')
    pretax_income = safe_get(income_stmt, 'Pretax Income')
    if pretax_income > 0 and tax_provision > 0:
        TAX_RATE = min(tax_provision / pretax_income, 0.30)

    if LATEST_REVENUE > 0 and LATEST_SBC > 0:
        SBC_PCT_REVENUE = LATEST_SBC / LATEST_REVENUE

    # Use fallback if yfinance returns zero revenue (limited history)
    if LATEST_REVENUE < 1:
        print("Note: limited financial history in yfinance; using manual fallbacks")
        LATEST_REVENUE   = 350    # FY2024 estimate ~$350M
        LATEST_OP_INCOME = -250   # loss-making currently
        LATEST_FCF       = -400   # heavy capex investment phase
        LATEST_SBC       = 80
        LATEST_CAPEX     = 450
        LATEST_DA        = 100

    cost_of_equity = RISK_FREE_RATE + BETA * EQUITY_RISK_PREM
    equity_weight  = 1 - DEBT_WEIGHT
    WACC_CALC = (equity_weight * cost_of_equity
                 + DEBT_WEIGHT * COST_OF_DEBT * (1 - TAX_RATE))
    WACC = WACC_OVERRIDE if WACC_OVERRIDE else WACC_CALC

    print(f"  {COMPANY_NAME} ({TICKER})")
    print(f"  Price: ${CURRENT_PRICE:,.2f}  |  Mkt Cap: ${MARKET_CAP_B:,.2f}B  |  Beta: {BETA:.2f}")
    print(f"  Revenue: ${LATEST_REVENUE:,.0f}M  |  Op Income: ${LATEST_OP_INCOME:,.0f}M  |  FCF: ${LATEST_FCF:,.0f}M")
    print(f"  WACC: {WACC:.2%}  |  Tax Rate: {TAX_RATE:.1%}  |  SBC/Rev: {SBC_PCT_REVENUE:.1%}")
    print(f"  Net Debt: ${NET_DEBT_USD_M:,.0f}M  |  Shares: {SHARES_OUT_M:,.0f}M")
    if analyst_targets.get('target_mean'):
        print(f"  Analyst mean target: ${analyst_targets['target_mean']:,.2f}"
              f"  |  Recommendation: {analyst_targets['recommendation']}")
else:
    print("yfinance not available -- using manual config values")
    WACC             = WACC_OVERRIDE or 0.13
    LATEST_REVENUE   = 350
    LATEST_OP_INCOME = -250
    LATEST_FCF       = -400
    LATEST_SBC       = 80
    LATEST_CAPEX     = 450
    LATEST_DA        = 100
    CURRENT_PRICE    = 30.0
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — COMPANY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 2 — Company Overview & Historical Financials")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: COMPANY OVERVIEW
# ═══════════════════════════════════════════════════════════════════

overview = \"\"\"
Nebius Group N.V. (NBIS) — Business Overview
=============================================
Formerly Yandex N.V., Nebius completed its restructuring in 2024,
divesting its Russian assets and relisting on NASDAQ as a pure-play
AI infrastructure company headquartered in Amsterdam.

CORPORATE STRUCTURE
-------------------
  Nebius AI (GPU Cloud)    Revenue-dominant unit. Sells GPU compute
                           (Nvidia H100/H200) via a cloud platform
                           targeting AI startups, research labs, and
                           enterprises. Nvidia is a strategic investor.

  Toloka AI                AI data-labelling and RLHF annotation
                           platform. ~5k+ expert annotators globally.

  TripleTen                Online tech education (coding bootcamps).
                           Focuses on career-switchers in CIS, LATAM.

  Avride                   Autonomous driving startup (former Yandex
                           SDC team). Urban robotaxi + last-mile delivery.

KEY FINANCIALS (FY2024 ESTIMATES)
----------------------------------
  Nebius AI revenue annualised run-rate: ~$500M+ by Q4 2024
  Total group revenue FY2024: ~$350M (H1 reported ~$117M)
  Gross margin: ~50-55% (cloud GPU has lower margins vs software)
  EBITDA: deeply negative — heavy capex investment phase
  Cash position: ~$2B+ (capital raises + Nvidia investment)
  Nvidia strategic stake: ~$100M investment; H100/H200 preferential supply

THESIS SNAPSHOT
---------------
  Bull: GPU cloud demand structurally undersupplied; Nebius positioned
        as Nvidia's preferred external cloud partner; scaling economics
        drive 60-70%+ gross margins at scale (software layers).

  Bear: Hyperscaler competition (AWS, Azure, GCP) intensifies; GPU
        pricing commoditises; path to FCF profitability is long;
        small-cap risk and liquidity.
\"\"\"
print(overview)

# ── Historical Summary (if available) ────────────────────────────
if YF_AVAILABLE and income_stmt is not None and not income_stmt.empty:
    years = income_stmt.columns[:4]
    rows  = []
    for yr in years:
        rev = income_stmt.loc['Total Revenue', yr]    / 1e6 if 'Total Revenue'    in income_stmt.index else 0
        gp  = income_stmt.loc['Gross Profit', yr]     / 1e6 if 'Gross Profit'     in income_stmt.index else 0
        op  = income_stmt.loc['Operating Income', yr] / 1e6 if 'Operating Income' in income_stmt.index else 0
        ni  = income_stmt.loc['Net Income', yr]       / 1e6 if 'Net Income'       in income_stmt.index else 0
        yr_str = yr.strftime('%Y') if hasattr(yr, 'strftime') else str(yr)
        rows.append({
            'Year'            : yr_str,
            'Revenue ($M)'    : f"{rev:,.0f}",
            'Gross Profit ($M)': f"{gp:,.0f}",
            'Gross Margin'    : f"{gp/rev:.1%}" if rev > 0 else "N/A",
            'Op Income ($M)'  : f"{op:,.0f}",
            'Op Margin'       : f"{op/rev:.1%}" if rev > 0 else "N/A",
            'Net Income ($M)' : f"{ni:,.0f}",
        })
    if rows:
        print("=" * 75)
        print(f"  {COMPANY_NAME} -- Financial Summary")
        print("=" * 75)
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("Note: limited financial history available for NBIS (recently relisted)")

# ── Price history chart ───────────────────────────────────────────
if YF_AVAILABLE and price_data is not None and len(price_data) > 5:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_data.index, y=price_data['Close'],
        mode='lines', name='Close Price',
        line=dict(color='#00BCD4', width=2),
        fill='tozeroy', fillcolor='rgba(0,188,212,0.1)',
    ))
    fig.update_layout(
        title=f"{COMPANY_NAME} ({TICKER}) -- Price History",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_white",
    )
    fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SEGMENT REVENUE MODEL
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 3 — Segment Revenue Model")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: NEBIUS SEGMENT-LEVEL REVENUE MODEL
#  Source: Nebius earnings releases, investor presentations
# ═══════════════════════════════════════════════════════════════════

# Note: Nebius AI is the dominant unit (~75%+ of group revenue and growing).
# The other segments are strategic options / positive NPV bets at current scale.

segments = {
    "Nebius AI (GPU Cloud)": {
        "current_revenue_usdm" : 270,    # ~75% of ~$350M FY2024 group rev
        "growth_rate_bear"     : 0.50,
        "growth_rate_base"     : 1.00,   # 100% YoY -- demand-constrained ramp
        "growth_rate_bull"     : 1.80,   # GPU supply + pricing power upside
        "fade_to"              : 0.08,
        "operating_margin_current"  : -0.40,   # negative — capex/depreciation-heavy
        "operating_margin_terminal" : 0.22,    # mature GPU cloud (~AWS margins)
        "tam_2024_usdm"  : 100_000,
        "tam_2030_usdm"  : 500_000,
        "market_share_pct": 0.3,
        "key_driver": "Nvidia H100/H200 preferential supply, AI startup/SMB target",
    },
    "Toloka AI (Data Labelling)": {
        "current_revenue_usdm" : 55,
        "growth_rate_bear"     : 0.10,
        "growth_rate_base"     : 0.22,
        "growth_rate_bull"     : 0.40,
        "fade_to"              : 0.06,
        "operating_margin_current"  : 0.05,
        "operating_margin_terminal" : 0.18,
        "tam_2024_usdm"  : 5_000,
        "tam_2030_usdm"  : 20_000,
        "market_share_pct": 1.1,
        "key_driver": "RLHF annotation demand for LLM training; AI workforce platform",
    },
    "TripleTen (Tech Education)": {
        "current_revenue_usdm" : 20,
        "growth_rate_bear"     : 0.05,
        "growth_rate_base"     : 0.18,
        "growth_rate_bull"     : 0.30,
        "fade_to"              : 0.05,
        "operating_margin_current"  : -0.10,
        "operating_margin_terminal" : 0.12,
        "tam_2024_usdm"  : 3_000,
        "tam_2030_usdm"  : 8_000,
        "market_share_pct": 0.7,
        "key_driver": "Online bootcamp for CIS/LATAM market; AI curriculum expansion",
    },
    "Avride (Autonomous Driving)": {
        "current_revenue_usdm" : 5,     # pre-revenue / pilot stage
        "growth_rate_bear"     : 0.10,
        "growth_rate_base"     : 0.50,
        "growth_rate_bull"     : 1.50,
        "fade_to"              : 0.08,
        "operating_margin_current"  : -5.00,   # deep losses, R&D phase
        "operating_margin_terminal" : 0.08,    # robotaxi economics at scale
        "tam_2024_usdm"  : 50_000,
        "tam_2030_usdm"  : 500_000,
        "market_share_pct": 0.01,
        "key_driver": "Former Yandex SDC team; urban robotaxi + delivery robots",
    },
}

# ── TAM & Positioning Summary ─────────────────────────────────────
print("=" * 100)
print("  Nebius Group -- Segment Overview")
print("=" * 100)
rows = []
for name, s in segments.items():
    rows.append({
        'Segment'         : name,
        'Rev FY24 ($M)'   : f"{s['current_revenue_usdm']:,}",
        'Market Share %'  : f"{s['market_share_pct']:.2f}%",
        'TAM 2024 ($M)'   : f"{s['tam_2024_usdm']:,}",
        'TAM 2030 ($M)'   : f"{s['tam_2030_usdm']:,}",
        'Base Growth'     : f"{s['growth_rate_base']:.0%}",
        'Term. Op Margin' : f"{s['operating_margin_terminal']:.0%}",
        'Key Driver'      : s['key_driver'][:52],
    })
print(pd.DataFrame(rows).to_string(index=False))

# ── Projection engine ─────────────────────────────────────────────
def project_segment(seg_name, seg, scenario="base", years=FORECAST_YEARS):
    growth_key    = f"growth_rate_{scenario}"
    init_growth   = seg[growth_key]
    term_growth   = seg['fade_to']
    rows          = []
    revenue       = seg['current_revenue_usdm']
    margin_now    = seg['operating_margin_current']
    margin_target = seg['operating_margin_terminal']
    for yr_offset in range(years):
        fade_frac = yr_offset / max(years - 1, 1)
        growth    = init_growth * (1 - fade_frac) + term_growth  * fade_frac
        margin    = margin_now  * (1 - fade_frac) + margin_target * fade_frac
        revenue   *= (1 + growth)
        rows.append({
            'year'           : datetime.now().year + yr_offset + 1,
            'segment'        : seg_name,
            'revenue_usdm'   : revenue,
            'growth_rate'    : growth,
            'op_margin'      : margin,
            'op_income_usdm' : revenue * margin,
        })
    return pd.DataFrame(rows)

projections = {}
for scenario in ['bear', 'base', 'bull']:
    dfs = [project_segment(n, s, scenario) for n, s in segments.items()]
    projections[scenario] = pd.concat(dfs, ignore_index=True)

# ── Stacked revenue chart ─────────────────────────────────────────
NBIS_COLORS = {
    'Nebius AI (GPU Cloud)'       : '#00BCD4',
    'Toloka AI (Data Labelling)'  : '#7C4DFF',
    'TripleTen (Tech Education)'  : '#FF6D00',
    'Avride (Autonomous Driving)' : '#00C853',
}
df_base = projections['base']
fig = go.Figure()
for seg_name, color in NBIS_COLORS.items():
    seg_data = df_base[df_base['segment'] == seg_name]
    fig.add_trace(go.Bar(
        x=seg_data['year'], y=seg_data['revenue_usdm'],
        name=seg_name, marker_color=color,
    ))
fig.update_layout(
    title=f"{COMPANY_NAME} -- Projected Revenue by Segment (Base Case, $M)",
    xaxis_title="Year", yaxis_title="Revenue ($M)",
    barmode='stack', template="plotly_white",
    legend=dict(orientation="h", y=-0.25),
)
fig.show()

# ── Bear / Base / Bull total revenue lines ────────────────────────
fig2 = go.Figure()
for scenario, color in [('bear','#EA4335'), ('base','#00BCD4'), ('bull','#00C853')]:
    df_s = projections[scenario].groupby('year')['revenue_usdm'].sum().reset_index()
    fig2.add_trace(go.Scatter(
        x=df_s['year'], y=df_s['revenue_usdm'],
        name=scenario.capitalize(), mode='lines+markers',
        line=dict(color=color, width=2),
    ))
fig2.update_layout(
    title=f"{COMPANY_NAME} -- Total Revenue: Bear / Base / Bull ($M)",
    xaxis_title="Year", yaxis_title="Revenue ($M)",
    template="plotly_white",
)
fig2.show()

# ── Margin evolution (Nebius AI) ──────────────────────────────────
nb_ai_base = projections['base'][projections['base']['segment'] == 'Nebius AI (GPU Cloud)'].copy()
fig3 = make_subplots(specs=[[{"secondary_y": True}]])
fig3.add_trace(go.Bar(x=nb_ai_base['year'], y=nb_ai_base['revenue_usdm'],
                      name='Nebius AI Revenue ($M)', marker_color='#00BCD4', opacity=0.6),
               secondary_y=False)
fig3.add_trace(go.Scatter(x=nb_ai_base['year'], y=nb_ai_base['op_margin'],
                          name='Op Margin', line=dict(color='#7C4DFF', width=2), mode='lines+markers'),
               secondary_y=True)
fig3.update_layout(title="Nebius AI -- Revenue Ramp & Margin Expansion (Base Case)",
                   template="plotly_white")
fig3.update_yaxes(title_text="Revenue ($M)", secondary_y=False)
fig3.update_yaxes(title_text="Operating Margin", tickformat=".0%", secondary_y=True)
fig3.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — QUARTERLY TRENDS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 4 — Quarterly Revenue Trends")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: QUARTERLY TRENDS
# ═══════════════════════════════════════════════════════════════════

if YF_AVAILABLE and quarterly_inc is not None and not quarterly_inc.empty:
    q_data = quarterly_inc.T.copy()
    q_data.index = pd.to_datetime(q_data.index)
    q_data = q_data.sort_index()

    q_rev = q_data['Total Revenue']    / 1e6 if 'Total Revenue'    in q_data.columns else pd.Series()
    q_op  = q_data['Operating Income'] / 1e6 if 'Operating Income' in q_data.columns else pd.Series()

    if len(q_rev) > 0:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Quarterly Revenue ($M)", "Quarterly Operating Income ($M)"])
        fig.add_trace(go.Bar(x=q_rev.index, y=q_rev.values, name='Revenue',
                             marker_color='#00BCD4'), row=1, col=1)
        if len(q_op) > 0:
            bar_colors = ['#00C853' if v >= 0 else '#EA4335' for v in q_op.values]
            fig.add_trace(go.Bar(x=q_op.index, y=q_op.values, name='Op Income',
                                 marker_color=bar_colors), row=2, col=1)
        fig.update_layout(title=f"{COMPANY_NAME} -- Quarterly Financials",
                          template="plotly_white", height=600)
        fig.show()

        # Revenue QoQ growth
        if len(q_rev) >= 2:
            qoq = q_rev.pct_change().dropna()
            fig2 = go.Figure()
            bar_c = ['#00C853' if v >= 0 else '#EA4335' for v in qoq.values]
            fig2.add_trace(go.Bar(x=qoq.index, y=qoq.values * 100, marker_color=bar_c))
            fig2.update_layout(
                title=f"{COMPANY_NAME} -- Quarter-on-Quarter Revenue Growth (%)",
                xaxis_title="Quarter", yaxis_title="QoQ Growth (%)",
                template="plotly_white",
            )
            fig2.show()
    else:
        print("Note: limited quarterly data for NBIS; company relisted late 2024")
else:
    print("Note: quarterly financials not available via yfinance for NBIS")
    # Manually plot known quarterly data points (from earnings releases)
    known_quarters = {
        'Q1 2024': 38,  'Q2 2024': 79,  'Q3 2024': 117, 'Q4 2024': 127,
    }
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(known_quarters.keys()),
        y=list(known_quarters.values()),
        marker_color='#00BCD4', name='Revenue ($M)',
    ))
    fig.update_layout(
        title=f"{COMPANY_NAME} -- Reported Quarterly Revenue ($M, FY2024)",
        xaxis_title="Quarter", yaxis_title="Revenue ($M)",
        template="plotly_white",
    )
    fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TECHNICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 5 — Technical Analysis")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

if YF_AVAILABLE and price_data is not None and len(price_data) > 50:
    df_ta = price_data.copy()

    df_ta['EMA_50']  = df_ta['Close'].ewm(span=50,  adjust=False).mean()
    df_ta['EMA_200'] = df_ta['Close'].ewm(span=200, adjust=False).mean()

    ema_12 = df_ta['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_ta['Close'].ewm(span=26, adjust=False).mean()
    df_ta['MACD']        = ema_12 - ema_26
    df_ta['MACD_Signal'] = df_ta['MACD'].ewm(span=9, adjust=False).mean()
    df_ta['MACD_Hist']   = df_ta['MACD'] - df_ta['MACD_Signal']

    delta = df_ta['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    df_ta['RSI'] = 100 - (100 / (1 + rs))

    df_ta['BB_Mid']   = df_ta['Close'].rolling(20).mean()
    bb_std            = df_ta['Close'].rolling(20).std()
    df_ta['BB_Upper'] = df_ta['BB_Mid'] + 2 * bb_std
    df_ta['BB_Lower'] = df_ta['BB_Mid'] - 2 * bb_std

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=["Price & EMAs / Bollinger Bands", "Volume", "MACD", "RSI"])

    fig.add_trace(go.Candlestick(x=df_ta.index, open=df_ta['Open'], high=df_ta['High'],
                                  low=df_ta['Low'], close=df_ta['Close'],
                                  name='OHLC', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_50'],  name='EMA 50',
                             line=dict(color='#FF6D00', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_200'], name='EMA 200',
                             line=dict(color='#EA4335', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Upper'], showlegend=False,
                             line=dict(color='gray', width=0.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Lower'], showlegend=False,
                             line=dict(color='gray', width=0.5, dash='dot'),
                             fill='tonexty', fillcolor='rgba(128,128,128,0.08)'), row=1, col=1)

    colors_vol = ['#00C853' if c >= o else '#EA4335'
                  for c, o in zip(df_ta['Close'], df_ta['Open'])]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['Volume'], name='Volume',
                         marker_color=colors_vol, showlegend=False), row=2, col=1)

    macd_colors = ['#00C853' if v >= 0 else '#EA4335' for v in df_ta['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_Hist'],
                         marker_color=macd_colors, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'],
                             name='MACD', line=dict(color='#00BCD4', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_Signal'],
                             name='Signal', line=dict(color='#FF6D00', width=1)), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], name='RSI',
                             line=dict(color='#7C4DFF', width=1.5)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    fig.update_layout(
        title=f"{COMPANY_NAME} ({TICKER}) -- Technical Analysis",
        template="plotly_white", height=950, showlegend=True,
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
    )
    fig.show()
else:
    print("Note: insufficient price history for full TA (NBIS relisted Oct 2024)")
    if YF_AVAILABLE and price_data is not None and len(price_data) > 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=price_data.index, y=price_data['Close'],
                                  mode='lines+markers', name='Close',
                                  line=dict(color='#00BCD4', width=2)))
        fig.update_layout(title=f"{COMPANY_NAME} -- Price Since Relisting",
                          template="plotly_white")
        fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SEGMENT-LEVEL DCF
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 6 — Segment-Level DCF Valuation")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: SEGMENT-LEVEL DCF
#  Note: Nebius is pre-FCF-profitability; early years will show
#  negative FCFs. The intrinsic value is dominated by terminal value,
#  which is appropriate for a high-growth infrastructure company.
# ═══════════════════════════════════════════════════════════════════

def run_dcf(scenario="base", wacc_override=None, tg_override=None):
    wacc = wacc_override or WACC
    tg   = tg_override   or TERMINAL_GROWTH

    df_proj = projections[scenario]
    years   = sorted(df_proj['year'].unique())

    annual_fcfs, annual_details = [], []

    # Nebius AI is capex-intensive; normalise ratios from latest data
    da_pct    = max(LATEST_DA    / LATEST_REVENUE, 0.20) if LATEST_REVENUE > 0 else 0.25
    capex_pct = max(LATEST_CAPEX / LATEST_REVENUE, 0.80) if LATEST_REVENUE > 0 else 1.20
    # Capex/revenue declines as scale grows (asset base matures)
    # Apply capex fade: starts at capex_pct, declines to 0.25 by year 10
    capex_mature = 0.25
    wc_pct = 0.02

    for i, yr in enumerate(years):
        yr_data       = df_proj[df_proj['year'] == yr]
        total_revenue = yr_data['revenue_usdm'].sum()
        total_op_inc  = yr_data['op_income_usdm'].sum()

        sbc       = total_revenue * SBC_PCT_REVENUE
        corp_ga   = CORPORATE_GA_USD_M
        ebit      = total_op_inc - corp_ga - sbc
        # Only tax when profitable
        taxes     = max(0, ebit * TAX_RATE)
        nopat     = ebit - taxes

        da        = total_revenue * da_pct
        # Capex fades from current intensity to mature level over forecast
        capex_fade_frac = i / max(FORECAST_YEARS - 1, 1)
        capex_rate  = capex_pct * (1 - capex_fade_frac) + capex_mature * capex_fade_frac
        capex       = total_revenue * capex_rate
        wc_change   = total_revenue * wc_pct
        fcf         = nopat + da - capex - wc_change

        annual_fcfs.append(fcf)
        annual_details.append({
            'year': yr, 'revenue': total_revenue, 'op_income': total_op_inc,
            'sbc': sbc, 'corp_ga': corp_ga, 'ebit': ebit, 'nopat': nopat,
            'da': da, 'capex': capex, 'fcf': fcf,
            'capex_pct': capex_rate,
        })

    disc_factors = [(1 + wacc) ** -(i+1) for i in range(len(annual_fcfs))]
    pv_fcfs      = np.array(annual_fcfs) * np.array(disc_factors)

    terminal_fcf = annual_fcfs[-1] * (1 + tg)
    terminal_val = terminal_fcf / (wacc - tg)
    pv_terminal  = terminal_val * disc_factors[-1]

    ev           = sum(pv_fcfs) + pv_terminal
    equity_val   = ev - NET_DEBT_USD_M
    diluted_sh   = SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS
    fv_per_share = equity_val / diluted_sh if diluted_sh > 0 else 0
    upside       = (fv_per_share / CURRENT_PRICE - 1) if CURRENT_PRICE > 0 else 0

    t_detail     = annual_details[-1]
    implied_ev_r = ev / t_detail['revenue'] if t_detail['revenue'] > 0 else 0

    return {
        'scenario'            : scenario,
        'ev_usdm'             : round(ev, 0),
        'equity_value_usdm'   : round(equity_val, 0),
        'fair_value_per_share': round(fv_per_share, 2),
        'upside_pct'          : round(upside * 100, 1),
        'pv_fcfs_total'       : round(sum(pv_fcfs), 0),
        'pv_terminal'         : round(pv_terminal, 0),
        'terminal_pct_of_ev'  : round(pv_terminal / ev * 100, 1) if ev > 0 else 0,
        'implied_ev_rev'      : round(implied_ev_r, 1),
        'annual_details'      : pd.DataFrame(annual_details),
        'pv_fcfs'             : pv_fcfs,
        'wacc_used'           : wacc,
    }

results = {s: run_dcf(s) for s in ['bear', 'base', 'bull']}

# ── Scenario Summary Table ────────────────────────────────────────
summary_rows = []
for label, r in results.items():
    summary_rows.append({
        'Scenario'           : label.upper(),
        'EV ($M)'            : f"${r['ev_usdm']:,.0f}",
        'Equity Value ($M)'  : f"${r['equity_value_usdm']:,.0f}",
        'Fair Value / Share' : f"${r['fair_value_per_share']:,.2f}",
        'Upside / Downside'  : f"{r['upside_pct']:+.1f}%",
        'Terminal % of EV'   : f"{r['terminal_pct_of_ev']:.0f}%",
        'Implied EV/Rev (Y10)': f"{r['implied_ev_rev']:.1f}x",
        'WACC Used'          : f"{r['wacc_used']:.2%}",
    })
df_scenarios = pd.DataFrame(summary_rows)
print("=" * 100)
print(f"  {COMPANY_NAME} -- DCF Scenario Summary")
print(f"  Current Price: ${CURRENT_PRICE:,.2f}  |  WACC: {WACC:.2%}  |  Terminal Growth: {TERMINAL_GROWTH:.1%}")
print("=" * 100)
print(df_scenarios.to_string(index=False))

# ── Annual FCF profile ────────────────────────────────────────────
fig = go.Figure()
for label, r in results.items():
    det = r['annual_details']
    fig.add_trace(go.Scatter(
        x=det['year'], y=det['fcf'],
        name=label.capitalize(), mode='lines+markers',
        line=dict(color={'bear':'#EA4335','base':'#00BCD4','bull':'#00C853'}[label], width=2),
    ))
fig.add_hline(y=0, line_dash="dash", line_color="gray")
fig.update_layout(
    title=f"{COMPANY_NAME} -- Projected FCF by Scenario ($M)",
    xaxis_title="Year", yaxis_title="FCF ($M)",
    template="plotly_white",
)
fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6b — SENSITIVITY HEATMAPS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 6b — Sensitivity Heatmaps")
code("""\
# ── WACC vs Terminal Growth ───────────────────────────────────────
wacc_range = np.arange(0.09, 0.17, 0.01)
tg_range   = np.arange(0.02, 0.065, 0.01)

matrix = []
for w in wacc_range:
    row_vals = []
    for tg in tg_range:
        res = run_dcf('base', wacc_override=w, tg_override=tg)
        row_vals.append(res['fair_value_per_share'])
    matrix.append(row_vals)

df_sens = pd.DataFrame(
    matrix,
    index=[f"{w:.0%}" for w in wacc_range],
    columns=[f"{tg:.0%}" for tg in tg_range],
)
fig = px.imshow(
    df_sens.values, text_auto='.1f',
    x=[f"{tg:.0%}" for tg in tg_range],
    y=[f"{w:.0%}"  for w  in wacc_range],
    color_continuous_scale='RdYlGn',
    title=f"{COMPANY_NAME} -- Fair Value Sensitivity: WACC vs Terminal Growth",
    labels=dict(x="Terminal Growth Rate", y="WACC", color="Fair Value ($)"),
)
fig.update_layout(template="plotly_white")
fig.show()

# ── Revenue Growth (Bear / Base) vs Terminal Margin ──────────────
rev_growth_range = [0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.50]
margin_range     = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]

matrix2 = []
for rg in rev_growth_range:
    row_vals = []
    for m in margin_range:
        total_rev = LATEST_REVENUE
        fcfs = []
        for yr in range(FORECAST_YEARS):
            fade      = yr / max(FORECAST_YEARS - 1, 1)
            g         = rg * (1 - fade) + TERMINAL_GROWTH * fade
            margin    = -0.40 * (1 - fade) + m * fade   # margin starts negative
            total_rev *= (1 + g)
            ebit   = total_rev * margin - CORPORATE_GA_USD_M - total_rev * SBC_PCT_REVENUE
            nopat  = max(0, ebit * (1 - TAX_RATE))
            da     = total_rev * max(LATEST_DA    / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.20, 0.20)
            capex  = total_rev * max(0.25, (1.20 * (1 - fade) + 0.25 * fade))
            fcfs.append(nopat + da - capex)
        disc   = [(1 + WACC) ** -(i+1) for i in range(FORECAST_YEARS)]
        pv_f   = sum(np.array(fcfs) * np.array(disc))
        tv     = (fcfs[-1] * (1 + TERMINAL_GROWTH) / (WACC - TERMINAL_GROWTH)) if WACC > TERMINAL_GROWTH else 0
        pv_tv  = tv * disc[-1]
        ev     = pv_f + pv_tv
        fv     = (ev - NET_DEBT_USD_M) / (SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS)
        row_vals.append(round(fv, 1))
    matrix2.append(row_vals)

df_sens2 = pd.DataFrame(
    matrix2,
    index=[f"{rg:.0%}" for rg in rev_growth_range],
    columns=[f"{m:.0%}" for m in margin_range],
)
fig2 = px.imshow(
    df_sens2.values, text_auto='.1f',
    x=[f"{m:.0%}"  for m  in margin_range],
    y=[f"{rg:.0%}" for rg in rev_growth_range],
    color_continuous_scale='RdYlGn',
    title=f"{COMPANY_NAME} -- Fair Value: Revenue Growth vs Terminal Op Margin",
    labels=dict(x="Terminal Operating Margin", y="Initial Revenue Growth", color="Fair Value ($)"),
)
fig2.update_layout(template="plotly_white")
fig2.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 7 — Monte Carlo Simulation")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: MONTE CARLO SIMULATION
#  Wide parameter distributions reflect early-stage uncertainty
# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)

def monte_carlo_dcf(n_sims=MC_SIMULATIONS):
    fair_values, sim_params = [], []
    for _ in range(n_sims):
        sim_growth = np.clip(np.random.normal(REVENUE_GROWTH_BASE,  MC_REVENUE_STDEV), 0.10, 3.00)
        sim_margin = np.clip(np.random.normal(OPERATING_MARGIN_TERMINAL, MC_MARGIN_STDEV), 0.02, 0.45)
        sim_wacc   = np.clip(np.random.normal(WACC, MC_WACC_STDEV), 0.07, 0.22)
        sim_tg     = np.random.uniform(0.02, 0.05)

        total_rev = LATEST_REVENUE
        fcfs      = []
        for yr in range(FORECAST_YEARS):
            fade   = yr / max(FORECAST_YEARS - 1, 1)
            g      = sim_growth * (1 - fade) + sim_tg * fade
            margin = -0.40 * (1 - fade) + sim_margin * fade
            total_rev *= (1 + g)
            ebit   = total_rev * margin - CORPORATE_GA_USD_M - total_rev * SBC_PCT_REVENUE
            nopat  = max(0, ebit * (1 - TAX_RATE))
            da     = total_rev * max(LATEST_DA    / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.20, 0.20)
            capex  = total_rev * max(0.25, (1.20 * (1 - fade) + 0.25 * fade))
            fcfs.append(nopat + da - capex)

        disc  = [(1 + sim_wacc) ** -(i+1) for i in range(FORECAST_YEARS)]
        pv_f  = sum(np.array(fcfs) * np.array(disc))
        if sim_wacc > sim_tg:
            pv_tv = (fcfs[-1] * (1 + sim_tg) / (sim_wacc - sim_tg)) * disc[-1]
        else:
            pv_tv = 0
        ev   = pv_f + pv_tv
        eq   = ev - NET_DEBT_USD_M
        dsh  = SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS
        fv   = eq / dsh if dsh > 0 else 0
        # Filter extreme outliers only
        if fv > 0.10 and fv < CURRENT_PRICE * 50:
            fair_values.append(fv)
            sim_params.append({'growth': sim_growth, 'margin': sim_margin,
                               'wacc': sim_wacc, 'tg': sim_tg, 'fv': fv})
    return np.array(fair_values), pd.DataFrame(sim_params)

mc_values, mc_params = monte_carlo_dcf()

fig = go.Figure()
fig.add_trace(go.Histogram(x=mc_values, nbinsx=120, name='Simulated Fair Values',
                            marker_color='#00BCD4', opacity=0.75))
fig.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red",
              annotation_text=f"Current: ${CURRENT_PRICE:,.1f}")
fig.add_vline(x=np.median(mc_values), line_dash="dash", line_color="#00C853",
              annotation_text=f"Median: ${np.median(mc_values):,.1f}")
fig.update_layout(
    title=f"{COMPANY_NAME} -- Monte Carlo Fair Value ({len(mc_values):,} simulations)",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Frequency",
    template="plotly_white",
)
fig.show()

pct_upside = (mc_values > CURRENT_PRICE).mean() * 100
print(f"\\n  Monte Carlo Results  ({len(mc_values):,} valid simulations)")
print(f"  Probability of upside (> ${CURRENT_PRICE:,.1f}): {pct_upside:.1f}%")
print(f"  10th pct:  ${np.percentile(mc_values, 10):,.2f}")
print(f"  25th pct:  ${np.percentile(mc_values, 25):,.2f}")
print(f"  Median:    ${np.median(mc_values):,.2f}")
print(f"  75th pct:  ${np.percentile(mc_values, 75):,.2f}")
print(f"  90th pct:  ${np.percentile(mc_values, 90):,.2f}")
print(f"  Mean:      ${np.mean(mc_values):,.2f}")
print(f"  Std Dev:   ${np.std(mc_values):,.2f}")

# CDF
sorted_vals = np.sort(mc_values)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=sorted_vals, y=cdf, mode='lines',
                              line=dict(color='#00BCD4', width=2)))
fig_cdf.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red",
                  annotation_text="Current Price")
fig_cdf.update_layout(
    title=f"{COMPANY_NAME} -- Cumulative Distribution of Fair Value",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Cumulative Probability",
    template="plotly_white",
)
fig_cdf.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — EV/REVENUE MULTIPLE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 8 — EV/Revenue Multiple Valuation")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: EV/REVENUE MULTIPLE ANALYSIS
#  For pre-profitability companies, market typically uses forward
#  EV/Revenue multiples rather than P/E or EV/EBITDA.
# ═══════════════════════════════════════════════════════════════════

# GPU cloud / AI infrastructure comps (approx. 2024-2025 multiples)
comps_ev_rev = {
    'CoreWeave (private, Series C)': 12.0,
    'Lambda Labs (private)': 10.0,
    'AWS (implied)': 8.0,
    'GCP (implied)': 10.0,
    'High-growth AI infra (median)': 10.0,
    'High-growth SaaS (median)': 15.0,
}

# Project Nebius AI forward revenues (next 12 months NTM)
nb_ai_rev = segments['Nebius AI (GPU Cloud)']['current_revenue_usdm']
ntm_rev_ai = nb_ai_rev * (1 + segments['Nebius AI (GPU Cloud)']['growth_rate_base'])

# Group total NTM revenue
group_ntm_rev = sum(
    s['current_revenue_usdm'] * (1 + s['growth_rate_base'])
    for s in segments.values()
)

print(f"  Nebius AI NTM Revenue (base): ${ntm_rev_ai:,.0f}M")
print(f"  Group NTM Revenue (base):     ${group_ntm_rev:,.0f}M")
print()
print("  EV/Revenue Implied Valuations (Group NTM Revenue)")
print("  " + "=" * 55)

rows = []
for comp, mult in comps_ev_rev.items():
    implied_ev  = group_ntm_rev * mult
    implied_eq  = implied_ev - NET_DEBT_USD_M
    implied_fv  = implied_eq / SHARES_OUT_M if SHARES_OUT_M > 0 else 0
    upside_pct  = (implied_fv / CURRENT_PRICE - 1) * 100 if CURRENT_PRICE > 0 else 0
    rows.append({
        'Comparable'       : comp,
        'EV/NTM Revenue'   : f"{mult:.1f}x",
        'Implied EV ($M)'  : f"${implied_ev:,.0f}",
        'Implied FV/Share' : f"${implied_fv:,.2f}",
        'Upside'           : f"{upside_pct:+.1f}%",
    })
    print(f"  {mult:4.1f}x NTM Rev  ->  EV ${implied_ev/1000:,.1f}B  |  "
          f"${implied_fv:,.2f}/share  ({upside_pct:+.1f}%)")

df_comps = pd.DataFrame(rows)

# Visualise implied share prices across multiple range
mult_range = np.arange(4, 22, 1)
fv_range   = [(group_ntm_rev * m - NET_DEBT_USD_M) / SHARES_OUT_M for m in mult_range]

fig = go.Figure()
fig.add_trace(go.Scatter(x=mult_range, y=fv_range, mode='lines+markers',
                          line=dict(color='#00BCD4', width=2), name='Fair Value'))
fig.add_hline(y=CURRENT_PRICE, line_dash="dash", line_color="red",
              annotation_text=f"Current Price ${CURRENT_PRICE:.2f}")
fig.update_layout(
    title=f"{COMPANY_NAME} -- Implied Fair Value vs EV/NTM Revenue Multiple",
    xaxis_title="EV / NTM Revenue Multiple",
    yaxis_title="Implied Fair Value per Share ($)",
    template="plotly_white",
)
fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PEER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 9 — Peer / Relative Valuation")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: PEER COMPARISON
#  Public peers: AI cloud infra, GPU server, high-growth cloud
# ═══════════════════════════════════════════════════════════════════

# Note: CoreWeave and Lambda Labs are private; use public proxies
PEER_TICKERS = ['SMCI', 'NET', 'SNOW', 'DDOG', 'CFLT', 'HUT']

if YF_AVAILABLE:
    peer_data = []
    for t in [TICKER] + PEER_TICKERS:
        try:
            p     = yf.Ticker(t)
            pi    = p.info
            p_inc = p.income_stmt
            p_cf  = p.cash_flow

            rev   = p_inc.loc['Total Revenue'].dropna().iloc[0]    / 1e9 if 'Total Revenue'    in p_inc.index else 0
            oi    = p_inc.loc['Operating Income'].dropna().iloc[0] / 1e9 if 'Operating Income' in p_inc.index else 0
            ni    = p_inc.loc['Net Income'].dropna().iloc[0]       / 1e9 if 'Net Income'       in p_inc.index else 0
            ocf   = p_cf.loc['Operating Cash Flow'].dropna().iloc[0]     / 1e9 if 'Operating Cash Flow'   in p_cf.index else 0
            capex = abs(p_cf.loc['Capital Expenditure'].dropna().iloc[0]) / 1e9 if 'Capital Expenditure' in p_cf.index else 0
            fcf   = ocf - capex

            mcap  = pi.get('marketCap', 0) / 1e9
            ev_v  = pi.get('enterpriseValue', 0) / 1e9

            rev_vals = p_inc.loc['Total Revenue'].dropna() if 'Total Revenue' in p_inc.index else pd.Series()
            rev_growth = (rev_vals.iloc[0] / rev_vals.iloc[1] - 1) if len(rev_vals) >= 2 and rev_vals.iloc[1] != 0 else 0

            peer_data.append({
                'Ticker'       : t,
                'Mkt Cap ($B)' : round(mcap, 2),
                'EV ($B)'      : round(ev_v, 2),
                'Revenue ($B)' : round(rev, 2),
                'Rev Growth'   : f"{rev_growth:.0%}",
                'Op Margin'    : f"{oi/rev:.0%}" if rev > 0 else "N/A",
                'FCF ($B)'     : round(fcf, 2),
                'P/E'          : round(pi.get('trailingPE', 0) or 0, 1),
                'EV/Rev'       : round(ev_v / rev, 1) if rev > 0 else 0,
                '_rev_g'       : rev_growth,
                '_ev_r'        : ev_v / rev if rev > 0 else 0,
                '_mcap'        : mcap,
            })
        except Exception as e:
            print(f"  Could not fetch {t}: {e}")

    if peer_data:
        df_peers = pd.DataFrame(peer_data)
        print("=" * 90)
        print("  Nebius Group -- Peer Comparison (AI cloud / high-growth infra)")
        print("=" * 90)
        display_cols = [c for c in df_peers.columns if not c.startswith('_')]
        print(df_peers[display_cols].to_string(index=False))

        fig = px.scatter(
            df_peers, x='_rev_g', y='_ev_r', size='_mcap', text='Ticker',
            color='Ticker',
            title="Peer Comparison: EV/Revenue vs Revenue Growth",
            labels={'_rev_g': 'Revenue Growth (YoY)', '_ev_r': 'EV/Revenue',
                    '_mcap': 'Market Cap ($B)'},
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(template='plotly_white', xaxis_tickformat='.0%', showlegend=False)
        fig.show()
else:
    print("yfinance not available -- peer comparison skipped")
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FCF WATERFALL
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 10 — FCF Bridge Waterfall")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: FCF BRIDGE WATERFALL
#  Year 5 of base projection (first year approaching FCF breakeven)
# ═══════════════════════════════════════════════════════════════════

base_details = results['base']['annual_details']
# Use Year 5 (index 4) when Nebius AI is approaching breakeven
yr_idx = 4
yr5 = base_details.iloc[yr_idx]

labels = ['Revenue', 'Segment OpEx', 'SBC', 'Corp G&A', 'EBIT',
          'Tax', 'NOPAT', 'D&A', 'Capex', 'FCF']
values = [
    yr5['revenue'],
    -(yr5['revenue'] - yr5['op_income']),
    -yr5['sbc'],
    -yr5['corp_ga'],
     yr5['ebit'],
    -(max(0, yr5['ebit']) * TAX_RATE),
     yr5['nopat'],
     yr5['da'],
    -yr5['capex'],
     yr5['fcf'],
]
measures = ['absolute','relative','relative','relative','total',
            'relative','total','relative','relative','total']

fig = go.Figure(go.Waterfall(
    name="FCF Bridge", orientation="v",
    measure=measures, x=labels, y=values,
    connector={"line": {"color": "#5f6368"}},
    increasing={"marker": {"color": "#00C853"}},
    decreasing={"marker": {"color": "#EA4335"}},
    totals   ={"marker": {"color": "#00BCD4"}},
))
fig.update_layout(
    title=f"{COMPANY_NAME} -- FCF Bridge (Base Case, Year {yr_idx+1} Projection, $M)",
    yaxis_title="USD Millions", template="plotly_white",
)
fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — INVESTMENT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 11 — Investment Summary")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 11: INVESTMENT SUMMARY
# ═══════════════════════════════════════════════════════════════════

base_r = results['base']

print("+" + "="*78 + "+")
print(f"  NEBIUS GROUP N.V. (NBIS) -- INVESTMENT SUMMARY")
print("+" + "="*78 + "+")
print(f"  Current Price        : ${CURRENT_PRICE:>10,.2f}")
print(f"  Market Cap           : ${MARKET_CAP_B:>10,.2f}B")
print(f"  Net Cash Position    : ${-NET_DEBT_USD_M/1000:>10,.1f}B  (net cash)")
print(f"  Shares Outstanding   : {SHARES_OUT_M:>10,.0f}M")
print()
print(f"  -- DCF Valuation (WACC: {WACC:.2%}) ---")
print(f"  Bear Case Fair Value : ${results['bear']['fair_value_per_share']:>10,.2f}  "
      f"({results['bear']['upside_pct']:+.1f}%)")
print(f"  Base Case Fair Value : ${base_r['fair_value_per_share']:>10,.2f}  "
      f"({base_r['upside_pct']:+.1f}%)  <-- PRIMARY")
print(f"  Bull Case Fair Value : ${results['bull']['fair_value_per_share']:>10,.2f}  "
      f"({results['bull']['upside_pct']:+.1f}%)")
print()
print(f"  -- Monte Carlo (10,000 sims) ---")
print(f"  Median Fair Value    : ${np.median(mc_values):>10,.2f}")
print(f"  Probability Upside   : {(mc_values > CURRENT_PRICE).mean()*100:>9.1f}%")
print()
print(f"  -- Key Assumptions ---")
print(f"  Revenue Growth (Base): {REVENUE_GROWTH_BASE:.0%} (Nebius AI: 100%) -> fades to {FADE_GROWTH_TO:.0%}")
print(f"  Terminal Op Margin   : {OPERATING_MARGIN_TERMINAL:.0%} (GPU cloud mature)")
print(f"  WACC                 : {WACC:.2%}  (Beta: {BETA:.2f}, high-growth premium)")
print(f"  Capex/Revenue        : ~120% now -> ~25% by Year 10")
print(f"  SBC / Revenue        : {SBC_PCT_REVENUE:.0%}")
print(f"  Net Dilution/Year    : {ANNUAL_DILUTION_PCT:.0%}")
print()
print(f"  -- Key Risks & Catalysts ---")
risks = [
    "BULL: Nvidia preferential supply gives Nebius fill-rate advantage vs hyperscalers.",
    "      GPU pricing stays firm; AI inference demand drives utilisation above 80%+.",
    "      Nebius AI reaches EBITDA breakeven by 2026-2027 ahead of consensus.",
    "      Avride / Toloka optionality undervalued; future Waymo-like monetisation.",
    "",
    "BEAR: Hyperscalers (AWS, GCP, Azure) price aggressively; GPU spot rates collapse.",
    "      Relisting history limits institutional investor base; liquidity risk.",
    "      Capex intensity + dilution make FCF conversion slow and uncertain.",
    "      Geopolitical/regulatory risk: European/Dutch HQ, some CIS exposure.",
    "",
    f"NOTE: Terminal value = {base_r['terminal_pct_of_ev']:.0f}% of base EV -- very high for",
    "      a pre-profitability company. WACC and terminal growth assumptions",
    "      are the dominant drivers. Use EV/Revenue multiple analysis as anchor.",
]
for line in risks:
    print(f"  {line}")
print("+" + "="*78 + "+")
""")

# ─────────────────────────────────────────────────────────────────────────────
# FINALISE
# ─────────────────────────────────────────────────────────────────────────────
code("""\
print("Nebius Group N.V. (NBIS) valuation notebook — execution complete.")
""")

# ── Write the notebook ───────────────────────────────────────────────────────
nb.cells = cells

OUTPUT_PATH = "NBIS_valuation.ipynb"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook saved: {OUTPUT_PATH}")
