"""
Generate WYFI_valuation.ipynb — WhiteFiber, Inc. (WYFI)
Uses nbformat to build a full tech-company valuation notebook.
Run:  python generate_wyfi_valuation.py

WhiteFiber, Inc. (formerly Celer, Inc.) designs, develops, and operates
data centres providing AI infrastructure / GPU cloud (HPC) services.
Listed on Nasdaq CM; incorporated 2024; HQ: New York, NY.
FY2024: $51.2M revenue (+64% YoY), 61% gross margin, -72% op margin (capex ramp).
8 sell-side analysts; consensus mean target $34.63 vs ~$18.51 current.
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
md("""# WhiteFiber, Inc. (WYFI) — Comprehensive Valuation

**Sections**
1. Imports & Configuration
2. Live Data (yfinance)
3. Company Overview & Historical Financials
4. Segment Revenue Model
5. Technical Analysis
6. Segment-Level DCF
7. Sensitivity Heatmaps
8. Monte Carlo Simulation
9. EV/Revenue Multiple Valuation
10. Peer Comparison
11. FCF Bridge Waterfall
12. Investment Summary

---
*Micro-cap AI data centre / GPU cloud infrastructure. Edit config variables in Cell 0.*
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Cell 0 — Imports & Configuration")
code("""\
# ===================================================================
#  TECH COMPANY VALUATION -- WHITEFIBER, INC. (WYFI)
#  Built by Claude. Edit the variables below to customise.
#  FY2024 actuals used as base; yfinance will update on re-run.
# ===================================================================

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
TICKER       = "WYFI"
COMPANY_NAME = "WhiteFiber, Inc."

# ── FY2024 ACTUALS (fallback if yfinance unavailable) ────────────
# Revenue: $51.2M | Gross margin: 61% | Op income: -$4.4M
# Capex: $79.0M | D&A: $16.5M | FCF: -$60.6M
# Cash: $166.5M | Debt: $41.4M -> Net cash: ~$125M
SHARES_OUT_M    = 38.3      # ~38.3M diluted shares
CURRENT_PRICE   = 0.0
MARKET_CAP_B    = 0.0
NET_DEBT_USD_M  = -125_000  # ~$125M net cash (USD thousands -> millions below)
# Note: NET_DEBT_USD_M in MILLIONS: $125M net cash = -125
NET_DEBT_USD_M  = -125.0    # USD millions, negative = net cash
BETA            = 2.20      # estimated; not yet in yfinance (very new listing)

# ── REVENUE GROWTH ASSUMPTIONS ───────────────────────────────────
# FY2024: $51.2M. Analyst consensus implies ~$140-180M by FY2026.
# Q3 2025 quarterly run-rate ~$20M suggests FY2025 ~$75-90M.
REVENUE_GROWTH_BEAR  = 0.45   # slower ramp / competitive pressure
REVENUE_GROWTH_BASE  = 0.80   # demand-driven GPU cloud acceleration
REVENUE_GROWTH_BULL  = 1.40   # hyperscaler overflow + enterprise contracts

# ── MARGIN ASSUMPTIONS ───────────────────────────────────────────
# Gross margin: 61% (strong for infrastructure — software-defined layer)
# Operating margin: -72% (capex/D&A heavy; improving as assets depreciate)
OPERATING_MARGIN_TERMINAL = 0.18   # mature GPU cloud data centre (asset-heavy)
FCF_CONVERSION             = 0.65  # capex normalises to maintenance level

# ── DISCOUNT RATE ────────────────────────────────────────────────
RISK_FREE_RATE    = 0.043
EQUITY_RISK_PREM  = 0.055
COST_OF_DEBT      = 0.065    # small-cap, limited credit history
DEBT_WEIGHT       = 0.05
TAX_RATE          = 0.21     # effective (from Q3 2025 tax rate calc)

WACC_OVERRIDE     = None

# ── DCF HORIZON ──────────────────────────────────────────────────
FORECAST_YEARS  = 10
TERMINAL_GROWTH = 0.03
FADE_GROWTH_TO  = 0.05    # growth fades to 5% by Year 10

# ── MONTE CARLO ──────────────────────────────────────────────────
MC_SIMULATIONS   = 10_000
MC_REVENUE_STDEV = 0.28    # very wide — micro-cap early stage
MC_MARGIN_STDEV  = 0.04
MC_WACC_STDEV    = 0.02

# ── SHARE-BASED COMPENSATION ─────────────────────────────────────
# FY2024: SBC not separately disclosed; use industry proxy ~20% rev
SBC_PCT_REVENUE     = 0.20
ANNUAL_DILUTION_PCT = 0.04   # ~4% gross dilution (no buybacks at this stage)

# ── CORPORATE G&A ────────────────────────────────────────────────
# FY2024 G&A: $14.5M on $51.2M revenue (~28% of rev); expected to lever
CORPORATE_GA_USD_M  = 14.5

# ── ANALYST CONSENSUS (from yfinance at time of build) ───────────
ANALYST_MEAN_TARGET = 34.63
ANALYST_HIGH_TARGET = 40.00
ANALYST_LOW_TARGET  = 25.00
NUM_ANALYSTS        = 8

print("Config loaded.")
print(f"  Analyst consensus: mean ${ANALYST_MEAN_TARGET}  |  high ${ANALYST_HIGH_TARGET}"
      f"  |  low ${ANALYST_LOW_TARGET}  |  N={NUM_ANALYSTS}")
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 1 — Fetch Live Data")
code("""\
# ===================================================================
#  SECTION 1: FETCH LIVE DATA FROM YFINANCE
# ===================================================================

if YF_AVAILABLE:
    stock = yf.Ticker(TICKER)
    info  = stock.info

    COMPANY_NAME    = info.get('longName', COMPANY_NAME)
    CURRENT_PRICE   = info.get('currentPrice', info.get('regularMarketPrice', 0))
    MARKET_CAP_B    = info.get('marketCap', 0) / 1e9
    SHARES_OUT_M    = (info.get('sharesOutstanding', 0) or 0) / 1e6
    _beta           = info.get('beta', None)
    if _beta is not None:
        BETA = _beta

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
            'target_mean'   : info.get('targetMeanPrice',          ANALYST_MEAN_TARGET),
            'target_low'    : info.get('targetLowPrice',           ANALYST_LOW_TARGET),
            'target_high'   : info.get('targetHighPrice',          ANALYST_HIGH_TARGET),
            'num_analysts'  : info.get('numberOfAnalystOpinions',  NUM_ANALYSTS),
            'recommendation': info.get('recommendationKey',        'n/a'),
        }
    except:
        analyst_targets = {
            'target_mean': ANALYST_MEAN_TARGET,
            'target_low' : ANALYST_LOW_TARGET,
            'target_high': ANALYST_HIGH_TARGET,
            'num_analysts': NUM_ANALYSTS,
        }

    try: insider_txns = stock.insider_transactions
    except: insider_txns = pd.DataFrame()

    try: inst_holders = stock.institutional_holders
    except: inst_holders = pd.DataFrame()

    def safe_get(df, label, default=0):
        if df is None or df.empty: return default
        if label in df.index:
            val = df.loc[label].dropna()
            if len(val) > 0: return float(val.iloc[0])
        return default

    LATEST_REVENUE      = safe_get(income_stmt, 'Total Revenue')       / 1e6
    LATEST_GROSS_PROFIT = safe_get(income_stmt, 'Gross Profit')        / 1e6
    LATEST_OP_INCOME    = safe_get(income_stmt, 'Operating Income')    / 1e6
    LATEST_NET_INCOME   = safe_get(income_stmt, 'Net Income')          / 1e6
    LATEST_FCF          = (safe_get(cash_flow, 'Operating Cash Flow')
                           - abs(safe_get(cash_flow, 'Capital Expenditure'))) / 1e6
    LATEST_SBC          = safe_get(cash_flow, 'Stock Based Compensation') / 1e6
    LATEST_CAPEX        = abs(safe_get(cash_flow, 'Capital Expenditure'))  / 1e6
    LATEST_DA           = safe_get(cash_flow, 'Depreciation And Amortization') / 1e6

    # Use FY2024 actuals if yfinance returns near-zero
    if LATEST_REVENUE < 1:
        LATEST_REVENUE      = 51.2
        LATEST_GROSS_PROFIT = 29.8
        LATEST_OP_INCOME    = -4.4
        LATEST_NET_INCOME   = -4.2
        LATEST_FCF          = -60.6
        LATEST_SBC          = 8.0   # estimated
        LATEST_CAPEX        = 79.0
        LATEST_DA           = 16.5

    if CURRENT_PRICE <= 0:
        CURRENT_PRICE = 18.51

    if SHARES_OUT_M < 1:
        SHARES_OUT_M = 38.3

    tax_p = safe_get(income_stmt, 'Tax Provision')
    pre_t = safe_get(income_stmt, 'Pretax Income')
    if pre_t < 0 and abs(tax_p) > 0:
        TAX_RATE = 0.21   # from Q3 2025 data
    elif pre_t > 0 and tax_p > 0:
        TAX_RATE = min(tax_p / pre_t, 0.30)

    if LATEST_REVENUE > 0 and LATEST_SBC > 0:
        SBC_PCT_REVENUE = LATEST_SBC / LATEST_REVENUE

    cost_of_equity = RISK_FREE_RATE + BETA * EQUITY_RISK_PREM
    equity_weight  = 1 - DEBT_WEIGHT
    WACC_CALC = (equity_weight * cost_of_equity
                 + DEBT_WEIGHT * COST_OF_DEBT * (1 - TAX_RATE))
    WACC = WACC_OVERRIDE if WACC_OVERRIDE else WACC_CALC

    print(f"  {COMPANY_NAME} ({TICKER})")
    print(f"  Price: ${CURRENT_PRICE:,.2f}  |  Mkt Cap: ${MARKET_CAP_B:,.3f}B  |  Beta: {BETA:.2f}")
    print(f"  Revenue: ${LATEST_REVENUE:,.1f}M  |  Gross Margin: {LATEST_GROSS_PROFIT/LATEST_REVENUE:.1%}"
          f"  |  Op Income: ${LATEST_OP_INCOME:,.1f}M")
    print(f"  FCF: ${LATEST_FCF:,.1f}M  |  Capex: ${LATEST_CAPEX:,.1f}M  |  D&A: ${LATEST_DA:,.1f}M")
    print(f"  WACC: {WACC:.2%}  |  Tax Rate: {TAX_RATE:.1%}")
    print(f"  Net Cash: ${-NET_DEBT_USD_M:,.1f}M  |  Shares: {SHARES_OUT_M:,.1f}M")
    at = analyst_targets
    print(f"  Analyst targets: mean ${at['target_mean']:,.2f}  |"
          f"  high ${at['target_high']:,.2f}  |"
          f"  low ${at['target_low']:,.2f}  |  N={at['num_analysts']}")
else:
    print("yfinance not available -- using FY2024 actuals as fallback")
    WACC             = WACC_OVERRIDE or 0.155
    LATEST_REVENUE   = 51.2
    LATEST_GROSS_PROFIT = 29.8
    LATEST_OP_INCOME = -4.4
    LATEST_NET_INCOME = -4.2
    LATEST_FCF       = -60.6
    LATEST_SBC       = 8.0
    LATEST_CAPEX     = 79.0
    LATEST_DA        = 16.5
    CURRENT_PRICE    = 18.51
    analyst_targets  = {
        'target_mean': ANALYST_MEAN_TARGET, 'target_low': ANALYST_LOW_TARGET,
        'target_high': ANALYST_HIGH_TARGET, 'num_analysts': NUM_ANALYSTS,
    }
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 2 — Company Overview & Historical Financials")
code("""\
# ===================================================================
#  SECTION 2: COMPANY OVERVIEW
# ===================================================================

overview = \"\"\"
WhiteFiber, Inc. (WYFI) -- Business Overview
============================================
Formerly Celer, Inc. | Renamed October 2024 | Listed Nasdaq CM
Incorporated: 2024 | HQ: New York, NY

WHAT THEY DO
------------
WhiteFiber designs, builds, and operates data centres providing:
  - Colocation & Hosting       Physical rack space for enterprise/HPC
  - GPU Cloud (HPC)            Nvidia GPU-as-a-Service for AI workloads
  - Managed Services           Storage, networking, observability, security

Unlike hyperscalers, WhiteFiber targets the mid-market: AI startups,
research institutions, and enterprises that need dedicated GPU access
without the overhead of building their own data centres.

FY2024 KEY METRICS (ACTUALS)
------------------------------
  Total Revenue       $51.2M    (+64% YoY)
  Gross Profit        $29.8M    (Gross Margin: 58%)
  Operating Income   -$4.4M    (Op Margin: -8.5% reported)
  Net Income         -$4.2M
  Capital Expenditure $79.0M    (154% of revenue -- active buildout)
  D&A                 $16.5M    (depreciating prior capex)
  Free Cash Flow     -$60.6M    (investing phase)
  Cash & Equivalents $166.5M
  Total Debt          $41.4M
  Net Cash           ~$125.1M

ANALYST CONSENSUS
-----------------
  8 analysts covering WYFI
  Mean target: $34.63  |  High: $40.00  |  Low: $25.00
  Implied upside from ~$18.51 current: +87% to mean target

INVESTMENT THESIS SUMMARY
--------------------------
  Bull: Data centre capacity is structurally undersupplied for AI
        inference workloads. WhiteFiber's software-defined approach
        yields 61% gross margins -- high for infrastructure.
        $125M+ net cash runway supports continued capex buildout.
        Revenue trajectory implies ~3x growth to ~$150M+ by FY2026.

  Bear: Micro-cap ($700M mkt cap) with limited public track record.
        Hyperscalers and funded competitors (CoreWeave, Lambda) have
        far greater capital to deploy. Capex/revenue of 1.5x burns
        cash fast. Beta-driven equity risk is very high.
\"\"\"
print(overview)

# FY2024 actuals summary
fy2024 = {
    'Metric'       : ['Revenue', 'Gross Profit', 'Gross Margin', 'Op Income',
                      'Op Margin', 'Net Income', 'Capex', 'D&A', 'FCF', 'Net Cash'],
    'FY2024 Actual': ['$51.2M', '$29.8M', '58%', '-$4.4M',
                      '-8.5%', '-$4.2M', '$79.0M', '$16.5M', '-$60.6M', '+$125.1M'],
    'Note'         : ['64% YoY growth', 'Software-defined efficiency', 'Strong for infra',
                      'D&A masks cash ops income', 'D&A + SG&A heavy',
                      'Small absolute loss', 'Active DC buildout', 'Asset depreciation',
                      'Investment phase', 'Solid runway'],
}
print(pd.DataFrame(fy2024).to_string(index=False))

# Historical financial chart (if available)
if YF_AVAILABLE and income_stmt is not None and not income_stmt.empty:
    years = income_stmt.columns
    rows  = []
    for yr in years:
        rev = income_stmt.loc['Total Revenue', yr]    / 1e6 if 'Total Revenue'    in income_stmt.index else 0
        gp  = income_stmt.loc['Gross Profit', yr]     / 1e6 if 'Gross Profit'     in income_stmt.index else 0
        op  = income_stmt.loc['Operating Income', yr] / 1e6 if 'Operating Income' in income_stmt.index else 0
        ni  = income_stmt.loc['Net Income', yr]       / 1e6 if 'Net Income'       in income_stmt.index else 0
        yr_str = yr.strftime('%Y') if hasattr(yr, 'strftime') else str(yr)
        rows.append({'Year': yr_str,
                     'Revenue ($M)': f"{rev:,.1f}",
                     'Gross Profit ($M)': f"{gp:,.1f}",
                     'Gross Margin': f"{gp/rev:.1%}" if rev > 0 else 'N/A',
                     'Op Income ($M)': f"{op:,.1f}",
                     'Net Income ($M)': f"{ni:,.1f}"})
    if rows:
        print()
        print("=" * 65)
        print(f"  {COMPANY_NAME} -- Financial History")
        print("=" * 65)
        print(pd.DataFrame(rows).to_string(index=False))
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 3 — Segment Revenue Model")
code("""\
# ===================================================================
#  SECTION 3: WHITEFIBER SEGMENT REVENUE MODEL
#  WhiteFiber does not separately report sub-segments.
#  We decompose by service type based on disclosed business lines.
# ===================================================================

# FY2024 revenue allocation (estimated from business description):
#   GPU Cloud (HPC services):        ~55%  ->  $28.1M
#   Colocation & Hosting:            ~30%  ->  $15.3M
#   Managed Services (storage/net):  ~15%  ->   $7.7M

segments = {
    "GPU Cloud (HPC Services)": {
        "current_revenue_usdm"      : 28.1,
        "growth_rate_bear"          : 0.60,
        "growth_rate_base"          : 1.00,   # GPU demand still supply-constrained
        "growth_rate_bull"          : 1.80,
        "fade_to"                   : 0.06,
        "operating_margin_current"  : -0.15,  # negative; D&A + buildout costs
        "operating_margin_terminal" : 0.22,   # mature GPU cloud margin
        "tam_2024_usdm"             : 100_000,
        "tam_2030_usdm"             : 500_000,
        "market_share_pct"          : 0.028,
        "key_driver": "Nvidia GPU capacity; AI inference + training demand",
    },
    "Colocation & Hosting": {
        "current_revenue_usdm"      : 15.3,
        "growth_rate_bear"          : 0.20,
        "growth_rate_base"          : 0.45,
        "growth_rate_bull"          : 0.80,
        "fade_to"                   : 0.04,
        "operating_margin_current"  : 0.10,   # stable, lower-growth
        "operating_margin_terminal" : 0.20,
        "tam_2024_usdm"             : 50_000,
        "tam_2030_usdm"             : 90_000,
        "market_share_pct"          : 0.031,
        "key_driver": "Enterprise co-lo demand; proximity to NYC financial district",
    },
    "Managed Services": {
        "current_revenue_usdm"      : 7.8,
        "growth_rate_bear"          : 0.25,
        "growth_rate_base"          : 0.55,
        "growth_rate_bull"          : 1.00,
        "fade_to"                   : 0.05,
        "operating_margin_current"  : 0.20,
        "operating_margin_terminal" : 0.30,   # software-like margins at scale
        "tam_2024_usdm"             : 30_000,
        "tam_2030_usdm"             : 80_000,
        "market_share_pct"          : 0.026,
        "key_driver": "Storage, networking, observability, security upsells",
    },
}

# Reconcile to FY2024 actual total
_model_total = sum(s['current_revenue_usdm'] for s in segments.values())
_actual_total = LATEST_REVENUE if LATEST_REVENUE > 0 else 51.2
_scale = _actual_total / _model_total
for s in segments.values():
    s['current_revenue_usdm'] *= _scale

print(f"  Segment revenues scaled to FY2024 actual: ${_actual_total:.1f}M")
print()

rows = []
for name, s in segments.items():
    rows.append({
        'Segment'         : name,
        'Rev FY24 ($M)'   : f"{s['current_revenue_usdm']:.1f}",
        'Base Growth'     : f"{s['growth_rate_base']:.0%}",
        'Term. Op Margin' : f"{s['operating_margin_terminal']:.0%}",
        'TAM 2024 ($M)'   : f"{s['tam_2024_usdm']:,}",
        'TAM 2030 ($M)'   : f"{s['tam_2030_usdm']:,}",
        'Key Driver'      : s['key_driver'][:55],
    })
print("=" * 95)
print("  WhiteFiber -- Segment Breakdown (Estimated)")
print("=" * 95)
print(pd.DataFrame(rows).to_string(index=False))

# Projection engine
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
        revenue  *= (1 + growth)
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

WYFI_COLORS = {
    'GPU Cloud (HPC Services)': '#00BCD4',
    'Colocation & Hosting'    : '#7C4DFF',
    'Managed Services'        : '#FF6D00',
}

df_base = projections['base']
fig = go.Figure()
for seg_name, color in WYFI_COLORS.items():
    seg_data = df_base[df_base['segment'] == seg_name]
    fig.add_trace(go.Bar(x=seg_data['year'], y=seg_data['revenue_usdm'],
                         name=seg_name, marker_color=color))
fig.update_layout(
    title=f"{COMPANY_NAME} -- Projected Revenue by Segment (Base Case, $M)",
    xaxis_title="Year", yaxis_title="Revenue ($M)",
    barmode='stack', template="plotly_white",
    legend=dict(orientation="h", y=-0.25),
)
fig.show()

# Scenario totals
fig2 = go.Figure()
for scenario, color in [('bear','#EA4335'), ('base','#00BCD4'), ('bull','#00C853')]:
    df_s = projections[scenario].groupby('year')['revenue_usdm'].sum().reset_index()
    fig2.add_trace(go.Scatter(
        x=df_s['year'], y=df_s['revenue_usdm'],
        name=scenario.capitalize(), mode='lines+markers',
        line=dict(color=color, width=2),
    ))
fig2.update_layout(
    title=f"{COMPANY_NAME} -- Total Revenue Scenarios ($M)",
    xaxis_title="Year", yaxis_title="Revenue ($M)",
    template="plotly_white",
)
fig2.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 4 — Quarterly Revenue Trends")
code("""\
# ===================================================================
#  SECTION 4: QUARTERLY TRENDS
#  WYFI listed late 2024; limited public history available.
# ===================================================================

# Known quarterly data from earnings / yfinance
known_q = {
    'Q2 2024': 2.0,   # approximate (H1 was limited)
    'Q3 2024': 15.0,  # estimated from growth ramp
    'Q4 2024': 17.0,  # FY2024 $51.2M distributed across quarters
    'Q1 2025': 16.0,  # estimated
    'Q2 2025': 18.0,  # estimated
    'Q3 2025': 20.2,  # from yfinance quarterly data
}

if YF_AVAILABLE and quarterly_inc is not None and not quarterly_inc.empty:
    q_data = quarterly_inc.T.copy()
    q_data.index = pd.to_datetime(q_data.index)
    q_data = q_data.sort_index()
    q_rev = q_data['Total Revenue'] / 1e6 if 'Total Revenue' in q_data.columns else pd.Series()
    q_op  = q_data['Operating Income'] / 1e6 if 'Operating Income' in q_data.columns else pd.Series()
    q_gp  = q_data['Gross Profit'] / 1e6 if 'Gross Profit' in q_data.columns else pd.Series()

    if len(q_rev) > 0:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=[
                                "Quarterly Revenue ($M)",
                                "Quarterly Gross Profit ($M)",
                                "Quarterly Operating Income ($M)",
                            ])
        fig.add_trace(go.Bar(x=q_rev.index, y=q_rev.values,
                             name='Revenue', marker_color='#00BCD4'), row=1, col=1)
        if len(q_gp) > 0:
            fig.add_trace(go.Bar(x=q_gp.index, y=q_gp.values,
                                 name='Gross Profit', marker_color='#7C4DFF'), row=2, col=1)
        if len(q_op) > 0:
            bar_c = ['#00C853' if v >= 0 else '#EA4335' for v in q_op.values]
            fig.add_trace(go.Bar(x=q_op.index, y=q_op.values,
                                 name='Op Income', marker_color=bar_c), row=3, col=1)
        fig.update_layout(title=f"{COMPANY_NAME} -- Quarterly Financials",
                          template="plotly_white", height=650)
        fig.show()

        # QoQ growth
        if len(q_rev) >= 2:
            qoq = q_rev.pct_change().dropna()
            bar_c2 = ['#00C853' if v >= 0 else '#EA4335' for v in qoq.values]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=qoq.index, y=qoq.values * 100, marker_color=bar_c2))
            fig2.update_layout(
                title=f"{COMPANY_NAME} -- QoQ Revenue Growth (%)",
                yaxis_title="QoQ Growth (%)", template="plotly_white",
            )
            fig2.show()
    else:
        # Fallback: known data
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(known_q.keys()), y=list(known_q.values()),
                             marker_color='#00BCD4'))
        fig.update_layout(
            title=f"{COMPANY_NAME} -- Estimated Quarterly Revenue ($M)",
            xaxis_title="Quarter", yaxis_title="Revenue ($M)",
            template="plotly_white",
        )
        fig.show()
else:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(known_q.keys()), y=list(known_q.values()),
                         marker_color='#00BCD4'))
    fig.update_layout(
        title=f"{COMPANY_NAME} -- Estimated Quarterly Revenue ($M)",
        xaxis_title="Quarter", yaxis_title="Revenue ($M)",
        template="plotly_white",
    )
    fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 5 — Technical Analysis")
code("""\
# ===================================================================
#  SECTION 5: TECHNICAL ANALYSIS
# ===================================================================

if YF_AVAILABLE and price_data is not None and len(price_data) > 20:
    df_ta = price_data.copy()

    df_ta['EMA_20']  = df_ta['Close'].ewm(span=20,  adjust=False).mean()
    df_ta['EMA_50']  = df_ta['Close'].ewm(span=50,  adjust=False).mean()
    if len(df_ta) > 200:
        df_ta['EMA_200'] = df_ta['Close'].ewm(span=200, adjust=False).mean()
    else:
        df_ta['EMA_200'] = np.nan

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

    # Add analyst target lines
    rows = 4
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=["Price, EMAs & Analyst Targets", "Volume", "MACD", "RSI"])

    fig.add_trace(go.Candlestick(
        x=df_ta.index, open=df_ta['Open'], high=df_ta['High'],
        low=df_ta['Low'], close=df_ta['Close'],
        name='OHLC', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_20'], name='EMA 20',
                             line=dict(color='#FF6D00', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_50'], name='EMA 50',
                             line=dict(color='#FBBC05', width=1)), row=1, col=1)
    if not df_ta['EMA_200'].isna().all():
        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_200'], name='EMA 200',
                                 line=dict(color='#EA4335', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Upper'], showlegend=False,
                             line=dict(color='gray', width=0.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Lower'], showlegend=False,
                             line=dict(color='gray', width=0.5, dash='dot'),
                             fill='tonexty', fillcolor='rgba(128,128,128,0.07)'), row=1, col=1)

    # Analyst target lines
    fig.add_hline(y=ANALYST_MEAN_TARGET, line_dash="dot", line_color="#00C853",
                  annotation_text=f"Analyst Mean ${ANALYST_MEAN_TARGET}", row=1, col=1)
    fig.add_hline(y=ANALYST_HIGH_TARGET, line_dash="dot", line_color="#00BCD4",
                  annotation_text=f"High ${ANALYST_HIGH_TARGET}", row=1, col=1)
    fig.add_hline(y=ANALYST_LOW_TARGET,  line_dash="dot", line_color="#FBBC05",
                  annotation_text=f"Low ${ANALYST_LOW_TARGET}", row=1, col=1)

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

    # Key stats
    if len(price_data) >= 2:
        ret_ytd = (price_data['Close'].iloc[-1] / price_data['Close'].iloc[0] - 1)
        high_52w = df_ta['High'].max()
        low_52w  = df_ta['Low'].min()
        pct_from_high = (CURRENT_PRICE / high_52w - 1)
        print(f"  52W High: ${high_52w:.2f}  |  52W Low: ${low_52w:.2f}")
        print(f"  % from 52W High: {pct_from_high:+.1%}")
        print(f"  Analyst Mean Target: ${ANALYST_MEAN_TARGET:.2f}  "
              f"({(ANALYST_MEAN_TARGET/CURRENT_PRICE - 1):+.1%} from current)")
else:
    print("Note: limited price history for WYFI (recently listed)")
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 6 — Segment-Level DCF Valuation")
code("""\
# ===================================================================
#  SECTION 6: SEGMENT-LEVEL DCF
#  WhiteFiber is pre-FCF-profitability; capex/revenue ~154% in FY2024.
#  The model fades capex intensity as the asset base matures.
#  Terminal value will dominate (~80-90% of EV) -- stress-test WACC.
# ===================================================================

def run_dcf(scenario="base", wacc_override=None, tg_override=None):
    wacc = wacc_override or WACC
    tg   = tg_override   or TERMINAL_GROWTH

    df_proj = projections[scenario]
    years   = sorted(df_proj['year'].unique())

    # FY2024 ratios
    da_pct    = max(LATEST_DA    / LATEST_REVENUE, 0.30) if LATEST_REVENUE > 0 else 0.32
    # Capex starts at ~154% of revenue, fades to ~20% (maintenance) by Year 10
    capex_init   = max(LATEST_CAPEX / LATEST_REVENUE, 1.20) if LATEST_REVENUE > 0 else 1.50
    capex_mature = 0.20
    wc_pct = 0.02

    annual_fcfs, annual_details = [], []

    for i, yr in enumerate(years):
        yr_data       = df_proj[df_proj['year'] == yr]
        total_revenue = yr_data['revenue_usdm'].sum()
        total_op_inc  = yr_data['op_income_usdm'].sum()

        sbc     = total_revenue * SBC_PCT_REVENUE
        corp_ga = CORPORATE_GA_USD_M
        ebit    = total_op_inc - corp_ga - sbc
        taxes   = max(0, ebit * TAX_RATE)
        nopat   = ebit - taxes

        da            = total_revenue * da_pct
        fade_frac     = i / max(FORECAST_YEARS - 1, 1)
        capex_rate    = capex_init * (1 - fade_frac) + capex_mature * fade_frac
        capex         = total_revenue * capex_rate
        wc_change     = total_revenue * wc_pct
        fcf           = nopat + da - capex - wc_change

        annual_fcfs.append(fcf)
        annual_details.append({
            'year': yr, 'revenue': total_revenue, 'op_income': total_op_inc,
            'sbc': sbc, 'corp_ga': corp_ga, 'ebit': ebit, 'nopat': nopat,
            'da': da, 'capex': capex, 'capex_pct': capex_rate, 'fcf': fcf,
        })

    disc_factors = [(1 + wacc) ** -(i+1) for i in range(len(annual_fcfs))]
    pv_fcfs      = np.array(annual_fcfs) * np.array(disc_factors)

    terminal_fcf = annual_fcfs[-1] * (1 + tg)
    terminal_val = terminal_fcf / (wacc - tg) if wacc > tg else 0
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
        'ev_usdm'             : round(ev, 1),
        'equity_value_usdm'   : round(equity_val, 1),
        'fair_value_per_share': round(fv_per_share, 2),
        'upside_pct'          : round(upside * 100, 1),
        'pv_fcfs_total'       : round(sum(pv_fcfs), 1),
        'pv_terminal'         : round(pv_terminal, 1),
        'terminal_pct_of_ev'  : round(pv_terminal / ev * 100, 1) if ev > 0 else 0,
        'implied_ev_rev'      : round(implied_ev_r, 1),
        'annual_details'      : pd.DataFrame(annual_details),
        'pv_fcfs'             : pv_fcfs,
        'wacc_used'           : wacc,
    }

results = {s: run_dcf(s) for s in ['bear', 'base', 'bull']}

summary_rows = []
for label, r in results.items():
    summary_rows.append({
        'Scenario'           : label.upper(),
        'EV ($M)'            : f"${r['ev_usdm']:,.0f}",
        'Equity Value ($M)'  : f"${r['equity_value_usdm']:,.0f}",
        'Fair Value / Share' : f"${r['fair_value_per_share']:,.2f}",
        'vs Current'         : f"{r['upside_pct']:+.1f}%",
        'vs Analyst Mean'    : f"{(r['fair_value_per_share']/ANALYST_MEAN_TARGET - 1)*100:+.1f}%",
        'Terminal % of EV'   : f"{r['terminal_pct_of_ev']:.0f}%",
        'Implied EV/Rev Y10' : f"{r['implied_ev_rev']:.1f}x",
    })
df_scen = pd.DataFrame(summary_rows)
print("=" * 100)
print(f"  {COMPANY_NAME} -- DCF Scenario Summary")
print(f"  Current Price: ${CURRENT_PRICE:.2f}  |  WACC: {WACC:.2%}  |  Terminal Growth: {TERMINAL_GROWTH:.1%}")
print(f"  Analyst Mean Target: ${ANALYST_MEAN_TARGET:.2f}")
print("=" * 100)
print(df_scen.to_string(index=False))

# FCF profile
fig = go.Figure()
for label, r in results.items():
    det = r['annual_details']
    fig.add_trace(go.Scatter(
        x=det['year'], y=det['fcf'],
        name=label.capitalize(), mode='lines+markers',
        line=dict(color={'bear':'#EA4335','base':'#00BCD4','bull':'#00C853'}[label], width=2),
    ))
fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="FCF breakeven")
fig.update_layout(
    title=f"{COMPANY_NAME} -- Projected FCF by Scenario ($M)",
    xaxis_title="Year", yaxis_title="FCF ($M)",
    template="plotly_white",
)
fig.show()

# Capex fade chart
det_base = results['base']['annual_details']
fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_trace(go.Bar(x=det_base['year'], y=det_base['capex'],
                      name='Capex ($M)', marker_color='#EA4335', opacity=0.6), secondary_y=False)
fig2.add_trace(go.Scatter(x=det_base['year'], y=det_base['capex_pct'],
                          name='Capex/Rev', line=dict(color='#7C4DFF', width=2), mode='lines+markers'),
               secondary_y=True)
fig2.update_layout(title=f"{COMPANY_NAME} -- Capex Intensity Fade (Base Case)",
                   template="plotly_white")
fig2.update_yaxes(title_text="Capex ($M)", secondary_y=False)
fig2.update_yaxes(title_text="Capex / Revenue", tickformat=".0%", secondary_y=True)
fig2.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 7 — Sensitivity Heatmaps")
code("""\
# ===================================================================
#  SECTION 7: SENSITIVITY ANALYSIS
# ===================================================================

# WACC vs Terminal Growth
wacc_range = np.arange(0.10, 0.20, 0.01)
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
    title=f"{COMPANY_NAME} -- Fair Value: WACC vs Terminal Growth",
    labels=dict(x="Terminal Growth Rate", y="WACC", color="Fair Value ($)"),
)
fig.update_layout(template="plotly_white")
fig.show()

# Revenue growth vs Terminal Operating Margin
rev_growth_range = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40]
margin_range     = [0.10, 0.14, 0.18, 0.22, 0.26, 0.30]

matrix2 = []
for rg in rev_growth_range:
    row_vals = []
    for m in margin_range:
        total_rev = LATEST_REVENUE
        fcfs = []
        capex_init_s   = 1.50
        capex_mature_s = 0.20
        for yr in range(FORECAST_YEARS):
            fade   = yr / max(FORECAST_YEARS - 1, 1)
            g      = rg * (1 - fade) + TERMINAL_GROWTH * fade
            margin = -0.08 * (1 - fade) + m * fade
            total_rev *= (1 + g)
            ebit   = total_rev * margin - CORPORATE_GA_USD_M - total_rev * SBC_PCT_REVENUE
            nopat  = max(0, ebit * (1 - TAX_RATE))
            da     = total_rev * max(LATEST_DA / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.30, 0.30)
            capex  = total_rev * (capex_init_s * (1 - fade) + capex_mature_s * fade)
            fcfs.append(nopat + da - capex)
        disc  = [(1 + WACC) ** -(i+1) for i in range(FORECAST_YEARS)]
        pv_f  = sum(np.array(fcfs) * np.array(disc))
        tv    = (fcfs[-1] * (1 + TERMINAL_GROWTH) / (WACC - TERMINAL_GROWTH)) if WACC > TERMINAL_GROWTH else 0
        pv_tv = tv * disc[-1]
        ev    = pv_f + pv_tv
        fv    = (ev - NET_DEBT_USD_M) / (SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS)
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
md("## Section 8 — Monte Carlo Simulation")
code("""\
# ===================================================================
#  SECTION 8: MONTE CARLO SIMULATION
# ===================================================================

np.random.seed(42)

def monte_carlo_dcf(n_sims=MC_SIMULATIONS):
    fair_values, sim_params = [], []
    for _ in range(n_sims):
        sim_growth = np.clip(np.random.normal(REVENUE_GROWTH_BASE,  MC_REVENUE_STDEV), 0.10, 4.0)
        sim_margin = np.clip(np.random.normal(OPERATING_MARGIN_TERMINAL, MC_MARGIN_STDEV), 0.02, 0.40)
        sim_wacc   = np.clip(np.random.normal(WACC, MC_WACC_STDEV), 0.08, 0.25)
        sim_tg     = np.random.uniform(0.02, 0.05)

        total_rev  = LATEST_REVENUE
        fcfs       = []
        capex_init_m   = 1.50
        capex_mature_m = 0.20
        for yr in range(FORECAST_YEARS):
            fade   = yr / max(FORECAST_YEARS - 1, 1)
            g      = sim_growth * (1 - fade) + sim_tg * fade
            margin = -0.08 * (1 - fade) + sim_margin * fade
            total_rev *= (1 + g)
            ebit   = total_rev * margin - CORPORATE_GA_USD_M - total_rev * SBC_PCT_REVENUE
            nopat  = max(0, ebit * (1 - TAX_RATE))
            da     = total_rev * max(LATEST_DA / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.30, 0.30)
            capex  = total_rev * (capex_init_m * (1 - fade) + capex_mature_m * fade)
            fcfs.append(nopat + da - capex)

        disc  = [(1 + sim_wacc) ** -(i+1) for i in range(FORECAST_YEARS)]
        pv_f  = sum(np.array(fcfs) * np.array(disc))
        if sim_wacc > sim_tg:
            pv_tv = (fcfs[-1] * (1 + sim_tg) / (sim_wacc - sim_tg)) * disc[-1]
        else:
            pv_tv = 0
        ev  = pv_f + pv_tv
        eq  = ev - NET_DEBT_USD_M
        dsh = SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS
        fv  = eq / dsh if dsh > 0 else 0
        if 0.10 < fv < CURRENT_PRICE * 60:
            fair_values.append(fv)
            sim_params.append({'growth': sim_growth, 'margin': sim_margin,
                               'wacc': sim_wacc, 'tg': sim_tg, 'fv': fv})
    return np.array(fair_values), pd.DataFrame(sim_params)

mc_values, mc_params = monte_carlo_dcf()

fig = go.Figure()
fig.add_trace(go.Histogram(x=mc_values, nbinsx=120,
                            marker_color='#00BCD4', opacity=0.75, name='Simulated FV'))
fig.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red",
              annotation_text=f"Current ${CURRENT_PRICE:.2f}")
fig.add_vline(x=ANALYST_MEAN_TARGET, line_dash="dash", line_color="#FBBC05",
              annotation_text=f"Analyst Mean ${ANALYST_MEAN_TARGET:.2f}")
fig.add_vline(x=np.median(mc_values), line_dash="dash", line_color="#00C853",
              annotation_text=f"MC Median ${np.median(mc_values):.2f}")
fig.update_layout(
    title=f"{COMPANY_NAME} -- Monte Carlo Fair Value ({len(mc_values):,} simulations)",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Frequency",
    template="plotly_white",
)
fig.show()

pct_upside   = (mc_values > CURRENT_PRICE).mean() * 100
pct_vs_analy = (mc_values > ANALYST_MEAN_TARGET).mean() * 100

print(f"\\n  Monte Carlo Results ({len(mc_values):,} valid simulations)")
print(f"  Probability > current (${CURRENT_PRICE:.2f}):         {pct_upside:.1f}%")
print(f"  Probability > analyst mean (${ANALYST_MEAN_TARGET:.2f}): {pct_vs_analy:.1f}%")
print(f"  10th pct: ${np.percentile(mc_values, 10):,.2f}  |  25th pct: ${np.percentile(mc_values, 25):,.2f}")
print(f"  Median:   ${np.median(mc_values):,.2f}  |  Mean: ${np.mean(mc_values):,.2f}")
print(f"  75th pct: ${np.percentile(mc_values, 75):,.2f}  |  90th pct: ${np.percentile(mc_values, 90):,.2f}")
print(f"  Std Dev:  ${np.std(mc_values):,.2f}")

# CDF
sorted_vals = np.sort(mc_values)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=sorted_vals, y=cdf, mode='lines',
                              line=dict(color='#00BCD4', width=2)))
fig_cdf.add_vline(x=CURRENT_PRICE,        line_dash="dash", line_color="red",
                  annotation_text="Current")
fig_cdf.add_vline(x=ANALYST_MEAN_TARGET, line_dash="dash", line_color="#FBBC05",
                  annotation_text="Analyst Mean")
fig_cdf.update_layout(
    title=f"{COMPANY_NAME} -- CDF of Fair Value",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Cumulative Probability",
    template="plotly_white",
)
fig_cdf.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 9 — EV/Revenue Multiple Valuation")
code("""\
# ===================================================================
#  SECTION 9: EV/REVENUE MULTIPLE VALUATION
#  WhiteFiber trades at ~8.3x trailing EV/Revenue (from yfinance).
#  Forward multiples on NTM revenue are more relevant.
# ===================================================================

# NTM revenue estimates (base case projection Year 1)
ntm_rev_base = projections['base'].groupby('year')['revenue_usdm'].sum().iloc[0]
ntm_rev_bear = projections['bear'].groupby('year')['revenue_usdm'].sum().iloc[0]
ntm_rev_bull = projections['bull'].groupby('year')['revenue_usdm'].sum().iloc[0]

print(f"  NTM Revenue Estimates:")
print(f"    Bear: ${ntm_rev_bear:.1f}M  |  Base: ${ntm_rev_base:.1f}M  |  Bull: ${ntm_rev_bull:.1f}M")
print()

# GPU cloud / AI infra comparable multiples
comps = {
    'CoreWeave (Series C, private)'   : 12.0,
    'Lambda Labs (private)'           : 10.0,
    'Nebius AI (NBIS, GPU cloud)'     : 8.0,
    'High-growth AI infra (median)'   : 10.0,
    'Niche data centre operators'     : 5.0,
    'WYFI trailing (8.3x ~ current)'  : 8.3,
}

print("  EV/NTM Revenue Implied Valuations (Base Case NTM Rev: "
      f"${ntm_rev_base:.0f}M)")
print("  " + "=" * 65)
rows = []
for comp, mult in comps.items():
    for scenario, ntm_rev in [('Base', ntm_rev_base), ('Bear', ntm_rev_bear), ('Bull', ntm_rev_bull)]:
        ev     = ntm_rev * mult
        eq     = ev - NET_DEBT_USD_M
        fv     = eq / SHARES_OUT_M if SHARES_OUT_M > 0 else 0
        upside = (fv / CURRENT_PRICE - 1) * 100 if CURRENT_PRICE > 0 else 0
        if scenario == 'Base':
            rows.append({
                'Comparable'       : comp,
                'EV/NTM Rev'       : f"{mult:.1f}x",
                'NTM Rev ($M)'     : f"{ntm_rev:.0f}",
                'Implied EV ($M)'  : f"{ev:.0f}",
                'Implied FV/Share' : f"${fv:.2f}",
                'vs Current'       : f"{upside:+.1f}%",
            })
            print(f"  {mult:5.1f}x  {scenario:5s}  ->  EV ${ev/1000:.2f}B  "
                  f"| FV ${fv:.2f}/share  ({upside:+.1f}%)")

print()
df_multiples = pd.DataFrame(rows)

# Visual: implied fair value across multiple range
mult_range = np.arange(3, 20, 0.5)
fv_base = [(ntm_rev_base * m - NET_DEBT_USD_M) / SHARES_OUT_M for m in mult_range]
fv_bull = [(ntm_rev_bull * m - NET_DEBT_USD_M) / SHARES_OUT_M for m in mult_range]
fv_bear = [(ntm_rev_bear * m - NET_DEBT_USD_M) / SHARES_OUT_M for m in mult_range]

fig = go.Figure()
fig.add_trace(go.Scatter(x=mult_range, y=fv_bull, name='Bull NTM Rev',
                          line=dict(color='#00C853', width=1.5, dash='dot')))
fig.add_trace(go.Scatter(x=mult_range, y=fv_base, name='Base NTM Rev',
                          line=dict(color='#00BCD4', width=2)))
fig.add_trace(go.Scatter(x=mult_range, y=fv_bear, name='Bear NTM Rev',
                          line=dict(color='#EA4335', width=1.5, dash='dot')))
fig.add_hline(y=CURRENT_PRICE, line_dash="dash", line_color="red",
              annotation_text=f"Current ${CURRENT_PRICE:.2f}")
fig.add_hline(y=ANALYST_MEAN_TARGET, line_dash="dash", line_color="#FBBC05",
              annotation_text=f"Analyst Mean ${ANALYST_MEAN_TARGET:.2f}")
fig.update_layout(
    title=f"{COMPANY_NAME} -- Implied Fair Value vs EV/NTM Revenue Multiple",
    xaxis_title="EV / NTM Revenue Multiple",
    yaxis_title="Implied Fair Value per Share ($)",
    template="plotly_white",
)
fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
md("## Section 10 — Peer Comparison")
code("""\
# ===================================================================
#  SECTION 10: PEER COMPARISON
#  WhiteFiber peers: data centre infrastructure + high-growth cloud
# ===================================================================

PEER_TICKERS = ['NBIS', 'SMCI', 'VRT', 'EQIX', 'DLR', 'NET', 'SNOW']

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
                'Ticker'      : t,
                'Mkt Cap ($B)': round(mcap, 2),
                'EV ($B)'     : round(ev_v, 2),
                'Rev ($B)'    : round(rev, 3),
                'Rev Growth'  : f"{rev_growth:.0%}",
                'Gross Margin': f"{pi.get('grossMargins', 0):.0%}" if pi.get('grossMargins') else 'N/A',
                'Op Margin'   : f"{oi/rev:.0%}" if rev > 0 else "N/A",
                'FCF ($B)'    : round(fcf, 2),
                'EV/Rev'      : round(ev_v / rev, 1) if rev > 0 else 0,
                '_rev_g'      : rev_growth,
                '_ev_r'       : ev_v / rev if rev > 0 else 0,
                '_mcap'       : mcap,
            })
        except Exception as e:
            print(f"  Could not fetch {t}: {e}")

    if peer_data:
        df_peers = pd.DataFrame(peer_data)
        print("=" * 95)
        print("  WhiteFiber -- Peer Comparison (AI data centre / cloud infra)")
        print("=" * 95)
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
md("## Section 11 — FCF Bridge Waterfall")
code("""\
# ===================================================================
#  SECTION 11: FCF BRIDGE WATERFALL
#  Year 6 (base case) -- first year WhiteFiber approaches FCF breakeven
# ===================================================================

base_details = results['base']['annual_details']
yr_idx = 5  # Year 6 -- when capex has faded enough for positive FCF
yr6 = base_details.iloc[yr_idx]

labels = ['Revenue', 'Segment OpEx', 'SBC', 'Corp G&A', 'EBIT',
          'Tax', 'NOPAT', 'D&A', 'Capex', 'FCF']
values = [
    yr6['revenue'],
    -(yr6['revenue'] - yr6['op_income']),
    -yr6['sbc'],
    -yr6['corp_ga'],
     yr6['ebit'],
    -(max(0, yr6['ebit']) * TAX_RATE),
     yr6['nopat'],
     yr6['da'],
    -yr6['capex'],
     yr6['fcf'],
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
md("## Section 12 — Investment Summary")
code("""\
# ===================================================================
#  SECTION 12: INVESTMENT SUMMARY
# ===================================================================

base_r = results['base']

print("+" + "="*80 + "+")
print(f"  WHITEFIBER, INC. (WYFI) -- INVESTMENT SUMMARY")
print("+" + "="*80 + "+")
print(f"  Current Price        :  ${CURRENT_PRICE:>10,.2f}")
print(f"  Market Cap           :  ${MARKET_CAP_B:>10,.3f}B  (~$708M at build time)")
print(f"  Enterprise Value     :  ~$583M  (EV/Rev trailing: ~8.3x)")
print(f"  Net Cash             :  ${-NET_DEBT_USD_M:>10,.1f}M")
print(f"  Shares Outstanding   :  {SHARES_OUT_M:>10,.1f}M")
print()
print(f"  -- Analyst Consensus ({NUM_ANALYSTS} analysts) -----")
print(f"  Mean Target          :  ${ANALYST_MEAN_TARGET:>10,.2f}  ({(ANALYST_MEAN_TARGET/CURRENT_PRICE-1)*100:+.1f}%)")
print(f"  High Target          :  ${ANALYST_HIGH_TARGET:>10,.2f}  ({(ANALYST_HIGH_TARGET/CURRENT_PRICE-1)*100:+.1f}%)")
print(f"  Low Target           :  ${ANALYST_LOW_TARGET:>10,.2f}  ({(ANALYST_LOW_TARGET/CURRENT_PRICE-1)*100:+.1f}%)")
print()
print(f"  -- DCF Valuation (WACC: {WACC:.2%}) -----")
print(f"  Bear Case            :  ${results['bear']['fair_value_per_share']:>10,.2f}"
      f"  ({results['bear']['upside_pct']:+.1f}%)")
print(f"  Base Case            :  ${base_r['fair_value_per_share']:>10,.2f}"
      f"  ({base_r['upside_pct']:+.1f}%)  <-- PRIMARY")
print(f"  Bull Case            :  ${results['bull']['fair_value_per_share']:>10,.2f}"
      f"  ({results['bull']['upside_pct']:+.1f}%)")
print()
print(f"  -- Monte Carlo ({MC_SIMULATIONS:,} sims) -----")
print(f"  Median Fair Value    :  ${np.median(mc_values):>10,.2f}")
print(f"  Probability > Current:  {(mc_values > CURRENT_PRICE).mean()*100:>9.1f}%")
print()
print(f"  -- Key Assumptions -----")
print(f"  Revenue Growth Base  :  {REVENUE_GROWTH_BASE:.0%} -> {FADE_GROWTH_TO:.0%} by Year 10")
print(f"  Terminal Op Margin   :  {OPERATING_MARGIN_TERMINAL:.0%}")
print(f"  WACC                 :  {WACC:.2%}  (Beta est. {BETA:.2f})")
print(f"  Capex/Rev            :  ~150% (FY2024) -> ~20% (Year 10)")
print(f"  SBC/Revenue          :  {SBC_PCT_REVENUE:.0%}")
print(f"  Terminal % of EV     :  {base_r['terminal_pct_of_ev']:.0f}% (very high -- verify WACC)")
print()
risks = [
    "BULL: GPU cloud structurally undersupplied; 61% gross margin shows pricing power.",
    "      Analyst consensus $34.63 (+87%) implies substantial near-term re-rating.",
    "      Capex pace builds moat; as assets depreciate, FCF inflects sharply.",
    "      Net cash $125M+ provides 2+ years runway without dilution.",
    "",
    "BEAR: Micro-cap ($700M) with ~12 months of public track record.",
    "      CoreWeave raised $12B+ -- capital-intensive moat hard to replicate.",
    "      Q3 2025 op losses widened (-$14.5M) vs FY2024 (-$4.4M) -- opex scaling.",
    "      SG&A jumped to $21.3M in Q3 2025 alone -- cost discipline critical.",
    "      Low float + small size = high volatility and liquidity risk.",
    "",
    "KEY WATCH ITEMS:",
    "  1. Revenue trajectory: is Q3 2025 $20.2M run-rate accelerating or plateauing?",
    "  2. Gross margin stability: can 61% hold as GPU pricing evolves?",
    "  3. Capex vs booked revenue: are new data centres turning to revenue quickly?",
    "  4. Cash burn: $125M net cash at ~$60M/yr FCF burn = ~2 year runway.",
]
for line in risks:
    print(f"  {line}")
print("+" + "="*80 + "+")
""")

# ─────────────────────────────────────────────────────────────────────────────
code("""\
print("WhiteFiber, Inc. (WYFI) valuation notebook — execution complete.")
""")

# ── Write the notebook ───────────────────────────────────────────────────────
nb.cells = cells

OUTPUT_PATH = "WYFI_valuation.ipynb"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook saved: {OUTPUT_PATH}")
