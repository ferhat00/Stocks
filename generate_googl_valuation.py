"""
Generate GOOGL_valuation.ipynb — Alphabet Inc. (GOOGL)
Uses nbformat to build a full tech-company valuation notebook.
Run:  python generate_googl_valuation.py
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
md("""# Alphabet Inc. (GOOGL) — Comprehensive Valuation

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
10. FCF Waterfall
11. Investment Summary

---
*Built with the tech-company-valuer skill. Edit config variables in Cell 0 to customise.*
""")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0 — IMPORTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
md("## Cell 0 — Imports & Configuration")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  TECH COMPANY VALUATION — ALPHABET INC. (GOOGL)
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
    print("⚠️  yfinance not installed. Run: pip install yfinance")
    YF_AVAILABLE = False

# ── COMPANY ───────────────────────────────────────────────────────
TICKER          = "GOOGL"
COMPANY_NAME    = "Alphabet Inc."

# ── Populated from yfinance if available ─────────────────────────
SHARES_OUT_M    = 12_300   # ~12.3B diluted shares (millions)
CURRENT_PRICE   = 0.0
MARKET_CAP_B    = 0.0
NET_DEBT_USD_M  = -100_000  # Alphabet is deeply net-cash (~$100B+)
BETA            = 1.12

# ── REVENUE GROWTH ASSUMPTIONS (annual %) ────────────────────────
# Alphabet blended — Search + YouTube + Cloud
REVENUE_GROWTH_BEAR  = 0.08   # macro/ad slowdown
REVENUE_GROWTH_BASE  = 0.13   # continued Search + Cloud expansion
REVENUE_GROWTH_BULL  = 0.20   # AI-driven upside (Gemini, GCP share gains)

# ── MARGIN ASSUMPTIONS ───────────────────────────────────────────
OPERATING_MARGIN_TERMINAL = 0.33  # long-run blended op margin
FCF_CONVERSION             = 0.88  # FCF / operating income at maturity

# ── DISCOUNT RATE ────────────────────────────────────────────────
RISK_FREE_RATE    = 0.043
EQUITY_RISK_PREM  = 0.055
COST_OF_DEBT      = 0.040
DEBT_WEIGHT       = 0.02   # Alphabet is nearly all-equity
TAX_RATE          = 0.14   # effective ~14% (international structure)

WACC_OVERRIDE     = None   # set e.g. 0.10 to override CAPM calc

# ── DCF HORIZON ──────────────────────────────────────────────────
FORECAST_YEARS      = 10
TERMINAL_GROWTH     = 0.03
FADE_GROWTH_TO      = 0.04   # revenue growth fades to this by Year 10

# ── MONTE CARLO ──────────────────────────────────────────────────
MC_SIMULATIONS      = 10_000
MC_REVENUE_STDEV    = 0.06
MC_MARGIN_STDEV     = 0.025
MC_WACC_STDEV       = 0.01

# ── SHARE-BASED COMPENSATION ─────────────────────────────────────
SBC_PCT_REVENUE     = 0.10   # ~10% of revenue (typical FAANG tier)
ANNUAL_DILUTION_PCT = 0.01   # ~1% net dilution after buybacks

# ── CORPORATE G&A ────────────────────────────────────────────────
CORPORATE_GA_USD_M  = 6_000  # ~$6B corp-level G&A above segment opex

print("✅  Config loaded. Edit variables above and re-run to update all outputs.")
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

    cost_of_equity = RISK_FREE_RATE + BETA * EQUITY_RISK_PREM
    equity_weight  = 1 - DEBT_WEIGHT
    WACC_CALC = (equity_weight * cost_of_equity
                 + DEBT_WEIGHT * COST_OF_DEBT * (1 - TAX_RATE))
    WACC = WACC_OVERRIDE if WACC_OVERRIDE else WACC_CALC

    print(f"📊  {COMPANY_NAME} ({TICKER})")
    print(f"    Price: ${CURRENT_PRICE:,.2f}  |  Mkt Cap: ${MARKET_CAP_B:,.1f}B  |  Beta: {BETA:.2f}")
    print(f"    Revenue: ${LATEST_REVENUE:,.0f}M  |  Op Income: ${LATEST_OP_INCOME:,.0f}M  |  FCF: ${LATEST_FCF:,.0f}M")
    print(f"    WACC: {WACC:.2%}  |  Tax Rate: {TAX_RATE:.1%}  |  SBC/Rev: {SBC_PCT_REVENUE:.1%}")
    print(f"    Net Debt: ${NET_DEBT_USD_M:,.0f}M  |  Shares: {SHARES_OUT_M:,.0f}M")
    if analyst_targets.get('target_mean'):
        print(f"    Analyst mean target: ${analyst_targets['target_mean']:,.2f}"
              f"  |  Recommendation: {analyst_targets['recommendation']}")
else:
    print("⚠️  yfinance not available — using manual config values")
    WACC = WACC_OVERRIDE or 0.095
    LATEST_REVENUE = 350_000   # $350B placeholder (FY2024 estimate)
    LATEST_OP_INCOME = 113_000
    LATEST_FCF = 72_000
    LATEST_SBC = 22_000
    LATEST_CAPEX = 53_000
    LATEST_DA = 17_000
    CURRENT_PRICE = 175.0
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — COMPANY OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 2 — Company Overview & Historical Financials")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: COMPANY OVERVIEW
# ═══════════════════════════════════════════════════════════════════

md_text = \"\"\"
### Alphabet Inc. (GOOGL) — Business Overview

Alphabet is the parent company of Google and several subsidiaries.
Its reportable segments are:

| Segment | FY2024 Rev (est.) | Growth | Key Driver |
|---------|------------------|--------|------------|
| Google Search & Other | ~$198B | ~10% | AI-enhanced search, continued monetisation |
| YouTube Ads | ~$36B | ~14% | Connected TV shift, Shorts monetisation |
| Google Network | ~$30B | ~3% | Programmatic display; declining share |
| Google Subscriptions/Platforms/Devices | ~$15B | ~8% | Google One, Play Store, Pixel |
| Google Cloud (GCP) | ~$43B | ~28% | AI/ML workloads, enterprise migration |
| Other Bets | ~$1.5B | ~15% | Waymo, Verily; early-stage |

**Investment thesis**: Google Search retains durable monetisation moat.
GCP is the fastest-growing segment and approaching profitability scale.
AI integration (Gemini) is a cross-segment tailwind. Net-cash balance
sheet + aggressive buybacks (~$70B/year) support shareholder returns.
\"\"\"
print(md_text)

# ── 5-Year Financial Summary ──────────────────────────────────────
if YF_AVAILABLE and income_stmt is not None and not income_stmt.empty:
    years = income_stmt.columns[:5]
    summary_data = []
    for yr in years:
        rev = income_stmt.loc['Total Revenue', yr] / 1e6 if 'Total Revenue' in income_stmt.index else 0
        gp  = income_stmt.loc['Gross Profit', yr]  / 1e6 if 'Gross Profit'  in income_stmt.index else 0
        op  = income_stmt.loc['Operating Income', yr] / 1e6 if 'Operating Income' in income_stmt.index else 0
        ni  = income_stmt.loc['Net Income', yr]    / 1e6 if 'Net Income'    in income_stmt.index else 0
        summary_data.append({
            'Year'             : yr.strftime('%Y') if hasattr(yr, 'strftime') else str(yr),
            'Revenue ($M)'     : f"{rev:,.0f}",
            'Gross Profit ($M)': f"{gp:,.0f}",
            'Gross Margin'     : f"{gp/rev:.1%}" if rev > 0 else "N/A",
            'Op Income ($M)'   : f"{op:,.0f}",
            'Op Margin'        : f"{op/rev:.1%}" if rev > 0 else "N/A",
            'Net Income ($M)'  : f"{ni:,.0f}",
            'Net Margin'       : f"{ni/rev:.1%}" if rev > 0 else "N/A",
        })
    df_summary = pd.DataFrame(summary_data)
    print("═" * 90)
    print(f"  {COMPANY_NAME} — 5-Year Financial Summary")
    print("═" * 90)
    print(df_summary.to_string(index=False))

# ── Revenue & Margin Trends Chart ─────────────────────────────────
if YF_AVAILABLE and income_stmt is not None and not income_stmt.empty:
    years_plot, revenues, op_margins, net_margins, gross_margins = [], [], [], [], []
    for yr in reversed(list(income_stmt.columns[:5])):
        rev = income_stmt.loc['Total Revenue', yr]    / 1e9 if 'Total Revenue'    in income_stmt.index else 0
        gp  = income_stmt.loc['Gross Profit', yr]     / 1e9 if 'Gross Profit'     in income_stmt.index else 0
        op  = income_stmt.loc['Operating Income', yr] / 1e9 if 'Operating Income' in income_stmt.index else 0
        ni  = income_stmt.loc['Net Income', yr]       / 1e9 if 'Net Income'       in income_stmt.index else 0
        yr_str = yr.strftime('%Y') if hasattr(yr, 'strftime') else str(yr)
        years_plot.append(yr_str)
        revenues.append(rev)
        gross_margins.append(gp / rev if rev > 0 else 0)
        op_margins.append(op / rev if rev > 0 else 0)
        net_margins.append(ni / rev if rev > 0 else 0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=years_plot, y=revenues, name="Revenue ($B)",
                         marker_color='#4285F4', opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=years_plot, y=gross_margins, name="Gross Margin",
                             line=dict(color='#34A853', width=2), mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Scatter(x=years_plot, y=op_margins, name="Op Margin",
                             line=dict(color='#FBBC05', width=2), mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Scatter(x=years_plot, y=net_margins, name="Net Margin",
                             line=dict(color='#EA4335', width=2), mode='lines+markers'), secondary_y=True)
    fig.update_layout(title=f"{COMPANY_NAME} — Revenue & Margin Trends",
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=False)
    fig.update_yaxes(title_text="Margin %", tickformat=".0%", secondary_y=True)
    fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SEGMENT REVENUE MODEL
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 3 — Segment Revenue Model")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: ALPHABET SEGMENT-LEVEL REVENUE MODEL
#  Source: Alphabet 10-K FY2024 (estimates)
# ═══════════════════════════════════════════════════════════════════

segments = {
    "Google Search & Other": {
        "current_revenue_usdm" : 198_000,
        "growth_rate_bear"     : 0.06,
        "growth_rate_base"     : 0.10,
        "growth_rate_bull"     : 0.16,
        "fade_to"              : 0.04,
        "operating_margin_current"  : 0.40,
        "operating_margin_terminal" : 0.42,
        "tam_2024_usdm"  : 650_000,
        "tam_2030_usdm"  : 1_100_000,
        "market_share_pct": 38,
        "key_driver": "AI-powered search (Gemini), strong query volume, pricing power",
    },
    "YouTube Ads": {
        "current_revenue_usdm" : 36_000,
        "growth_rate_bear"     : 0.08,
        "growth_rate_base"     : 0.14,
        "growth_rate_bull"     : 0.22,
        "fade_to"              : 0.05,
        "operating_margin_current"  : 0.28,
        "operating_margin_terminal" : 0.32,
        "tam_2024_usdm"  : 200_000,
        "tam_2030_usdm"  : 380_000,
        "market_share_pct": 18,
        "key_driver": "Shorts monetisation, Connected TV, YouTube Premium growth",
    },
    "Google Network": {
        "current_revenue_usdm" : 31_000,
        "growth_rate_bear"     : -0.02,
        "growth_rate_base"     : 0.03,
        "growth_rate_bull"     : 0.07,
        "fade_to"              : 0.02,
        "operating_margin_current"  : 0.18,
        "operating_margin_terminal" : 0.18,
        "tam_2024_usdm"  : 150_000,
        "tam_2030_usdm"  : 170_000,
        "market_share_pct": 20,
        "key_driver": "Programmatic display; declining share to walled gardens",
    },
    "Subscriptions, Platforms & Devices": {
        "current_revenue_usdm" : 15_000,
        "growth_rate_bear"     : 0.06,
        "growth_rate_base"     : 0.10,
        "growth_rate_bull"     : 0.15,
        "fade_to"              : 0.04,
        "operating_margin_current"  : 0.22,
        "operating_margin_terminal" : 0.28,
        "tam_2024_usdm"  : 80_000,
        "tam_2030_usdm"  : 130_000,
        "market_share_pct": 19,
        "key_driver": "Google One subscriber growth, Play Store, Pixel hardware",
    },
    "Google Cloud (GCP)": {
        "current_revenue_usdm" : 43_000,
        "growth_rate_bear"     : 0.18,
        "growth_rate_base"     : 0.27,
        "growth_rate_bull"     : 0.38,
        "fade_to"              : 0.08,
        "operating_margin_current"  : 0.12,
        "operating_margin_terminal" : 0.28,
        "tam_2024_usdm"  : 300_000,
        "tam_2030_usdm"  : 800_000,
        "market_share_pct": 11,
        "key_driver": "AI/ML workloads, BigQuery, enterprise cloud migration; gaining share",
    },
    "Other Bets": {
        "current_revenue_usdm" : 1_700,
        "growth_rate_bear"     : 0.05,
        "growth_rate_base"     : 0.15,
        "growth_rate_bull"     : 0.35,
        "fade_to"              : 0.05,
        "operating_margin_current"  : -0.80,   # deeply loss-making
        "operating_margin_terminal" : 0.05,
        "tam_2024_usdm"  : 500_000,
        "tam_2030_usdm"  : 2_000_000,
        "market_share_pct": 0.3,
        "key_driver": "Waymo robotaxi commercialisation; Verily life sciences; long-dated option",
    },
}

# ── TAM Summary Table ─────────────────────────────────────────────
print("═" * 95)
print("  Alphabet Segment — TAM & Market Position")
print("═" * 95)
rows = []
for name, s in segments.items():
    rows.append({
        'Segment'          : name,
        'Rev FY24 ($M)'    : f"{s['current_revenue_usdm']:,}",
        'Market Share %'   : f"{s['market_share_pct']}%",
        'TAM 2024 ($M)'    : f"{s['tam_2024_usdm']:,}",
        'TAM 2030 ($M)'    : f"{s['tam_2030_usdm']:,}",
        'Base Growth'      : f"{s['growth_rate_base']:.0%}",
        'Terminal Op Mgn'  : f"{s['operating_margin_terminal']:.0%}",
        'Key Driver'       : s['key_driver'][:50],
    })
print(pd.DataFrame(rows).to_string(index=False))

# ── Projection engine ─────────────────────────────────────────────
def project_segment(seg_name, seg, scenario="base", years=FORECAST_YEARS):
    growth_key = f"growth_rate_{scenario}"
    initial_growth   = seg[growth_key]
    terminal_growth  = seg['fade_to']
    rows = []
    revenue       = seg['current_revenue_usdm']
    margin_now    = seg['operating_margin_current']
    margin_target = seg['operating_margin_terminal']
    for yr_offset in range(years):
        fade_frac = yr_offset / max(years - 1, 1)
        growth    = initial_growth * (1 - fade_frac) + terminal_growth * fade_frac
        margin    = margin_now     * (1 - fade_frac) + margin_target   * fade_frac
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
df_base = projections['base']
GOOGL_COLORS = {
    'Google Search & Other'                : '#4285F4',
    'YouTube Ads'                          : '#EA4335',
    'Google Network'                       : '#FBBC05',
    'Subscriptions, Platforms & Devices'   : '#34A853',
    'Google Cloud (GCP)'                   : '#00BCD4',
    'Other Bets'                           : '#9C27B0',
}
fig = go.Figure()
for seg_name, color in GOOGL_COLORS.items():
    seg_data = df_base[df_base['segment'] == seg_name]
    fig.add_trace(go.Bar(
        x=seg_data['year'],
        y=seg_data['revenue_usdm'] / 1000,
        name=seg_name, marker_color=color,
    ))
fig.update_layout(
    title=f"{COMPANY_NAME} — Projected Revenue by Segment (Base Case, $B)",
    xaxis_title="Year", yaxis_title="Revenue ($B)",
    barmode='stack', template="plotly_white",
    legend=dict(orientation="h", y=-0.25),
)
fig.show()

# ── Scenario comparison total revenue ────────────────────────────
fig2 = go.Figure()
for scenario, color in [('bear','#EA4335'), ('base','#4285F4'), ('bull','#34A853')]:
    df_s = projections[scenario].groupby('year')['revenue_usdm'].sum().reset_index()
    fig2.add_trace(go.Scatter(
        x=df_s['year'], y=df_s['revenue_usdm'] / 1000,
        name=scenario.capitalize(), mode='lines+markers',
        line=dict(color=color, width=2),
    ))
fig2.update_layout(
    title=f"{COMPANY_NAME} — Total Revenue: Bear / Base / Bull ($B)",
    xaxis_title="Year", yaxis_title="Revenue ($B)",
    template="plotly_white",
)
fig2.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — QUARTERLY TRENDS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 4 — Quarterly Revenue & Earnings Trends")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: QUARTERLY TRENDS
# ═══════════════════════════════════════════════════════════════════

if YF_AVAILABLE and quarterly_inc is not None and not quarterly_inc.empty:
    q_data = quarterly_inc.T.copy()
    q_data.index = pd.to_datetime(q_data.index)
    q_data = q_data.sort_index()

    q_rev = q_data['Total Revenue']  / 1e9 if 'Total Revenue'    in q_data.columns else pd.Series()
    q_op  = q_data['Operating Income'] / 1e9 if 'Operating Income' in q_data.columns else pd.Series()
    q_ni  = q_data['Net Income']      / 1e9 if 'Net Income'       in q_data.columns else pd.Series()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Quarterly Revenue ($B)", "Quarterly Operating & Net Income ($B)"])
    fig.add_trace(go.Bar(x=q_rev.index, y=q_rev.values, name='Revenue',
                         marker_color='#4285F4'), row=1, col=1)
    if len(q_op) > 0:
        fig.add_trace(go.Scatter(x=q_op.index, y=q_op.values, name='Op Income',
                                 line=dict(color='#FBBC05', width=2), mode='lines+markers'), row=2, col=1)
    if len(q_ni) > 0:
        fig.add_trace(go.Scatter(x=q_ni.index, y=q_ni.values, name='Net Income',
                                 line=dict(color='#34A853', width=2), mode='lines+markers'), row=2, col=1)
    fig.update_layout(title=f"{COMPANY_NAME} — Quarterly Financials",
                      template="plotly_white", height=600)
    fig.show()
else:
    print("⚠️  Quarterly data not available")
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
                             line=dict(color='#FBBC05', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_200'], name='EMA 200',
                             line=dict(color='#EA4335', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Upper'], showlegend=False,
                             line=dict(color='gray', width=0.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Lower'], showlegend=False,
                             line=dict(color='gray', width=0.5, dash='dot'),
                             fill='tonexty', fillcolor='rgba(128,128,128,0.08)'), row=1, col=1)

    colors_vol = ['#34A853' if c >= o else '#EA4335'
                  for c, o in zip(df_ta['Close'], df_ta['Open'])]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['Volume'], name='Volume',
                         marker_color=colors_vol, showlegend=False), row=2, col=1)

    macd_colors = ['#34A853' if v >= 0 else '#EA4335' for v in df_ta['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_Hist'],
                         marker_color=macd_colors, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'],
                             name='MACD', line=dict(color='#4285F4', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_Signal'],
                             name='Signal', line=dict(color='#FBBC05', width=1)), row=3, col=1)

    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], name='RSI',
                             line=dict(color='#9C27B0', width=1.5)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red",   row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    fig.update_layout(
        title=f"{COMPANY_NAME} ({TICKER}) — Technical Analysis (2Y)",
        template="plotly_white", height=950, showlegend=True,
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
    )
    fig.show()
else:
    print("⚠️  Price data not available")
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SEGMENT-LEVEL DCF
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 6 — Segment-Level DCF Valuation")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: SEGMENT-LEVEL DCF
# ═══════════════════════════════════════════════════════════════════

def run_dcf(scenario="base", wacc_override=None, tg_override=None):
    wacc = wacc_override or WACC
    tg   = tg_override   or TERMINAL_GROWTH

    df_proj = projections[scenario]
    years   = sorted(df_proj['year'].unique())

    annual_fcfs, annual_details = [], []

    da_pct    = LATEST_DA    / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.05
    capex_pct = LATEST_CAPEX / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.15
    wc_pct    = 0.01  # 1% of revenue to working capital

    for yr in years:
        yr_data        = df_proj[df_proj['year'] == yr]
        total_revenue  = yr_data['revenue_usdm'].sum()
        total_op_inc   = yr_data['op_income_usdm'].sum()

        sbc            = total_revenue * SBC_PCT_REVENUE
        corp_ga        = CORPORATE_GA_USD_M
        ebit           = total_op_inc - corp_ga - sbc
        taxes          = max(0, ebit * TAX_RATE)
        nopat          = ebit - taxes

        da             = total_revenue * da_pct
        capex          = total_revenue * capex_pct
        wc_change      = total_revenue * wc_pct
        fcf            = nopat + da - capex - wc_change

        annual_fcfs.append(fcf)
        annual_details.append({
            'year': yr, 'revenue': total_revenue, 'op_income': total_op_inc,
            'sbc': sbc, 'corp_ga': corp_ga, 'ebit': ebit, 'nopat': nopat,
            'da': da, 'capex': capex, 'fcf': fcf,
        })

    disc_factors  = [(1 + wacc) ** -(i+1) for i in range(len(annual_fcfs))]
    pv_fcfs       = np.array(annual_fcfs) * np.array(disc_factors)

    terminal_fcf  = annual_fcfs[-1] * (1 + tg)
    terminal_val  = terminal_fcf / (wacc - tg)
    pv_terminal   = terminal_val * disc_factors[-1]

    ev            = sum(pv_fcfs) + pv_terminal
    equity_val    = ev - NET_DEBT_USD_M
    diluted_sh    = SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS
    fv_per_share  = equity_val / diluted_sh if diluted_sh > 0 else 0
    upside        = (fv_per_share / CURRENT_PRICE - 1) if CURRENT_PRICE > 0 else 0

    t_detail      = annual_details[-1]
    implied_pe    = ev / t_detail['nopat']  if t_detail['nopat']  > 0 else 0
    implied_ev_r  = ev / t_detail['revenue'] if t_detail['revenue'] > 0 else 0

    return {
        'scenario'          : scenario,
        'ev_usdm'           : round(ev, 0),
        'equity_value_usdm' : round(equity_val, 0),
        'fair_value_per_share': round(fv_per_share, 2),
        'upside_pct'        : round(upside * 100, 1),
        'pv_fcfs_total'     : round(sum(pv_fcfs), 0),
        'pv_terminal'       : round(pv_terminal, 0),
        'terminal_pct_of_ev': round(pv_terminal / ev * 100, 1) if ev > 0 else 0,
        'implied_pe'        : round(implied_pe, 1),
        'implied_ev_rev'    : round(implied_ev_r, 1),
        'annual_details'    : pd.DataFrame(annual_details),
        'pv_fcfs'           : pv_fcfs,
        'wacc_used'         : wacc,
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
        'Implied P/E'        : f"{r['implied_pe']:.1f}x",
        'Implied EV/Rev'     : f"{r['implied_ev_rev']:.1f}x",
    })
df_scenarios = pd.DataFrame(summary_rows)
print("═" * 100)
print(f"  {COMPANY_NAME} — DCF Scenario Summary  (WACC: {WACC:.2%}  |  Terminal Growth: {TERMINAL_GROWTH:.1%})")
print(f"  Current Price: ${CURRENT_PRICE:,.2f}")
print("═" * 100)
print(df_scenarios.to_string(index=False))

# ── FCF Profile Chart ─────────────────────────────────────────────
fig = go.Figure()
for label, r in results.items():
    det = r['annual_details']
    fig.add_trace(go.Scatter(
        x=det['year'], y=det['fcf'] / 1000, name=label.capitalize(),
        mode='lines+markers',
        line=dict(color={'bear':'#EA4335','base':'#4285F4','bull':'#34A853'}[label], width=2),
    ))
fig.update_layout(title=f"{COMPANY_NAME} — Projected Free Cash Flow by Scenario ($B)",
                  xaxis_title="Year", yaxis_title="FCF ($B)",
                  template="plotly_white")
fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6b — SENSITIVITY HEATMAPS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 6b — Sensitivity Heatmaps")
code("""\
# ── WACC vs Terminal Growth ───────────────────────────────────────
wacc_range = np.arange(0.07, 0.14, 0.01)
tg_range   = np.arange(0.02, 0.045, 0.005)

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
    columns=[f"{tg:.1%}" for tg in tg_range],
)

fig = px.imshow(
    df_sens.values, text_auto='.0f',
    x=[f"{tg:.1%}" for tg in tg_range],
    y=[f"{w:.0%}"  for w  in wacc_range],
    color_continuous_scale='RdYlGn',
    title=f"{COMPANY_NAME} — Fair Value Sensitivity: WACC vs Terminal Growth",
    labels=dict(x="Terminal Growth Rate", y="WACC", color="Fair Value ($)"),
)
fig.update_layout(template="plotly_white")
fig.show()

# ── Revenue Growth vs Terminal Op Margin ─────────────────────────
rev_growth_range = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
margin_range     = [0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38]

matrix2 = []
for rg in rev_growth_range:
    row_vals = []
    for m in margin_range:
        old_base   = segments['Google Search & Other']['growth_rate_base']
        old_margin = segments['Google Search & Other']['operating_margin_terminal']
        # Temporarily patch blended growth via REVENUE_GROWTH_BASE proxy
        orig_growth = REVENUE_GROWTH_BASE
        # Simple scalar DCF (not full segment) — approximate sensitivity
        total_rev = LATEST_REVENUE
        fcfs = []
        for yr in range(FORECAST_YEARS):
            fade = yr / max(FORECAST_YEARS - 1, 1)
            g    = rg * (1 - fade) + TERMINAL_GROWTH * fade
            total_rev *= (1 + g)
            ebit  = total_rev * m - CORPORATE_GA_USD_M - total_rev * SBC_PCT_REVENUE
            nopat = max(0, ebit * (1 - TAX_RATE))
            da    = total_rev * (LATEST_DA    / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.05)
            capex = total_rev * (LATEST_CAPEX / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.15)
            fcfs.append(nopat + da - capex)
        disc    = [(1 + WACC) ** -(i+1) for i in range(FORECAST_YEARS)]
        pv_f    = sum(np.array(fcfs) * np.array(disc))
        tv      = fcfs[-1] * (1 + TERMINAL_GROWTH) / (WACC - TERMINAL_GROWTH)
        pv_tv   = tv * disc[-1]
        ev      = pv_f + pv_tv
        fv      = (ev - NET_DEBT_USD_M) / (SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS)
        row_vals.append(round(fv, 0))
    matrix2.append(row_vals)

df_sens2 = pd.DataFrame(
    matrix2,
    index=[f"{rg:.0%}" for rg in rev_growth_range],
    columns=[f"{m:.0%}" for m in margin_range],
)
fig2 = px.imshow(
    df_sens2.values, text_auto='.0f',
    x=[f"{m:.0%}"  for m  in margin_range],
    y=[f"{rg:.0%}" for rg in rev_growth_range],
    color_continuous_scale='RdYlGn',
    title=f"{COMPANY_NAME} — Fair Value Sensitivity: Revenue Growth vs Terminal Op Margin",
    labels=dict(x="Terminal Operating Margin", y="Revenue Growth (Base Year)", color="Fair Value ($)"),
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
#  SECTION 7: MONTE CARLO SIMULATION  ({MC_SIMULATIONS:,} runs)
# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)

def monte_carlo_dcf(n_sims=MC_SIMULATIONS):
    fair_values, sim_params = [], []
    for _ in range(n_sims):
        sim_growth = np.clip(np.random.normal(REVENUE_GROWTH_BASE,  MC_REVENUE_STDEV), -0.05, 0.50)
        sim_margin = np.clip(np.random.normal(OPERATING_MARGIN_TERMINAL, MC_MARGIN_STDEV), 0.05, 0.60)
        sim_wacc   = np.clip(np.random.normal(WACC, MC_WACC_STDEV), 0.05, 0.18)
        sim_tg     = np.random.uniform(0.02, 0.04)

        total_rev = LATEST_REVENUE
        fcfs = []
        for yr in range(FORECAST_YEARS):
            fade = yr / max(FORECAST_YEARS - 1, 1)
            g    = sim_growth * (1 - fade) + sim_tg * fade
            total_rev *= (1 + g)
            ebit  = total_rev * sim_margin - CORPORATE_GA_USD_M - total_rev * SBC_PCT_REVENUE
            nopat = max(0, ebit * (1 - TAX_RATE))
            da    = total_rev * (LATEST_DA    / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.05)
            capex = total_rev * (LATEST_CAPEX / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.15)
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
        if 0 < fv < CURRENT_PRICE * 15:
            fair_values.append(fv)
            sim_params.append({'growth': sim_growth, 'margin': sim_margin,
                               'wacc': sim_wacc, 'tg': sim_tg, 'fv': fv})
    return np.array(fair_values), pd.DataFrame(sim_params)

mc_values, mc_params = monte_carlo_dcf()

# ── Histogram ─────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Histogram(x=mc_values, nbinsx=120, name='Simulated Fair Values',
                            marker_color='#4285F4', opacity=0.75))
fig.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red",
              annotation_text=f"Current: ${CURRENT_PRICE:,.0f}")
fig.add_vline(x=np.median(mc_values), line_dash="dash", line_color="#34A853",
              annotation_text=f"Median: ${np.median(mc_values):,.0f}")
fig.update_layout(
    title=f"{COMPANY_NAME} — Monte Carlo Fair Value Distribution ({len(mc_values):,} simulations)",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Frequency",
    template="plotly_white",
)
fig.show()

# ── Statistics ────────────────────────────────────────────────────
pct_upside = (mc_values > CURRENT_PRICE).mean() * 100
print(f"\\n📊  Monte Carlo Results  ({len(mc_values):,} valid simulations)")
print(f"   Probability of upside (> ${CURRENT_PRICE:,.0f}): {pct_upside:.1f}%")
print(f"   10th pct:  ${np.percentile(mc_values, 10):,.0f}")
print(f"   25th pct:  ${np.percentile(mc_values, 25):,.0f}")
print(f"   Median:    ${np.median(mc_values):,.0f}")
print(f"   75th pct:  ${np.percentile(mc_values, 75):,.0f}")
print(f"   90th pct:  ${np.percentile(mc_values, 90):,.0f}")
print(f"   Mean:      ${np.mean(mc_values):,.0f}")
print(f"   Std Dev:   ${np.std(mc_values):,.0f}")

# ── CDF ───────────────────────────────────────────────────────────
sorted_vals = np.sort(mc_values)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=sorted_vals, y=cdf, mode='lines',
                              line=dict(color='#4285F4', width=2)))
fig_cdf.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red",
                  annotation_text="Current Price")
fig_cdf.update_layout(
    title=f"{COMPANY_NAME} — Cumulative Distribution of Fair Value",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Cumulative Probability",
    template="plotly_white",
)
fig_cdf.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — BUYBACK & CAPITAL RETURN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 8 — Capital Return & Buyback Analysis")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: CAPITAL RETURN ANALYSIS
#  Alphabet does not pay a dividend; returns capital primarily via buybacks.
# ═══════════════════════════════════════════════════════════════════

# FY2024 actual buybacks ~$62B; assume forward trajectory
buyback_schedule_usdm = {
    2024: 62_000, 2025: 65_000, 2026: 68_000, 2027: 70_000, 2028: 72_000,
    2029: 74_000, 2030: 76_000, 2031: 78_000, 2032: 80_000, 2033: 82_000,
}

df_buyback = pd.DataFrame({
    'Year'        : list(buyback_schedule_usdm.keys()),
    'Buyback ($B)': [v/1000 for v in buyback_schedule_usdm.values()],
})

# Cumulative reduction in share count
base_shares = SHARES_OUT_M
running_shares = [base_shares]
for yr, amount_m in list(buyback_schedule_usdm.items())[:-1]:
    price_assumption = CURRENT_PRICE * (1.08 ** (yr - 2024 + 1))  # assume 8%/yr price appreciation
    shares_repurchased = (amount_m * 1e6) / (price_assumption if price_assumption > 0 else 1) / 1e6
    dilution = running_shares[-1] * ANNUAL_DILUTION_PCT
    running_shares.append(running_shares[-1] + dilution - shares_repurchased)

df_buyback['Shares Remaining (M)'] = [round(s, 0) for s in running_shares]
df_buyback['Pct of Shares Bought'] = [b * 1e3 / (s * CURRENT_PRICE) * 100
                                       for b, s in zip(df_buyback['Buyback ($B)'],
                                                       running_shares)]
print("═" * 65)
print("  Alphabet Capital Return — Buyback Schedule (Estimated)")
print("═" * 65)
print(df_buyback.to_string(index=False))

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Bar(x=df_buyback['Year'], y=df_buyback['Buyback ($B)'],
                     name='Buybacks ($B)', marker_color='#4285F4', opacity=0.7), secondary_y=False)
fig.add_trace(go.Scatter(x=df_buyback['Year'], y=df_buyback['Shares Remaining (M)'],
                         name='Shares Outstanding (M)', mode='lines+markers',
                         line=dict(color='#EA4335', width=2)), secondary_y=True)
fig.update_layout(title=f"{COMPANY_NAME} — Share Buyback & Dilution Impact",
                  template="plotly_white")
fig.update_yaxes(title_text="Buybacks ($B)", secondary_y=False)
fig.update_yaxes(title_text="Shares Outstanding (M)", secondary_y=True)
fig.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — PEER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 9 — Peer / Relative Valuation")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: PEER COMPARISON
#  Peers: mega-cap platform + digital advertising universe
# ═══════════════════════════════════════════════════════════════════

PEER_TICKERS = ['META', 'MSFT', 'AMZN', 'AAPL', 'NFLX', 'TTD']

if YF_AVAILABLE:
    peer_data = []
    for t in [TICKER] + PEER_TICKERS:
        try:
            p    = yf.Ticker(t)
            pi   = p.info
            p_inc = p.income_stmt
            p_cf  = p.cash_flow

            rev   = p_inc.loc['Total Revenue'].dropna().iloc[0]    / 1e9 if 'Total Revenue'    in p_inc.index else 0
            oi    = p_inc.loc['Operating Income'].dropna().iloc[0] / 1e9 if 'Operating Income' in p_inc.index else 0
            ni    = p_inc.loc['Net Income'].dropna().iloc[0]       / 1e9 if 'Net Income'       in p_inc.index else 0
            ocf   = p_cf.loc['Operating Cash Flow'].dropna().iloc[0]   / 1e9 if 'Operating Cash Flow'   in p_cf.index else 0
            capex = abs(p_cf.loc['Capital Expenditure'].dropna().iloc[0]) / 1e9 if 'Capital Expenditure' in p_cf.index else 0
            fcf   = ocf - capex

            mcap  = pi.get('marketCap', 0) / 1e9
            ev_v  = pi.get('enterpriseValue', 0) / 1e9

            rev_vals = p_inc.loc['Total Revenue'].dropna() if 'Total Revenue' in p_inc.index else pd.Series()
            rev_growth = (rev_vals.iloc[0] / rev_vals.iloc[1] - 1) if len(rev_vals) >= 2 and rev_vals.iloc[1] != 0 else 0

            peer_data.append({
                'Ticker'        : t,
                'Mkt Cap ($B)'  : round(mcap, 1),
                'EV ($B)'       : round(ev_v, 1),
                'Revenue ($B)'  : round(rev, 1),
                'Rev Growth'    : f"{rev_growth:.0%}",
                'Op Margin'     : f"{oi/rev:.0%}" if rev > 0 else "N/A",
                'FCF ($B)'      : round(fcf, 1),
                'P/E'           : round(pi.get('trailingPE', 0) or 0, 1),
                'EV/EBITDA'     : round(pi.get('enterpriseToEbitda', 0) or 0, 1),
                'EV/Rev'        : round(ev_v / rev, 1) if rev > 0 else 0,
                'P/FCF'         : round(mcap / fcf, 1) if fcf > 0 else 0,
                '_rev_g'        : rev_growth,
                '_ev_r'         : ev_v / rev if rev > 0 else 0,
                '_mcap'         : mcap,
            })
        except Exception as e:
            print(f"⚠️  Could not fetch {t}: {e}")

    df_peers = pd.DataFrame(peer_data)
    print("═" * 100)
    print("  Alphabet Peer Comparison")
    print("═" * 100)
    display_cols = [c for c in df_peers.columns if not c.startswith('_')]
    print(df_peers[display_cols].to_string(index=False))

    # Bubble chart: EV/Revenue vs Rev Growth (size = Market Cap)
    fig = px.scatter(
        df_peers, x='_rev_g', y='_ev_r', size='_mcap', text='Ticker',
        color='Ticker',
        title="Peer Comparison: EV/Revenue vs Revenue Growth",
        labels={'_rev_g': 'Revenue Growth (YoY)', '_ev_r': 'EV/Revenue', '_mcap': 'Market Cap ($B)'},
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(template='plotly_white', xaxis_tickformat='.0%', showlegend=False)
    fig.show()
else:
    print("⚠️  yfinance not available — peer comparison skipped")
""")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — FCF WATERFALL
# ─────────────────────────────────────────────────────────────────────────────
md("## Section 10 — FCF Bridge Waterfall (Base Case, Year 3)")
code("""\
# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: FCF BRIDGE WATERFALL
# ═══════════════════════════════════════════════════════════════════

base_details = results['base']['annual_details']
yr3 = base_details.iloc[2]   # Year 3 projection

labels = ['Revenue', 'Segment OpEx', 'SBC', 'Corp G&A', 'EBIT',
          'Tax', 'NOPAT', 'D&A', 'Capex', 'WC Change', 'FCF']
values = [
    yr3['revenue'],
    -(yr3['revenue'] - yr3['op_income']),
    -yr3['sbc'],
    -yr3['corp_ga'],
     yr3['ebit'],
    -(yr3['ebit'] * TAX_RATE),
     yr3['nopat'],
     yr3['da'],
    -yr3['capex'],
    -(yr3['revenue'] * 0.01),   # WC change estimate
     yr3['fcf'],
]
measures = ['absolute','relative','relative','relative','total',
            'relative','total','relative','relative','relative','total']

fig = go.Figure(go.Waterfall(
    name="FCF Bridge", orientation="v",
    measure=measures, x=labels, y=values,
    connector={"line": {"color": "#5f6368"}},
    increasing={"marker": {"color": "#34A853"}},
    decreasing={"marker": {"color": "#EA4335"}},
    totals   ={"marker": {"color": "#4285F4"}},
))
fig.update_layout(
    title=f"{COMPANY_NAME} — FCF Bridge (Base Case, Year 3 of Projection, $M)",
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

print("╔" + "═"*78 + "╗")
print(f"  ALPHABET INC. (GOOGL) — INVESTMENT SUMMARY")
print("╠" + "═"*78 + "╣")
print(f"  Current Price        : ${CURRENT_PRICE:>10,.2f}")
print(f"  Market Cap           : ${MARKET_CAP_B:>10,.1f}B")
print(f"  Net Cash Position    : ${-NET_DEBT_USD_M/1000:>10,.1f}B  (net cash)")
print()
print(f"  ── DCF Valuation (WACC: {WACC:.2%}) ─────────────────────────────────")
print(f"  Bear Case Fair Value : ${results['bear']['fair_value_per_share']:>10,.2f}  "
      f"({results['bear']['upside_pct']:+.1f}%)")
print(f"  Base Case Fair Value : ${base_r['fair_value_per_share']:>10,.2f}  "
      f"({base_r['upside_pct']:+.1f}%)  ← PRIMARY")
print(f"  Bull Case Fair Value : ${results['bull']['fair_value_per_share']:>10,.2f}  "
      f"({results['bull']['upside_pct']:+.1f}%)")
print()
print(f"  ── Monte Carlo (10,000 sims) ────────────────────────────────────────")
print(f"  Median Fair Value    : ${np.median(mc_values):>10,.2f}")
print(f"  Probability Upside   : {(mc_values > CURRENT_PRICE).mean()*100:>9.1f}%")
print()
print(f"  ── Key Assumptions ──────────────────────────────────────────────────")
print(f"  Revenue Growth (Base): {REVENUE_GROWTH_BASE:.0%} → fades to {FADE_GROWTH_TO:.0%} by Year 10")
print(f"  Terminal Op Margin   : {OPERATING_MARGIN_TERMINAL:.0%}")
print(f"  WACC                 : {WACC:.2%}  (Beta: {BETA:.2f})")
print(f"  Terminal Growth      : {TERMINAL_GROWTH:.1%}")
print(f"  SBC / Revenue        : {SBC_PCT_REVENUE:.1%}")
print()
print(f"  ── Investment Thesis ────────────────────────────────────────────────")
thesis_lines = [
    "BULL: AI integration (Gemini) boosts Search monetisation & GCP share gains.",
    "      YouTube Shorts monetisation + Connected TV drive 14%+ YouTube growth.",
    "      Net cash + ~$65B/year buybacks reduce share count materially.",
    "",
    "BEAR: AI search disruption (Perplexity, ChatGPT) erodes Search query share.",
    "      Regulatory risk: DOJ antitrust ruling could force Search distribution change.",
    "      GCP margin expansion slower than expected; Other Bets remain cash sinks.",
    "",
    "RISK: Beta ~1.1 — moderate market sensitivity. China/geo risk modest for GOOGL.",
    "      Terminal value is ~" + f"{base_r['terminal_pct_of_ev']:.0f}" +
      "% of base EV — stress-test WACC & terminal growth.",
]
for line in thesis_lines:
    print(f"  {line}")
print("╚" + "═"*78 + "╝")
""")

# ─────────────────────────────────────────────────────────────────────────────
# FINALISE
# ─────────────────────────────────────────────────────────────────────────────
code("""\
import nbformat

nb.cells = cells

OUTPUT_PATH = "GOOGL_valuation.ipynb"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook saved: {OUTPUT_PATH}")
""")

# ── Write the notebook ───────────────────────────────────────────────────────
nb.cells = cells

OUTPUT_PATH = "GOOGL_valuation.ipynb"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook saved: {OUTPUT_PATH}")
