# Notebook Template Reference — Tech Company Valuer

Use this as the base pattern for building the notebook with `nbformat`.

## Python Skeleton

```python
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
}
cells = []

def md(text):
    cells.append(new_markdown_cell(text))

def code(src):
    cells.append(new_code_cell(src))
```

---

## Cell 0 — Imports & Config

```python
# ═══════════════════════════════════════════════════════════════════
#  TECH COMPANY VALUATION — [COMPANY NAME] ([TICKER])
#  Built by Claude · Edit the variables below to customise
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Attempt yfinance import ───────────────────────────────────────
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    print("⚠️ yfinance not installed. Run: pip install yfinance")
    YF_AVAILABLE = False

# ── COMPANY ──────────────────────────────────────────────────────
TICKER          = "[TICKER]"
COMPANY_NAME    = "[Company Name]"

# ── These will be populated from yfinance if available ───────────
SHARES_OUT_M    = 0       # million shares outstanding (diluted)
CURRENT_PRICE   = 0.0     # USD/share
MARKET_CAP_B    = 0.0     # USD billions
NET_DEBT_USD_M  = 0       # net debt (negative = net cash)
BETA            = 1.2     # from yfinance

# ── REVENUE GROWTH ASSUMPTIONS (annual %) ────────────────────────
REVENUE_GROWTH_BEAR  = 0.08
REVENUE_GROWTH_BASE  = 0.15   # ← primary assumption; edit me
REVENUE_GROWTH_BULL  = 0.25

# ── MARGIN ASSUMPTIONS ───────────────────────────────────────────
OPERATING_MARGIN_TERMINAL = 0.30  # long-run operating margin target
FCF_CONVERSION             = 0.85  # FCF as % of operating income at maturity

# ── DISCOUNT RATE ────────────────────────────────────────────────
RISK_FREE_RATE    = 0.043   # will attempt to pull from FRED
EQUITY_RISK_PREM  = 0.055   # Damodaran ERP estimate
COST_OF_DEBT      = 0.045
DEBT_WEIGHT       = 0.05    # most mega-cap tech is equity-heavy
TAX_RATE          = 0.15    # effective tax rate from financials

# WACC calculated below — or override directly
WACC_OVERRIDE     = None    # set a float (e.g. 0.10) to override calc

# ── DCF HORIZON ──────────────────────────────────────────────────
FORECAST_YEARS      = 10
TERMINAL_GROWTH     = 0.03
FADE_GROWTH_TO      = 0.04   # revenue growth fades to this by final year

# ── MONTE CARLO ──────────────────────────────────────────────────
MC_SIMULATIONS      = 10_000
MC_REVENUE_STDEV    = 0.08
MC_MARGIN_STDEV     = 0.03
MC_WACC_STDEV       = 0.01

# ── SHARE-BASED COMPENSATION ─────────────────────────────────────
SBC_PCT_REVENUE     = 0.08   # SBC as % of revenue (dilution cost)
ANNUAL_DILUTION_PCT = 0.015  # net annual share dilution

# ── CORPORATE G&A (USD millions per year, above segment opex) ────
CORPORATE_GA_USD_M  = 500

print("✅ Config loaded. Edit variables above and re-run to update all outputs.")
```

---

## Cell 1 — Fetch Live Data from yfinance

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: FETCH LIVE DATA
# ═══════════════════════════════════════════════════════════════════

if YF_AVAILABLE:
    stock = yf.Ticker(TICKER)
    info = stock.info

    # ── Company basics ────────────────────────────────────────────
    COMPANY_NAME    = info.get('longName', COMPANY_NAME)
    CURRENT_PRICE   = info.get('currentPrice', info.get('regularMarketPrice', 0))
    MARKET_CAP_B    = info.get('marketCap', 0) / 1e9
    SHARES_OUT_M    = info.get('sharesOutstanding', 0) / 1e6
    BETA            = info.get('beta', BETA)
    
    # ── Net debt ──────────────────────────────────────────────────
    total_debt = info.get('totalDebt', 0) or 0
    total_cash = info.get('totalCash', 0) or 0
    NET_DEBT_USD_M  = (total_debt - total_cash) / 1e6
    
    # ── Financial statements ──────────────────────────────────────
    income_stmt    = stock.income_stmt          # annual
    balance_sheet  = stock.balance_sheet        # annual
    cash_flow      = stock.cash_flow            # annual
    quarterly_inc  = stock.quarterly_income_stmt
    
    # ── Price history (2 years) ───────────────────────────────────
    price_data = stock.history(period="2y")
    
    # ── Analyst estimates ─────────────────────────────────────────
    try:
        analyst_targets = {
            'target_mean': info.get('targetMeanPrice', None),
            'target_low': info.get('targetLowPrice', None),
            'target_high': info.get('targetHighPrice', None),
            'num_analysts': info.get('numberOfAnalystOpinions', None),
            'recommendation': info.get('recommendationKey', None),
        }
    except:
        analyst_targets = {}
    
    # ── Insider transactions ──────────────────────────────────────
    try:
        insider_txns = stock.insider_transactions
    except:
        insider_txns = pd.DataFrame()
    
    # ── Institutional holders ─────────────────────────────────────
    try:
        inst_holders = stock.institutional_holders
    except:
        inst_holders = pd.DataFrame()
    
    # ── Extract key financials from income statement ──────────────
    def safe_get(df, label, default=0):
        """Safely extract most recent value from a financial statement row."""
        if df is None or df.empty:
            return default
        for name in [label]:
            if name in df.index:
                val = df.loc[name].dropna()
                if len(val) > 0:
                    return float(val.iloc[0])
        return default
    
    LATEST_REVENUE      = safe_get(income_stmt, 'Total Revenue') / 1e6  # USD millions
    LATEST_GROSS_PROFIT = safe_get(income_stmt, 'Gross Profit') / 1e6
    LATEST_OP_INCOME    = safe_get(income_stmt, 'Operating Income') / 1e6
    LATEST_NET_INCOME   = safe_get(income_stmt, 'Net Income') / 1e6
    LATEST_FCF          = (safe_get(cash_flow, 'Operating Cash Flow') - 
                           abs(safe_get(cash_flow, 'Capital Expenditure'))) / 1e6
    LATEST_SBC          = safe_get(cash_flow, 'Stock Based Compensation') / 1e6
    LATEST_CAPEX        = abs(safe_get(cash_flow, 'Capital Expenditure')) / 1e6
    LATEST_DA           = safe_get(cash_flow, 'Depreciation And Amortization') / 1e6
    
    # ── Effective tax rate from financials ─────────────────────────
    tax_provision = safe_get(income_stmt, 'Tax Provision')
    pretax_income = safe_get(income_stmt, 'Pretax Income')
    if pretax_income > 0 and tax_provision > 0:
        TAX_RATE = min(tax_provision / pretax_income, 0.30)
    
    # ── SBC as % of revenue ───────────────────────────────────────
    if LATEST_REVENUE > 0 and LATEST_SBC > 0:
        SBC_PCT_REVENUE = LATEST_SBC / LATEST_REVENUE
    
    # ── Calculate WACC ────────────────────────────────────────────
    cost_of_equity = RISK_FREE_RATE + BETA * EQUITY_RISK_PREM
    equity_weight  = 1 - DEBT_WEIGHT
    WACC_CALC = (equity_weight * cost_of_equity + 
                 DEBT_WEIGHT * COST_OF_DEBT * (1 - TAX_RATE))
    WACC = WACC_OVERRIDE if WACC_OVERRIDE else WACC_CALC
    
    print(f"📊 {COMPANY_NAME} ({TICKER})")
    print(f"   Price: ${CURRENT_PRICE:,.2f}  |  Mkt Cap: ${MARKET_CAP_B:,.1f}B  |  Beta: {BETA:.2f}")
    print(f"   Revenue: ${LATEST_REVENUE:,.0f}M  |  Op Income: ${LATEST_OP_INCOME:,.0f}M  |  FCF: ${LATEST_FCF:,.0f}M")
    print(f"   WACC: {WACC:.1%}  |  Tax Rate: {TAX_RATE:.1%}  |  SBC/Rev: {SBC_PCT_REVENUE:.1%}")
    print(f"   Net Debt: ${NET_DEBT_USD_M:,.0f}M  |  Shares: {SHARES_OUT_M:,.0f}M")
else:
    print("⚠️ yfinance not available — using manual config values above")
    WACC = WACC_OVERRIDE or 0.10
```

---

## Section 2 — Company Overview Dashboard

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: COMPANY OVERVIEW
# ═══════════════════════════════════════════════════════════════════

# 5-Year Financial Summary Table
if YF_AVAILABLE and income_stmt is not None and not income_stmt.empty:
    years = income_stmt.columns[:5]
    summary_data = []
    for yr in years:
        rev = income_stmt.loc['Total Revenue', yr] / 1e6 if 'Total Revenue' in income_stmt.index else 0
        gp  = income_stmt.loc['Gross Profit', yr] / 1e6 if 'Gross Profit' in income_stmt.index else 0
        op  = income_stmt.loc['Operating Income', yr] / 1e6 if 'Operating Income' in income_stmt.index else 0
        ni  = income_stmt.loc['Net Income', yr] / 1e6 if 'Net Income' in income_stmt.index else 0
        
        summary_data.append({
            'Year': yr.strftime('%Y') if hasattr(yr, 'strftime') else str(yr),
            'Revenue ($M)': f"{rev:,.0f}",
            'Gross Profit ($M)': f"{gp:,.0f}",
            'Gross Margin': f"{gp/rev:.1%}" if rev > 0 else "N/A",
            'Op Income ($M)': f"{op:,.0f}",
            'Op Margin': f"{op/rev:.1%}" if rev > 0 else "N/A",
            'Net Income ($M)': f"{ni:,.0f}",
            'Net Margin': f"{ni/rev:.1%}" if rev > 0 else "N/A",
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("═" * 80)
    print(f"  {COMPANY_NAME} — 5-Year Financial Summary")
    print("═" * 80)
    print(df_summary.to_string(index=False))
    print()

# ── Historical Revenue & Margin Trends Chart ──────────────────────
if YF_AVAILABLE and income_stmt is not None and not income_stmt.empty:
    years_plot = []
    revenues = []
    op_margins = []
    net_margins = []
    gross_margins = []
    
    for yr in reversed(list(income_stmt.columns[:5])):
        rev = income_stmt.loc['Total Revenue', yr] / 1e9 if 'Total Revenue' in income_stmt.index else 0
        gp  = income_stmt.loc['Gross Profit', yr] / 1e9 if 'Gross Profit' in income_stmt.index else 0
        op  = income_stmt.loc['Operating Income', yr] / 1e9 if 'Operating Income' in income_stmt.index else 0
        ni  = income_stmt.loc['Net Income', yr] / 1e9 if 'Net Income' in income_stmt.index else 0
        
        yr_str = yr.strftime('%Y') if hasattr(yr, 'strftime') else str(yr)
        years_plot.append(yr_str)
        revenues.append(rev)
        gross_margins.append(gp/rev if rev > 0 else 0)
        op_margins.append(op/rev if rev > 0 else 0)
        net_margins.append(ni/rev if rev > 0 else 0)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=years_plot, y=revenues, name="Revenue ($B)", 
                         marker_color='#2196F3', opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=years_plot, y=gross_margins, name="Gross Margin",
                             line=dict(color='#4CAF50', width=2), mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Scatter(x=years_plot, y=op_margins, name="Op Margin",
                             line=dict(color='#FF9800', width=2), mode='lines+markers'), secondary_y=True)
    fig.add_trace(go.Scatter(x=years_plot, y=net_margins, name="Net Margin",
                             line=dict(color='#9C27B0', width=2), mode='lines+markers'), secondary_y=True)
    
    fig.update_layout(title=f"{COMPANY_NAME} — Revenue & Margin Trends",
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text="Revenue ($B)", secondary_y=False)
    fig.update_yaxes(title_text="Margin %", tickformat=".0%", secondary_y=True)
    fig.show()
```

---

## Section 3 — Segment Revenue Model

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: SEGMENT-LEVEL REVENUE MODEL
# ═══════════════════════════════════════════════════════════════════

# ── Segment Configuration (EDIT THESE FROM RESEARCH) ──────────────
# This is the heart of the model — populate from SEC filings & research

segments = {
    "Segment A": {
        "current_revenue_usdm": 0,       # FILL from 10-K
        "growth_rate_bear": 0.05,
        "growth_rate_base": 0.12,         # ← edit per segment
        "growth_rate_bull": 0.20,
        "fade_to": 0.04,                  # terminal segment growth
        "operating_margin_current": 0.30, # current segment margin
        "operating_margin_terminal": 0.35,# long-run margin target
        "tam_2024_usdm": 50_000,          # ESTIMATED
        "tam_2030_usdm": 120_000,         # ESTIMATED
        "market_share_pct": 25,           # ESTIMATED
        "key_driver": "Description of growth driver",
    },
    # ... add more segments from research
}

def project_segment(seg_name, seg, scenario="base", years=FORECAST_YEARS):
    """Project segment revenue with fading growth rate."""
    growth_key = f"growth_rate_{scenario}"
    initial_growth = seg[growth_key]
    terminal_growth = seg['fade_to']
    
    rows = []
    revenue = seg['current_revenue_usdm']
    margin_current = seg['operating_margin_current']
    margin_terminal = seg['operating_margin_terminal']
    
    for yr_offset in range(years):
        # Linear fade from initial growth to terminal
        fade_frac = yr_offset / max(years - 1, 1)
        growth = initial_growth * (1 - fade_frac) + terminal_growth * fade_frac
        margin = margin_current * (1 - fade_frac) + margin_terminal * fade_frac
        
        revenue *= (1 + growth)
        op_income = revenue * margin
        
        rows.append({
            'year': datetime.now().year + yr_offset + 1,
            'segment': seg_name,
            'revenue_usdm': revenue,
            'growth_rate': growth,
            'op_margin': margin,
            'op_income_usdm': op_income,
        })
    
    return pd.DataFrame(rows)

# Project all segments for all scenarios
projections = {}
for scenario in ['bear', 'base', 'bull']:
    dfs = []
    for seg_name, seg in segments.items():
        dfs.append(project_segment(seg_name, seg, scenario))
    projections[scenario] = pd.concat(dfs, ignore_index=True)

# ── Stacked revenue chart (base case) ────────────────────────────
df_base = projections['base']
fig = go.Figure()
colors = px.colors.qualitative.Set2
for i, seg_name in enumerate(segments.keys()):
    seg_data = df_base[df_base['segment'] == seg_name]
    fig.add_trace(go.Bar(
        x=seg_data['year'], y=seg_data['revenue_usdm'] / 1000,
        name=seg_name, marker_color=colors[i % len(colors)]
    ))

fig.update_layout(
    title=f"{COMPANY_NAME} — Projected Revenue by Segment (Base Case, $B)",
    xaxis_title="Year", yaxis_title="Revenue ($B)",
    barmode='stack', template="plotly_white",
    legend=dict(orientation="h", y=-0.2)
)
fig.show()
```

---

## Section 5 — Technical Analysis

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: TECHNICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

if YF_AVAILABLE and price_data is not None and len(price_data) > 50:
    df_ta = price_data.copy()
    
    # ── EMAs ──────────────────────────────────────────────────────
    df_ta['EMA_50']  = df_ta['Close'].ewm(span=50, adjust=False).mean()
    df_ta['EMA_200'] = df_ta['Close'].ewm(span=200, adjust=False).mean()
    
    # ── MACD ──────────────────────────────────────────────────────
    ema_12 = df_ta['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_ta['Close'].ewm(span=26, adjust=False).mean()
    df_ta['MACD'] = ema_12 - ema_26
    df_ta['MACD_Signal'] = df_ta['MACD'].ewm(span=9, adjust=False).mean()
    df_ta['MACD_Hist'] = df_ta['MACD'] - df_ta['MACD_Signal']
    
    # ── RSI ───────────────────────────────────────────────────────
    delta = df_ta['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df_ta['RSI'] = 100 - (100 / (1 + rs))
    
    # ── Bollinger Bands ───────────────────────────────────────────
    df_ta['BB_Mid'] = df_ta['Close'].rolling(20).mean()
    bb_std = df_ta['Close'].rolling(20).std()
    df_ta['BB_Upper'] = df_ta['BB_Mid'] + 2 * bb_std
    df_ta['BB_Lower'] = df_ta['BB_Mid'] - 2 * bb_std
    
    # ── Multi-panel chart ─────────────────────────────────────────
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.5, 0.15, 0.15, 0.2],
                        subplot_titles=["Price & EMAs", "Volume", "MACD", "RSI"])
    
    # Panel 1: Candlestick + EMAs + Bollinger
    fig.add_trace(go.Candlestick(x=df_ta.index, open=df_ta['Open'], high=df_ta['High'],
                                  low=df_ta['Low'], close=df_ta['Close'], name='OHLC',
                                  showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_50'], name='EMA 50',
                             line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['EMA_200'], name='EMA 200',
                             line=dict(color='red', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Upper'], name='BB Upper',
                             line=dict(color='gray', width=0.5, dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['BB_Lower'], name='BB Lower',
                             line=dict(color='gray', width=0.5, dash='dot'), fill='tonexty',
                             fillcolor='rgba(128,128,128,0.1)', showlegend=False), row=1, col=1)
    
    # Panel 2: Volume
    colors_vol = ['#4CAF50' if c >= o else '#F44336' 
                  for c, o in zip(df_ta['Close'], df_ta['Open'])]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['Volume'], name='Volume',
                         marker_color=colors_vol, showlegend=False), row=2, col=1)
    
    # Panel 3: MACD
    macd_colors = ['#4CAF50' if v >= 0 else '#F44336' for v in df_ta['MACD_Hist']]
    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['MACD_Hist'], name='MACD Hist',
                         marker_color=macd_colors, showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD'], name='MACD',
                             line=dict(color='#2196F3', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['MACD_Signal'], name='Signal',
                             line=dict(color='#FF9800', width=1)), row=3, col=1)
    
    # Panel 4: RSI
    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['RSI'], name='RSI',
                             line=dict(color='#9C27B0', width=1.5)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
    
    fig.update_layout(
        title=f"{COMPANY_NAME} ({TICKER}) — Technical Analysis",
        template="plotly_white", height=900, showlegend=True,
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False
    )
    fig.show()
```

---

## Section 6 — Segment-Level DCF

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: SEGMENT-LEVEL DCF VALUATION
# ═══════════════════════════════════════════════════════════════════

def run_dcf(scenario="base", wacc_override=None, tg_override=None):
    """
    Full segment-level DCF with SBC deduction and dilution adjustment.
    Returns dict with EV, equity value, fair value per share, and details.
    """
    wacc = wacc_override or WACC
    tg   = tg_override or TERMINAL_GROWTH
    
    # ── Project segments ──────────────────────────────────────────
    df_proj = projections[scenario]
    years = sorted(df_proj['year'].unique())
    
    annual_fcfs = []
    annual_details = []
    
    for yr in years:
        yr_data = df_proj[df_proj['year'] == yr]
        total_revenue = yr_data['revenue_usdm'].sum()
        total_op_income = yr_data['op_income_usdm'].sum()
        
        # Corporate-level adjustments
        sbc = total_revenue * SBC_PCT_REVENUE
        corporate_ga = CORPORATE_GA_USD_M
        ebit = total_op_income - corporate_ga - sbc
        
        # Tax
        taxes = max(0, ebit * TAX_RATE)
        nopat = ebit - taxes
        
        # Approximate FCF: NOPAT + D&A - Capex - Working Capital change
        # Use latest ratios as proxy
        da_pct = LATEST_DA / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.05
        capex_pct = LATEST_CAPEX / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.10
        wc_change_pct = 0.02  # assume 2% of revenue growth goes to WC
        
        da = total_revenue * da_pct
        capex = total_revenue * capex_pct
        wc_change = total_revenue * wc_change_pct
        
        fcf = nopat + da - capex - wc_change
        
        annual_fcfs.append(fcf)
        annual_details.append({
            'year': yr, 'revenue': total_revenue, 'op_income': total_op_income,
            'sbc': sbc, 'ebit': ebit, 'nopat': nopat, 'da': da,
            'capex': capex, 'fcf': fcf
        })
    
    # ── Discount FCFs ─────────────────────────────────────────────
    discount_factors = [(1 + wacc) ** -(i+1) for i in range(len(annual_fcfs))]
    pv_fcfs = np.array(annual_fcfs) * np.array(discount_factors)
    
    # ── Terminal Value ────────────────────────────────────────────
    terminal_fcf = annual_fcfs[-1] * (1 + tg)
    terminal_value = terminal_fcf / (wacc - tg)
    pv_terminal = terminal_value * discount_factors[-1]
    
    # ── Enterprise & Equity Value ─────────────────────────────────
    enterprise_value = sum(pv_fcfs) + pv_terminal
    equity_value = enterprise_value - NET_DEBT_USD_M
    
    # Adjust shares for dilution over forecast period
    diluted_shares = SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS
    fair_value_per_share = equity_value / diluted_shares if diluted_shares > 0 else 0
    
    upside = (fair_value_per_share / CURRENT_PRICE - 1) if CURRENT_PRICE > 0 else 0
    
    # Implied multiples at fair value
    terminal_revenue = annual_details[-1]['revenue']
    terminal_ebit = annual_details[-1]['ebit']
    implied_pe = enterprise_value / (annual_details[-1]['nopat']) if annual_details[-1]['nopat'] > 0 else 0
    implied_ev_rev = enterprise_value / terminal_revenue if terminal_revenue > 0 else 0
    
    return {
        'scenario': scenario,
        'ev_usdm': round(enterprise_value, 0),
        'equity_value_usdm': round(equity_value, 0),
        'fair_value_per_share': round(fair_value_per_share, 2),
        'upside_pct': round(upside * 100, 1),
        'pv_fcfs_total': round(sum(pv_fcfs), 0),
        'pv_terminal': round(pv_terminal, 0),
        'terminal_pct_of_ev': round(pv_terminal / enterprise_value * 100, 1) if enterprise_value > 0 else 0,
        'implied_pe': round(implied_pe, 1),
        'implied_ev_rev': round(implied_ev_rev, 1),
        'annual_details': pd.DataFrame(annual_details),
        'pv_fcfs': pv_fcfs,
        'wacc_used': wacc,
    }

# ── Run all three scenarios ───────────────────────────────────────
results = {
    'bear': run_dcf('bear'),
    'base': run_dcf('base'),
    'bull': run_dcf('bull'),
}

# ── Scenario Summary Table ────────────────────────────────────────
summary_rows = []
for label, r in results.items():
    summary_rows.append({
        'Scenario': label.upper(),
        'EV ($M)': f"${r['ev_usdm']:,.0f}",
        'Equity ($M)': f"${r['equity_value_usdm']:,.0f}",
        'Fair Value / Share': f"${r['fair_value_per_share']:,.2f}",
        'Upside/Downside': f"{r['upside_pct']:+.1f}%",
        'Terminal % of EV': f"{r['terminal_pct_of_ev']:.0f}%",
        'Implied P/E': f"{r['implied_pe']:.1f}x",
    })
df_scenarios = pd.DataFrame(summary_rows)
print("═" * 90)
print(f"  {COMPANY_NAME} — DCF Scenario Summary  (WACC: {WACC:.1%})")
print(f"  Current Price: ${CURRENT_PRICE:,.2f}")
print("═" * 90)
print(df_scenarios.to_string(index=False))
```

---

## Section 6b — Sensitivity Heatmaps

```python
# ── Sensitivity: WACC vs Terminal Growth ──────────────────────────
wacc_range = np.arange(0.07, 0.14, 0.01)
tg_range   = np.arange(0.02, 0.045, 0.005)

matrix = []
for w in wacc_range:
    row_vals = []
    for tg in tg_range:
        res = run_dcf('base', wacc_override=w, tg_override=tg)
        row_vals.append(res['fair_value_per_share'])
    matrix.append(row_vals)

df_sens = pd.DataFrame(matrix,
    index=[f"{w:.0%}" for w in wacc_range],
    columns=[f"{tg:.1%}" for tg in tg_range])

fig = px.imshow(df_sens.values, text_auto='.0f',
                x=[f"{tg:.1%}" for tg in tg_range],
                y=[f"{w:.0%}" for w in wacc_range],
                color_continuous_scale='RdYlGn',
                title=f"{COMPANY_NAME} — Fair Value Sensitivity: WACC vs Terminal Growth",
                labels=dict(x="Terminal Growth", y="WACC", color="Fair Value ($)"))
fig.show()
```

---

## Section 7 — Monte Carlo Simulation

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: MONTE CARLO SIMULATION
# ═══════════════════════════════════════════════════════════════════

np.random.seed(42)

def monte_carlo_dcf(n_sims=MC_SIMULATIONS):
    """Run MC simulation varying growth, margins, WACC, and terminal growth."""
    fair_values = []
    sim_params = []
    
    base_growth = REVENUE_GROWTH_BASE
    base_margin = OPERATING_MARGIN_TERMINAL
    base_wacc = WACC
    
    for _ in range(n_sims):
        # Draw random parameters
        sim_growth = np.random.normal(base_growth, MC_REVENUE_STDEV)
        sim_growth = np.clip(sim_growth, -0.05, 0.60)  # bound growth
        
        sim_margin = np.random.normal(base_margin, MC_MARGIN_STDEV)
        sim_margin = np.clip(sim_margin, 0.05, 0.70)  # bound margin
        
        sim_wacc = np.random.normal(base_wacc, MC_WACC_STDEV)
        sim_wacc = np.clip(sim_wacc, 0.05, 0.18)  # bound WACC
        
        sim_tg = np.random.uniform(0.02, 0.04)  # terminal growth
        
        # Project FCFs with simulated params
        total_revenue = LATEST_REVENUE
        fcfs = []
        for yr in range(FORECAST_YEARS):
            fade_frac = yr / max(FORECAST_YEARS - 1, 1)
            growth = sim_growth * (1 - fade_frac) + sim_tg * fade_frac
            margin = sim_margin
            
            total_revenue *= (1 + growth)
            sbc = total_revenue * SBC_PCT_REVENUE
            ebit = total_revenue * margin - CORPORATE_GA_USD_M - sbc
            nopat = ebit * (1 - TAX_RATE)
            
            da = total_revenue * (LATEST_DA / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.05)
            capex = total_revenue * (LATEST_CAPEX / LATEST_REVENUE if LATEST_REVENUE > 0 else 0.10)
            fcf = nopat + da - capex
            fcfs.append(fcf)
        
        # Discount
        disc = [(1 + sim_wacc) ** -(i+1) for i in range(FORECAST_YEARS)]
        pv_fcfs = sum(np.array(fcfs) * np.array(disc))
        
        # Terminal
        if sim_wacc > sim_tg:
            tv = fcfs[-1] * (1 + sim_tg) / (sim_wacc - sim_tg)
            pv_tv = tv * disc[-1]
        else:
            pv_tv = 0
        
        ev = pv_fcfs + pv_tv
        eq = ev - NET_DEBT_USD_M
        diluted = SHARES_OUT_M * (1 + ANNUAL_DILUTION_PCT) ** FORECAST_YEARS
        fv = eq / diluted if diluted > 0 else 0
        
        if fv > 0 and fv < CURRENT_PRICE * 20:  # filter extreme outliers
            fair_values.append(fv)
            sim_params.append({'growth': sim_growth, 'margin': sim_margin, 
                              'wacc': sim_wacc, 'tg': sim_tg, 'fv': fv})
    
    return np.array(fair_values), pd.DataFrame(sim_params)

mc_values, mc_params = monte_carlo_dcf()

# ── Histogram ─────────────────────────────────────────────────────
fig = go.Figure()
fig.add_trace(go.Histogram(x=mc_values, nbinsx=100, name='Simulated Fair Values',
                            marker_color='#2196F3', opacity=0.7))
fig.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red", 
              annotation_text=f"Current: ${CURRENT_PRICE:,.0f}")
fig.add_vline(x=np.median(mc_values), line_dash="dash", line_color="green",
              annotation_text=f"Median: ${np.median(mc_values):,.0f}")

fig.update_layout(
    title=f"{COMPANY_NAME} — Monte Carlo Fair Value Distribution ({MC_SIMULATIONS:,} simulations)",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Frequency",
    template="plotly_white"
)
fig.show()

# ── Statistics ────────────────────────────────────────────────────
pct_upside = (mc_values > CURRENT_PRICE).mean() * 100
print(f"\n📊 Monte Carlo Results ({len(mc_values):,} valid simulations)")
print(f"   Probability price > current (${CURRENT_PRICE:,.0f}): {pct_upside:.1f}%")
print(f"   10th percentile:  ${np.percentile(mc_values, 10):,.0f}")
print(f"   25th percentile:  ${np.percentile(mc_values, 25):,.0f}")
print(f"   Median:           ${np.median(mc_values):,.0f}")
print(f"   75th percentile:  ${np.percentile(mc_values, 75):,.0f}")
print(f"   90th percentile:  ${np.percentile(mc_values, 90):,.0f}")
print(f"   Mean:             ${np.mean(mc_values):,.0f}")
print(f"   Std Dev:          ${np.std(mc_values):,.0f}")

# ── CDF Chart ─────────────────────────────────────────────────────
sorted_vals = np.sort(mc_values)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(x=sorted_vals, y=cdf, mode='lines', name='CDF',
                              line=dict(color='#2196F3', width=2)))
fig_cdf.add_vline(x=CURRENT_PRICE, line_dash="dash", line_color="red",
                  annotation_text=f"Current Price")
fig_cdf.update_layout(
    title=f"{COMPANY_NAME} — Cumulative Distribution of Fair Value",
    xaxis_title="Fair Value per Share ($)", yaxis_title="Cumulative Probability",
    template="plotly_white"
)
fig_cdf.show()
```

---

## Section 9 — Peer Comparison

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: PEER COMPARISON
# ═══════════════════════════════════════════════════════════════════

# Define peers (EDIT THESE based on research)
PEER_TICKERS = ['PEER1', 'PEER2', 'PEER3', 'PEER4', 'PEER5']  # FILL IN

if YF_AVAILABLE:
    peer_data = []
    for t in [TICKER] + PEER_TICKERS:
        try:
            p = yf.Ticker(t)
            pi = p.info
            p_inc = p.income_stmt
            p_cf  = p.cash_flow
            
            rev = p_inc.loc['Total Revenue'].dropna().iloc[0] / 1e9 if 'Total Revenue' in p_inc.index else 0
            oi  = p_inc.loc['Operating Income'].dropna().iloc[0] / 1e9 if 'Operating Income' in p_inc.index else 0
            ni  = p_inc.loc['Net Income'].dropna().iloc[0] / 1e9 if 'Net Income' in p_inc.index else 0
            
            ocf = p_cf.loc['Operating Cash Flow'].dropna().iloc[0] / 1e9 if 'Operating Cash Flow' in p_cf.index else 0
            capex = abs(p_cf.loc['Capital Expenditure'].dropna().iloc[0]) / 1e9 if 'Capital Expenditure' in p_cf.index else 0
            fcf = ocf - capex
            
            mcap = pi.get('marketCap', 0) / 1e9
            ev_val = pi.get('enterpriseValue', 0) / 1e9
            
            # Revenue growth (YoY)
            if 'Total Revenue' in p_inc.index and len(p_inc.loc['Total Revenue'].dropna()) >= 2:
                rev_vals = p_inc.loc['Total Revenue'].dropna()
                rev_growth = (rev_vals.iloc[0] / rev_vals.iloc[1] - 1) if rev_vals.iloc[1] != 0 else 0
            else:
                rev_growth = 0
            
            peer_data.append({
                'Ticker': t,
                'Mkt Cap ($B)': round(mcap, 1),
                'EV ($B)': round(ev_val, 1),
                'Revenue ($B)': round(rev, 1),
                'Rev Growth': f"{rev_growth:.0%}",
                'Op Margin': f"{oi/rev:.0%}" if rev > 0 else "N/A",
                'Net Margin': f"{ni/rev:.0%}" if rev > 0 else "N/A",
                'FCF ($B)': round(fcf, 1),
                'P/E': round(pi.get('trailingPE', 0), 1),
                'EV/EBITDA': round(pi.get('enterpriseToEbitda', 0), 1),
                'EV/Rev': round(ev_val / rev, 1) if rev > 0 else 0,
                'P/FCF': round(mcap / fcf, 1) if fcf > 0 else 0,
                '_rev_growth_num': rev_growth,
                '_ev_rev_num': ev_val / rev if rev > 0 else 0,
                '_mcap_num': mcap,
            })
        except Exception as e:
            print(f"⚠️ Could not fetch {t}: {e}")
    
    df_peers = pd.DataFrame(peer_data)
    print("═" * 100)
    print(f"  Peer Comparison")
    print("═" * 100)
    display_cols = [c for c in df_peers.columns if not c.startswith('_')]
    print(df_peers[display_cols].to_string(index=False))
    
    # ── Scatter: EV/Revenue vs Revenue Growth ─────────────────────
    fig = px.scatter(df_peers, x='_rev_growth_num', y='_ev_rev_num', 
                     size='_mcap_num', text='Ticker',
                     title="Peer Comparison: EV/Revenue vs Revenue Growth",
                     labels={'_rev_growth_num': 'Revenue Growth (YoY)', 
                             '_ev_rev_num': 'EV/Revenue',
                             '_mcap_num': 'Market Cap'})
    fig.update_traces(textposition='top center')
    fig.update_layout(template='plotly_white',
                      xaxis_tickformat='.0%')
    fig.show()
```

---

## Section 10 — FCF Waterfall

```python
# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: FCF BRIDGE WATERFALL
# ═══════════════════════════════════════════════════════════════════

base_details = results['base']['annual_details']
yr3 = base_details.iloc[2]  # Year 3 of projection

labels = ['Revenue', 'Segment OpEx', 'SBC', 'Corp G&A', 'EBIT', 'Tax', 'NOPAT', 'D&A', 'Capex', 'FCF']
values = [
    yr3['revenue'],
    -(yr3['revenue'] - yr3['op_income']),  # segment-level costs
    -yr3['sbc'],
    -CORPORATE_GA_USD_M,
    yr3['ebit'],
    -(yr3['ebit'] * TAX_RATE),
    yr3['nopat'],
    yr3['da'],
    -yr3['capex'],
    yr3['fcf']
]
measures = ['absolute', 'relative', 'relative', 'relative', 'total',
            'relative', 'total', 'relative', 'relative', 'total']

fig = go.Figure(go.Waterfall(
    name="FCF Bridge", orientation="v",
    measure=measures, x=labels, y=values,
    connector={"line": {"color": "rgb(63, 63, 63)"}},
    increasing={"marker": {"color": "#4CAF50"}},
    decreasing={"marker": {"color": "#F44336"}},
    totals={"marker": {"color": "#2196F3"}}
))
fig.update_layout(
    title=f"{COMPANY_NAME} — FCF Bridge (Base Case, Year 3 of Projection)",
    yaxis_title="USD Millions", template="plotly_white"
)
fig.show()
```

---

## Notebook Finalisation

```python
nb.cells = cells

with open(f"/home/claude/{TICKER.lower()}_valuation.ipynb", "w") as f:
    nbformat.write(nb, f)

print(f"✅ Notebook saved: {TICKER.lower()}_valuation.ipynb")
```
