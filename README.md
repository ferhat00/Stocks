# Quantamental Stock Analysis

A collection of LLM-inspired Jupyter notebooks for comprehensive stock analysis combining quantitative metrics with fundamental research.

## Overview

This repository contains data-driven equity analysis notebooks that blend quantitative analytics (quant) with fundamental analysis (fundamental) - a methodology known as "quantamental" investing. Each notebook provides institutional-grade research similar to what you'd find at Goldman Sachs or other bulge bracket investment banks.

## Features

- **Comprehensive Financial Metrics**: Market cap, revenue growth, profitability margins, and valuation multiples
- **Comparative Analysis**: Side-by-side comparison with industry peers and competitors
- **Technical Analysis**: Price momentum, moving averages, volatility analysis
- **Risk Metrics**: Beta, Sharpe ratio, maximum drawdown, annualized volatility
- **Valuation Frameworks**:
  - Trading comparables (EV/Sales, P/E, P/B ratios)
  - Sum-of-Parts (SOTP) valuation for multi-segment businesses
  - DCF-ready financial modeling
- **Professional Visualizations**: Publication-quality charts and heatmaps
- **Quantitative Scoring**: Investment recommendation framework with weighted factors
- **Data Export**: CSV exports for integration with portfolio management systems

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Required Packages

Install all dependencies with a single command:

```bash
pip install yfinance pandas numpy matplotlib seaborn
```

Or install individually:

```bash
pip install yfinance      # Financial data from Yahoo Finance
pip install pandas        # Data manipulation
pip install numpy         # Numerical computing
pip install matplotlib    # Visualization
pip install seaborn       # Statistical visualization
```

## Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Stocks.git
cd Stocks
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open any analysis notebook (e.g., `AIRO_Competitor_Analysis.ipynb`)

4. Run all cells to generate the complete analysis

## Notebooks

### AIRO Competitor Analysis
**File**: `AIRO_Competitor_Analysis.ipynb`

Comprehensive analysis of AIRO Group Holdings (NASDAQ: AIRO) against aerospace, defense, and eVTOL sector competitors.

**Analyzed Companies**:
- AIRO - AIRO Group Holdings (Diversified: Drones + eVTOL + Training)
- AVAV - AeroVironment (Military UAS)
- KTOS - Kratos Defense (Unmanned Systems)
- JOBY - Joby Aviation (Air Taxi)
- ACHR - Archer Aviation (Air Taxi)

**Analysis Sections**:
1. Data Collection & Company Universe
2. Key Financial Metrics Extraction
3. Valuation Analysis - Trading Comparables
4. Stock Price Performance Analysis
5. Sum-of-Parts (SOTP) Valuation
6. Risk-Adjusted Return Analysis
7. Comparative Valuation Dashboard
8. Investment Recommendation Framework
9. Export Results

**Output**:
- 7 comprehensive visualizations
- Quantitative investment scoring
- CSV exports for further analysis

## Methodology

This repository employs a **quantamental** approach that combines:

### Quantitative Components
- Statistical analysis of price movements
- Correlation and covariance analysis
- Risk-adjusted return calculations
- Factor-based scoring models
- Technical indicators and momentum analysis

### Fundamental Components
- Business segment analysis
- Competitive positioning
- Management guidance assessment
- Industry trend evaluation
- Sum-of-Parts valuation for conglomerates

### LLM Enhancement
- Automated data collection and processing
- Dynamic visualization generation
- Multi-factor scoring algorithms
- Peer group identification
- Comprehensive reporting

## Data Sources

- **Yahoo Finance API**: Historical prices, financial statements, analyst estimates
- **Company Filings**: Segment-level data, guidance, strategic initiatives
- **Market Data**: Real-time and historical OHLCV data

## Output Files

Each analysis notebook can export results to CSV:

- `*_Competitor_Comparison.csv` - Complete metrics comparison
- `*_Performance_Metrics.csv` - Price performance data
- `*_Risk_Analysis.csv` - Risk-adjusted returns
- `*_SOTP_Valuation.csv` - Segment valuation breakdown
- `*_Investment_Scores.csv` - Quantitative recommendations

## Customization

### Analyzing Different Stocks

Modify the `companies` dictionary in any notebook:

```python
companies = {
    'TICKER': {
        'name': 'Company Name',
        'category': 'Industry Category',
        'segment': 'Business Description'
    },
    # Add more companies...
}
```

### Adjusting Time Periods

Change the historical data window:

```python
start_date = end_date - timedelta(days=730)  # 2 years (default)
start_date = end_date - timedelta(days=365)  # 1 year
start_date = end_date - timedelta(days=1825) # 5 years
```

### Custom Valuation Multiples

Adjust sector-specific multiples in SOTP analysis:

```python
'Comparable Multiple (EV/Sales)': [4.5, 2.0, 1.5, 0.0]
```

## Disclaimer

**This analysis is for informational and educational purposes only and does not constitute investment advice.**

- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal
- Consult with a qualified financial advisor before making investment decisions
- The authors are not responsible for any financial losses incurred

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for enhancement:

- Additional valuation methodologies (DCF, DDM)
- Machine learning price prediction models
- Sentiment analysis from news/social media
- Options analysis and Greeks calculations
- Portfolio optimization algorithms
- ESG scoring integration

## License

This project is open source and available under the MIT License.

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue on GitHub.

---

**Built with**: Python, Jupyter, yfinance, pandas, matplotlib, seaborn

**Analysis Style**: Goldman Sachs-inspired equity research with quantitative scoring

**Last Updated**: January 2026
