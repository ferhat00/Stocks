# Tech Valuation Context Reference

Background on technology company valuation methodologies, common multiples by sub-sector,
SBC treatment, TAM frameworks, and sector-specific considerations.

## Table of Contents
1. Valuation Multiples by Sub-Sector
2. Share-Based Compensation (SBC) Treatment
3. TAM (Total Addressable Market) Framework
4. Segment Revenue Modelling
5. AI Infrastructure Cycle Context
6. Cloud Computing Market Context
7. Digital Advertising Market Context
8. Semiconductor Market Context
9. WACC Considerations for Tech
10. Common Pitfalls

---

## 1. Valuation Multiples by Sub-Sector

Typical ranges for large-cap tech (as of 2024-2025, varies with market conditions):

| Sub-Sector              | EV/Revenue | EV/EBITDA | P/E (fwd) | P/FCF | Revenue Growth |
|-------------------------|------------|-----------|-----------|-------|----------------|
| Mega-cap Platform (AAPL, MSFT, GOOGL) | 8-12x | 20-30x | 25-35x | 25-35x | 5-15% |
| Cloud Infrastructure (AMZN AWS, MSFT Azure) | 10-20x | 25-40x | 30-50x | 30-50x | 15-30% |
| High-Growth SaaS (SNOW, DDOG, NET) | 15-30x | 50-100x+ | 60-100x+ | 50-100x+ | 25-40% |
| Semiconductors - AI (NVDA, AVGO) | 15-30x | 30-50x | 30-60x | 35-60x | 20-50%+ |
| Semiconductors - Mature (INTC, TXN) | 3-6x | 12-20x | 15-25x | 15-25x | 0-10% |
| Digital Advertising (META, GOOGL ads) | 6-10x | 15-25x | 20-30x | 20-30x | 10-20% |
| E-Commerce (AMZN retail, SHOP) | 2-5x | 15-30x | 30-50x | 25-40x | 10-20% |
| Enterprise Software (ORCL, SAP, CRM) | 8-15x | 20-35x | 25-40x | 25-40x | 10-20% |
| Consumer Tech/Hardware (AAPL products) | 6-8x | 18-25x | 25-32x | 25-32x | 2-8% |

**PEG Ratio**: For growth companies, PEG (P/E ÷ EPS growth rate) < 1.0 = cheap, 1.0-2.0 = fair, > 2.0 = expensive. Less useful for very high-growth or loss-making companies.

**Rule of 40**: For SaaS — Revenue Growth % + FCF Margin % should exceed 40%. Above 60% = exceptional.

---

## 2. Share-Based Compensation (SBC) Treatment

SBC is a **real economic cost** for tech companies and must be treated carefully in valuation:

### Why SBC Matters
- Tech companies issue significant equity compensation (often 5-15% of revenue)
- This dilutes existing shareholders even if "non-cash"
- Must be deducted from FCF when calculating fair value
- Net of buybacks, look at **net dilution** (gross dilution minus buyback effect)

### SBC Benchmarks (% of Revenue)
| Company Tier | Typical SBC/Revenue |
|-------------|-------------------|
| FAANG/Mega-cap | 5-12% |
| Large SaaS | 15-25% |
| Mid-cap Growth | 20-35% |
| Early-stage | 30-50%+ |

### How to Handle in DCF
1. **Method 1 (Preferred)**: Treat SBC as operating expense, deduct from EBIT before tax
2. **Method 2**: Add back SBC to FCF but increase shares outstanding by dilution %
3. **Never**: Simply ignore SBC as "non-cash" — this overstates intrinsic value

### Net Dilution Calculation
```
Net annual dilution % = (New shares issued via SBC) / (Total shares at start)
                        - (Shares repurchased via buybacks) / (Total shares at start)
```

Typical mega-cap tech: 1-3% gross dilution, 0-1.5% net dilution after buybacks.

---

## 3. TAM (Total Addressable Market) Framework

### TAM Estimation Approaches
1. **Top-down**: Start with total market → apply relevant % → company's addressable slice
2. **Bottom-up**: Number of potential customers × average revenue per customer
3. **Value-theory**: How much value does the product create → willingness to pay

### TAM Credibility Rules
- TAM > $1T → almost certainly too broad, need to narrow
- TAM = Serviceable Addressable Market (SAM) = 20-50% of TAM
- Market share > 40% in a single segment → natural ceiling approaching
- TAM growth should slow as market matures (S-curve)

### Key TAM Estimates (2024-2030 approximate)

| Market | 2024 Est. | 2030 Est. | CAGR |
|--------|-----------|-----------|------|
| Global Cloud Infrastructure | $300B | $800B | 18% |
| AI Training + Inference Hardware | $100B | $500B | 30%+ |
| Digital Advertising (global) | $650B | $1.1T | 9% |
| Enterprise Software | $350B | $600B | 10% |
| Semiconductors (total) | $600B | $1T | 9% |
| E-Commerce (global) | $6.3T | $10T+ | 8% |
| Cybersecurity | $200B | $400B | 12% |

---

## 4. Segment Revenue Modelling

### Growth Rate Fading
Tech revenue growth naturally decelerates as companies scale. Model with a fade:

```
Year N growth = Initial_Growth × (1 - fade_fraction) + Terminal_Growth × fade_fraction
where fade_fraction = N / Forecast_Years
```

### Typical Growth Trajectories
- **Hypergrowth (>40%)**: Can sustain 2-3 years max for large-cap, then fade rapidly
- **High growth (20-40%)**: Sustainable for 3-5 years for $50B+ revenue companies
- **Moderate growth (10-20%)**: Sustainable for 5-10 years for mega-caps
- **Mature (<10%)**: Long-term sustainable, approaching GDP + inflation

### Margin Expansion Framework
Most tech companies see operating margin expansion as they scale (operating leverage):
- **Gross margin**: Relatively stable for software (70-80%), hardware (40-60%)
- **Operating margin**: Typically expands 100-200bps/year during growth phase
- **Terminal operating margin**: Software 30-45%, Hardware 20-35%, Advertising platform 35-50%

---

## 5. AI Infrastructure Cycle Context

### Key Metrics to Track
- **Hyperscaler capex**: MSFT + GOOGL + AMZN + META combined capex ($150B+ in 2024)
- **GPU shipments**: NVIDIA data center revenue as proxy
- **AI model training costs**: Doubling annually (GPT-3: $5M → GPT-4: $100M+)
- **Inference vs training split**: Shifting toward inference (60-70% of compute by 2027)
- **Power consumption**: Data center power demand (GW), nuclear/renewable buildout

### AI Revenue Monetisation Lag
- **Phase 1** (current): Infrastructure buildout (chips, servers, networking)
- **Phase 2** (emerging): Platform/tooling (cloud AI services, model APIs)
- **Phase 3** (developing): Application layer (enterprise AI, consumer AI products)
- Revenue recognition flows from Phase 1 → 3 with 12-24 month lags

### Companies by AI Value Chain Position
| Position | Companies | Revenue Timing |
|----------|-----------|---------------|
| Picks & Shovels | NVDA, AVGO, AMD, TSM, ASML | Immediate |
| Infrastructure | AMZN, MSFT, GOOGL (cloud) | Current + growing |
| Platform | MSFT (Copilot), GOOGL (Gemini), META (Llama) | Emerging |
| Application | CRM, ADBE, various startups | Future growth |

---

## 6. Cloud Computing Market Context

### Market Share (IaaS + PaaS, approximate 2024)
- AWS: 31% → stable/slight decline
- Microsoft Azure: 25% → gaining share
- Google Cloud: 11% → gaining share
- Others (Alibaba, Oracle, IBM): 33% → mixed

### Cloud Economics
- Gross margins: 60-70% (AWS, Azure, GCP)
- Operating margins: 25-35% at scale (AWS most mature)
- Customer retention: 95%+ net revenue retention for enterprise
- Growth: On-premise to cloud migration still <30% complete globally

---

## 7. Digital Advertising Market Context

### Market Share (approximate 2024)
- Google (Search + YouTube): 38%
- Meta (Facebook + Instagram): 22%
- Amazon Ads: 8%
- TikTok/ByteDance: 5%
- Microsoft/LinkedIn: 3%
- Others: 24%

### Key Metrics
- **ARPU** (Average Revenue Per User): Meta ~$45/user globally, $75+ US
- **Impression pricing**: CPM varies $5-50 by platform and format
- **Advertiser ROI**: ROAS > 3x typically justifies continued spend
- **Cyclicality**: Ad revenue correlates with GDP, but digital gaining share from traditional

---

## 8. Semiconductor Market Context

### Market Segments
- **Logic/Processing**: CPUs, GPUs, AI accelerators (NVDA, AMD, INTC)
- **Memory**: DRAM, NAND (Samsung, SK Hynix, Micron)
- **Analog/Mixed-signal**: TI, ADI, ON Semi
- **Foundry**: TSM, Samsung, Intel Foundry
- **Equipment**: ASML, AMAT, LRCX, KLA

### AI Chip Market
- NVIDIA dominates training (80%+ share) and growing in inference
- AMD gaining in inference with MI300 series
- Custom silicon emerging: Google (TPU), Amazon (Trainium/Inferentia), Meta
- China restrictions creating bifurcated market

### Semiconductor Cycle
- Historically 3-5 year cycles (inventory build → correction → recovery)
- AI may be extending current upcycle beyond normal duration
- Watch: memory pricing, foundry utilization rates, lead times

---

## 9. WACC Considerations for Tech

### Typical WACC Components for Mega-Cap Tech
| Component | Typical Range | Notes |
|-----------|--------------|-------|
| Risk-free rate | 3.5-5.0% | 10-year Treasury |
| Equity risk premium | 4.5-6.0% | Damodaran estimate |
| Beta | 0.8-1.8 | Higher for growth, lower for mature |
| Cost of equity | 8-15% | CAPM: Rf + β × ERP |
| Cost of debt | 3-6% | Most mega-caps are investment grade |
| Debt weight | 0-15% | Tech tends to be equity-heavy |
| Effective tax rate | 12-21% | International structures, GILTI |
| **WACC** | **8-13%** | Mega-cap: 8-10%, Growth: 10-13% |

### Special Considerations
- **Negative net debt**: Many tech companies have net cash → WACC should use market value weights
- **Beta instability**: AI/semiconductor betas elevated in 2023-2025; use 2-3 year rolling
- **Country risk**: Companies with China exposure may warrant 50-100bp premium
- **Size premium**: None needed for mega-caps; add 100-200bp for mid-cap tech

---

## 10. Common Pitfalls in Tech Valuation

1. **Ignoring SBC**: Overstates FCF by 5-15% of revenue
2. **Extrapolating hypergrowth**: 50%+ growth rates cannot sustain for >3 years at scale
3. **Terminal value dominance**: If TV > 70% of DCF, assumptions are doing too much work — stress test
4. **Single-segment analysis**: Companies like AMZN/GOOGL have vastly different segment profiles
5. **Ignoring capex cycles**: AI infrastructure capex is front-loaded; FCF will improve as cycle matures
6. **Currency blindness**: 40-60% of mega-cap revenue is international → USD strength impacts
7. **Regulatory risk discount**: Antitrust actions could force breakups, limit M&A, restrict practices
8. **Narrative-driven TAM**: TAM estimates are often inflated to justify high valuations
9. **Backward-looking multiples**: Using trailing P/E for growth company understates value
10. **Peer comparison traps**: Median peer multiple × company earnings assumes similar growth/risk profile
