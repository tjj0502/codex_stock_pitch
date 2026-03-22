# Online Research

Checked on 2026-03-22.

## Table of Contents

- Technical and price-based evidence
- Fundamental and quality evidence
- Event and regime data sources
- Design implications
- Source list

## Technical and price-based evidence

### Momentum remains the strongest simple technical building block

- Kenneth French's emerging-market momentum construction uses lagged returns from `t-12` to `t-2` and includes China in the emerging-market set. Practical use: include medium-horizon momentum and skip the most recent month rather than using naive trailing returns.
- Jegadeesh and Titman (1993) documented medium-horizon persistence in winner-minus-loser portfolios. Practical use: support 3-month, 6-month, and 12-month relative-strength signals.
- Daniel and Moskowitz (2014) showed momentum crashes are concentrated after market declines and when volatility is high. Practical use: de-risk momentum during panic regimes instead of deleting it entirely.

### Cost discipline matters more for fast signals

- Novy-Marx and Velikov (2016) found many anomalies weaken materially after transaction costs, especially high-turnover designs. Practical use: keep short-term reversal or intramonth signals small until the strategy clears a cost-aware backtest.

## Fundamental and quality evidence

### Value, profitability, and investment should be treated as separate pillars

- Kenneth French's five-factor construction defines value (`HML`), profitability (`RMW`), and investment (`CMA`) as distinct return drivers. Practical use: do not collapse all fundamentals into one value ratio.
- Novy-Marx (2010/2013) found gross profits-to-assets predicts average returns with power comparable to book-to-market. Practical use: add profitability even if value is already present.

### Accounting-strength composites are useful inside cheap stocks

- Piotroski (2001) reported that within high book-to-market stocks, selecting financially strong firms increased the mean return by at least 7 percent annually, and a long-short strategy earned 23 percent annually over 1976-1996. Practical use: use an `F-score` style composite or similar quality screen instead of relying only on single ratios.

### Quality is broader than profitability alone

- AQR's `Quality Minus Junk` work describes quality as profitability, growth, safety, and payout, and reports persistent long-run quality premia across markets. Practical use: if the data are available, split quality into multiple subcomponents instead of using one accounting metric.

## Event and regime data sources

### Fed and monetary policy events

- The official Federal Reserve FOMC calendar page lists meeting dates, statements, minutes, and press-conference timing. As of 2026-03-22, the Fed's published 2026 meeting schedule is January 27-28, March 17-18, April 28-29, June 16-17, July 28-29, September 15-16, October 27-28, and December 8-9.
- Federal Reserve IFDP 844 describes a standard surprise measure: the target surprise is the change in the current-month fed funds futures rate in a 30-minute window around the announcement. Practical use: use surprise magnitude and sign, not only a meeting dummy.

### Geopolitical risk

- The official Geopolitical Risk Index site states that higher geopolitical risk foreshadows lower investment, stock prices, and employment. It provides monthly and daily data, including subindexes for threats and acts and country-specific series. The site reported monthly data last updated on 2026-03-01 and daily data last updated on 2026-03-09.
- The country-specific GPR site includes China, Russia, and Ukraine series. Practical use: combine a broad GPR measure with country-specific measures that match the market's external exposures.

### Policy and trade uncertainty

- Baker, Bloom, and Davis describe both monthly and daily Economic Policy Uncertainty data, and note that the daily U.S. index is updated at approximately 9 a.m. EST on the public site. Practical use: use EPU as a regime indicator and use policy-category variants when the macro question is specific.
- Matteo Iacoviello's Trade Policy Uncertainty site provides monthly and daily TPU data and explicitly highlights spikes around U.S.-China trade tensions. The site reported monthly data last updated on 2026-03-01 and daily data last updated on 2026-03-09. Practical use: TPU is especially relevant for exporter-heavy markets and sectors.

### Financial stress

- FRED's `STLFSI4` measures market stress from 18 weekly series and interprets zero as normal financial conditions. The table feed shows data through 2026-03-06 and a last update on 2026-03-11. Practical use: use stress as a regime switch or to shrink risk-on factor weights during fragile periods.

## Design implications

- Inference from the sources: the base score should usually come from slower-moving technical and fundamental signals, while event variables should enter as overlays or interactions with stock exposures.
- Inference from the sources: a market-level event series should not be added equally to every stock. Instead, interact it with sector, country, duration, leverage, commodity, or export exposure.
- Inference from the sources: a compact first model with value, profitability, investment discipline, medium-horizon momentum, and one stress overlay is a better starting point than a wide menu of loosely justified indicators.
- Use point-in-time inputs only. Rebuild the universe on rebalance dates, lag financial statements, and document missing-data rules.
- Evaluate with both raw and neutralized rankings so you can tell whether the model is stock selection or disguised style timing.

## Source list

- [Kenneth French - Description of Momentum Factor for Emerging Markets](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_emerging_mom.html)
- [Kenneth French - Description of Fama/French 5 Factors for Emerging Markets](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library/f-f_5emerging.html)
- [Piotroski - Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers](https://www.researchgate.net/publication/228316791_Value_Investing_The_Use_of_Historical_Financial_Statement_Information_to_Separate_Winners_from_Losers)
- [Novy-Marx - The Other Side of Value: Good Growth and the Gross Profitability Premium](https://www.nber.org/papers/w15940)
- [AQR - Quality Minus Junk data and paper links](https://www.aqr.com/Learning-Center/Systematic-Equities/Further-Reading)
- [Jegadeesh and Titman - Returns to Buying Winners and Selling Losers](https://www.deepdyve.com/lp/wiley/returns-to-buying-winners-and-selling-losers-implications-for-stock-953hyBps5z)
- [Daniel and Moskowitz - Momentum Crashes](https://www.nber.org/papers/w20439)
- [Novy-Marx and Velikov - A Taxonomy of Anomalies and Their Trading Costs](https://www.nber.org/papers/w20721)
- [Federal Reserve - FOMC meeting calendars and information](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)
- [Federal Reserve - 2025 and 2026 FOMC tentative schedule](https://www.federalreserve.gov/newsevents/pressreleases/monetary20240809a.htm)
- [Federal Reserve IFDP 844 - The Response of Global Equity Indexes to U.S. Monetary Policy Announcements](https://www.federalreserve.gov/pubs/ifdp/2005/844/ifdp844.htm)
- [Geopolitical Risk Index official site](https://www.matteoiacoviello.com/gpr.htm)
- [Country-Specific Geopolitical Risk Index](https://www.matteoiacoviello.com/gpr_country.htm)
- [Baker, Bloom, and Davis - Measuring Economic Policy Uncertainty](https://www.policyuncertainty.com/media/EPU_BBD_Mar2016.pdf)
- [Trade Policy Uncertainty official site](https://www.matteoiacoviello.com/tpu.htm)
- [FRED - St. Louis Fed Financial Stress Index (STLFSI4)](https://fred.stlouisfed.org/series/STLFSI4)
