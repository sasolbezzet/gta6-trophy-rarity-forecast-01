# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sasolbezzet/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sasolbezzet/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Hierarchical Constrained Ensemble (HCE) statistical engine for trophy earn rate forecasting.

## Methodology

1. **Dual-Trend Regression**: Logarithmic and linear time-series regressions combined with non-negative growth constraints.
2. **Bayesian Prior Calibration**: Variance and drift priors based on item tier and unlock categories.
3. **Monotonic Hierarchy Projection**: Sequential prerequisite ordering via Pool Adjacent Violators Algorithm (PAVA).
4. **Quantile Calibration**: Non-crossing multi-quantile sorting (`[0.05, 0.25, 0.50, 0.75, 0.95]`) optimized for pinball loss.

## Run it

```sh
python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=7kn3j477&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
