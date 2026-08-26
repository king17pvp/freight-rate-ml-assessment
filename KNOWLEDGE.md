# KNOWLEDGE.md — Time Series Forecasting: Tips, Tricks & Theory

A reference guide compiled for the freight-rate prediction project. Covers data problems, splitting strategies, model choices, the underlying math, and practitioner wisdom from forecasting competitions and industry.

---

## 0. Core Concepts: Lag & Stationarity — and How They Drive Every Decision

These two concepts are the diagnostic foundation of the whole field: you measure them *first*, and their answers tell you how to clean, transform, feature-engineer, validate, and model.

### 0.1 What is a lag?

A **lag** is simply the value of the series $k$ steps in the past. Formally, using the **backshift (lag) operator** $B$:

$$B^k y_t = y_{t-k}$$

So "lag 1" of today's freight rate is yesterday's rate; "lag 7" of a daily series is the same weekday last week. Lags matter because autocorrelation — the defining property of time series (§1.1) — is *correlation between the series and its own lags*.

**Autocorrelation function (ACF)** at lag $k$ measures the *total* linear relationship between $y_t$ and $y_{t-k}$:

$$\rho_k = \frac{\mathrm{Cov}(y_t,\, y_{t-k})}{\mathrm{Var}(y_t)} = \frac{\sum_{t=k+1}^{T}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T}(y_t - \bar{y})^2}$$

**Partial autocorrelation function (PACF)** at lag $k$ measures the *direct* relationship between $y_t$ and $y_{t-k}$ **after removing** the influence of the intermediate lags $1, \dots, k-1$ (it's the coefficient $\phi_{kk}$ on $y_{t-k}$ in a regression of $y_t$ on all lags up to $k$). The distinction matters: in a strongly trending series, lag 30 has a huge ACF simply because it correlates through lags 1–29 — PACF tells you whether it adds anything *new*.

**Reading the ACF/PACF plots (the classic diagnostic):**

| Pattern | Interpretation | What to do |
|---|---|---|
| ACF decays very slowly / near-linearly | Non-stationary (trend or unit root) | Difference the series before anything else |
| ACF spikes at lags $s, 2s, 3s$ (e.g., 7, 14, 21) | Seasonality with period $s$ | Seasonal differencing, seasonal dummies, or lag-$s$ features |
| ACF cuts off after lag $q$, PACF decays gradually | MA($q$) signature | ARIMA with `q` MA terms |
| PACF cuts off after lag $p$, ACF decays gradually | AR($p$) signature | ARIMA with `p` AR terms; use lags $1..p$ as ML features |
| No significant spikes anywhere (within $\pm 1.96/\sqrt{T}$ bands) | White noise — no learnable structure | Stop: nothing forecasts white noise; if these are your *residuals*, your model is done |

For ML models, the PACF is a principled **lag-feature selector**: lags with significant partial autocorrelation are the ones worth engineering into the feature matrix of §4.1 ([statsmodels stationarity notebook](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html), [TDS on ACF/PACF diagnostics](https://towardsdatascience.com/when-a-time-series-only-quacks-like-a-duck-10de9e165e/)).

### 0.2 What is stationarity?

Intuitively: a series is stationary when *its statistical behavior doesn't depend on when you look at it* — same average level, same amount of wobble, same relationship between neighboring points ([TDS](https://towardsdatascience.com/when-a-time-series-only-quacks-like-a-duck-10de9e165e/)).

**Strict stationarity:** the joint distribution of $(y_{t_1}, \dots, y_{t_n})$ equals that of $(y_{t_1+\tau}, \dots, y_{t_n+\tau})$ for every shift $\tau$ — too strong to verify in practice.

**Weak (covariance) stationarity** — the working definition:

$$\mathbb{E}[y_t] = \mu \;\; \forall t, \qquad \mathrm{Var}(y_t) = \sigma^2 < \infty \;\; \forall t, \qquad \mathrm{Cov}(y_t, y_{t-k}) = \gamma_k \;\; \text{(depends only on } k\text{)}$$

A series with a trend violates the constant mean; volatility clustering violates constant variance; seasonality violates both.

**The canonical non-stationary case — the random walk (unit root):**

$$y_t = y_{t-1} + \varepsilon_t \quad \Rightarrow \quad \mathrm{Var}(y_t) = t\sigma^2 \to \infty$$

Its variance grows without bound, and shocks are *permanent* (a jump never decays). Freight rates, FX, and commodity prices behave close to this. Crucially, its first difference $\Delta y_t = \varepsilon_t$ is perfectly stationary — which is why differencing works.

**Two kinds of non-stationarity, two different cures:**
- **Difference-stationary** (unit root, stochastic trend): becomes stationary after differencing $\Delta y_t = y_t - y_{t-1}$. Detrending does *not* fix it.
- **Trend-stationary** (deterministic trend): $y_t = \alpha + \beta t + \varepsilon_t$ becomes stationary after *removing the fitted trend*. Differencing "works" but over-differences, injecting an MA(1) artifact into the errors.

**Testing — use ADF and KPSS together** (they have *opposite* null hypotheses; the most common mistake is reading "p < 0.05 ⇒ stationary" for both — that's correct for ADF and exactly backwards for KPSS):

- **ADF (Augmented Dickey–Fuller):** $H_0$ = unit root (non-stationary). Small p ⇒ reject ⇒ **stationary**.
- **KPSS:** $H_0$ = stationary. Small p ⇒ reject ⇒ **non-stationary**.

| ADF says | KPSS says | Conclusion | Action |
|---|---|---|---|
| Stationary | Stationary | Stationary | Model directly |
| Non-stationary | Non-stationary | Unit root (difference-stationary) | **Difference**, re-test |
| Non-stationary | Stationary | Trend-stationary | **Detrend** (fit & remove trend), don't difference |
| Stationary | Non-stationary | Heteroskedasticity / structural break suspected | Inspect; consider transform, regime split |

([statsmodels ADF/KPSS guide](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html), [Analytics Vidhya stationarity tests](https://www.analyticsvidhya.com/blog/2021/06/statistical-tests-to-check-stationarity-in-time-series-part-1/), [FPP3 ch. 9.1](https://otexts.com/fpp3/stationarity.html))

### 0.3 How these properties dictate your approach

This is the payoff — lag structure and stationarity are not trivia, they are the **routing logic** for the entire pipeline:

**→ Data cleaning (§1).** Outlier detection assumes a stable distribution, which a non-stationary series doesn't have — that's exactly why global IQR/z-score mislabels the 2021 rate regime as "outliers." So you detect outliers on a *stationarized* representation (STL residuals, or the differenced series), never on raw levels. Likewise, imputation method choice depends on autocorrelation strength: high lag-1 autocorrelation ⇒ forward-fill/interpolation is nearly lossless; weak autocorrelation ⇒ interpolation invents structure that isn't there.

**→ Transformation.** The ADF×KPSS table above literally decides *which* transform: difference vs. detrend vs. log/Box–Cox (for variance non-stationarity). Then re-run the tests on the transformed series until it passes — the number of differences needed is ARIMA's $d$.

**→ Feature engineering (§4.1, §5.2).** Significant PACF lags = your lag-feature shortlist. ACF spikes at lag $s$ = your seasonal period, telling you which seasonal lags ($y_{t-s}$), rolling-window lengths, and calendar features to build. If the (differenced) series shows no significant lags, no amount of lag engineering will help — go find exogenous drivers instead (bunker fuel, port congestion).

**→ Model choice (§3).** ARIMA *requires* stationarizing (that's the "I" in the name). Linear regression on lags is only valid inference-wise on stationary data (regressing one random walk on another produces spurious correlation — the classic Granger–Newbold result). Tree models are distribution-learners: a unit-root series drifts outside the training range, hitting their can't-extrapolate wall — which is precisely why §5.2 says predict $\Delta y_t$ or log-returns instead of levels. Strong persistence (ACF near 1 at lag 1) also explains why the naive forecast is so hard to beat: for a pure random walk, naive is *optimal*.

**→ Validation strategy (§2).** Stationary process ⇒ expanding window is best (all history is relevant). Non-stationary with drift/regime changes ⇒ sliding window (old regimes mislead). Long significant lag structure ⇒ the train/validation gap $g$ must be at least that long (§2.4), or your folds leak.

**→ Multi-step strategy (§4.1).** Near-unit-root persistence means recursive forecasting compounds errors fastest exactly when the series is least mean-reverting — another argument for direct models at long horizons on rate-like data.

**Quick diagnostic ritual for a new series (do this before any modeling):** plot the series → plot ACF/PACF → run ADF + KPSS → apply the indicated transform → re-plot ACF/PACF on the transformed series → *now* read off seasonality, lag features, and model class from the patterns above.

---

## 1. Data Problems in Time Series (vs. Traditional Tabular Data)

### 1.1 Why time series is fundamentally different

In classic tabular ML we assume rows are **i.i.d.** (independent and identically distributed): shuffling rows changes nothing. In time series, both assumptions break:

| Property | Tabular data | Time series |
|---|---|---|
| Row independence | Assumed i.i.d. | **Autocorrelated** — $y_t$ depends on $y_{t-1}, y_{t-2}, \dots$ |
| Distribution | Stationary by assumption | **Non-stationary** — mean/variance drift over time (trend, regime shifts) |
| Row order | Irrelevant, shuffle freely | **Order IS the signal** — shuffling destroys the data |
| Validation | Random K-fold works | Random K-fold causes **look-ahead leakage** (training on the future) |
| Outlier definition | Point far from the global distribution | Point far from its **local temporal context** (a value can be globally normal but locally anomalous) |

### 1.2 Common problems

**a) Non-stationarity.** A series is (weakly) stationary if $\mathbb{E}[y_t] = \mu$, $\mathrm{Var}(y_t) = \sigma^2$, and $\mathrm{Cov}(y_t, y_{t-k})$ depend only on lag $k$, not on $t$. Real series (freight rates especially) have trend, seasonality, and volatility clustering — all violations.
*Fix:* differencing ($y'_t = y_t - y_{t-1}$), seasonal differencing ($y'_t = y_t - y_{t-s}$), log/Box–Cox transforms for variance stabilization, or explicit trend/seasonal features. Test with ADF (null: unit root/non-stationary) or KPSS (null: stationary).

**b) Outliers.** In the time-series literature (Tsay 1988; Chen & Liu 1993) outliers are typed by *how* they interact with the temporal structure:

- **Additive outlier (AO):** a single-point spike; $y_t = x_t + \omega \cdot I(t = t_0)$ — affects only one observation (e.g., a data-entry error in one week's rate).
- **Innovational outlier (IO):** a shock that enters the *innovation* (error) term and propagates through the model dynamics — its effect decays over subsequent points.
- **Level shift (LS):** a permanent step change in the mean from $t_0$ onward (e.g., COVID-era container rate explosion, a new canal toll regime).
- **Transient change (TC):** a level shift that decays back exponentially.

**Key distinction from tabular data:** an observation is anomalous *relative to its neighborhood*, not the global distribution. A freight rate of \$10,000/FEU is an outlier in 2019 and completely normal in late 2021. Global IQR/z-score on raw values will mislabel entire regimes as outliers.

**c) Noise types.**
- **White noise:** $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$, uncorrelated — irreducible; if residuals are white noise, your model is done.
- **Measurement noise:** sensor/reporting error layered on the true signal.
- **Heteroskedastic noise:** variance changes over time (volatility clustering — very common in freight/financial rates; motivates GARCH-family models or log transforms).
- **Structural breaks / change points:** the data-generating process itself changes — "a change in the distribution," distinct from outliers ([Forecasting: Principles and Practice](https://otexts.com/fpp2/missing-outliers.html)).

**d) Missing data & irregular sampling.** Unlike tabular data, you cannot just drop rows (it breaks lag structure) and gaps can be *sequential* (whole stretches missing). Missingness may be informative (e.g., no rate quoted = no trade on that lane).

**e) Data leakage (time-series-specific).** Features computed with future information: centered rolling means, target encoding fit on the full series, scalers fit on the full series before splitting, or using same-day exogenous data that wouldn't be known at prediction time. This is the #1 silent killer of time-series projects.

### 1.3 How these problems are resolved

| Problem | Standard resolutions |
|---|---|
| Outlier **detection** | (1) **Decompose first, then test residuals**: run STL (Seasonal-Trend decomposition via Loess), apply IQR/z-score to the remainder component so trend/seasonality don't trigger false positives. (2) **Hampel filter**: rolling median ± $k \cdot \mathrm{MAD}$ (median absolute deviation) — robust and local. (3) Model-based: flag points where forecast error exceeds a threshold. (4) ML-based: Isolation Forest, Local Outlier Factor on windowed features — chosen for robustness to non-normal distributions ([ResearchGate study](https://www.researchgate.net/publication/391552163_Improving_Time_Series_Data_Quality_Identifying_Outliers_and_Handling_Missing_Values_in_a_Multilocation_Gas_and_Weather_Dataset)) |
| Outlier **treatment** | **Winsorization/capping** (clip to percentile bounds — keeps the observation, limits its leverage); replacement by interpolation/rolling median; **or deliberately keep it** — "simply replacing outliers without thinking about why they occurred is a dangerous practice; they may provide useful information about the process" ([Hyndman & Athanasopoulos, FPP](https://otexts.com/fpp2/missing-outliers.html)). A level shift is *not* an outlier to remove — it's a regime your model must learn. Adding a dummy/indicator feature for known events (strikes, COVID, Suez blockage) is often better than deleting. |
| Missing values | Forward-fill (safe: uses only the past — no leakage); linear/spline/PCHIP interpolation for short gaps; seasonal-decomposition-based imputation for seasonal series; model-based imputation (Kalman smoothing via state-space models) for longer gaps ([Towards Data Science](https://towardsdatascience.com/handling-gaps-in-time-series-dc47ae883990/)). **Beware:** interpolation uses future points → only apply within the training window or accept the (usually mild) leakage consciously. |
| Non-stationarity | Differencing, log/Box–Cox transform, detrending, or feed trend as a feature; for tree models, predict the *difference* or *ratio* rather than the level (see §5). |
| Structural breaks | Change-point detection (PELT, Bayesian online change-point detection, `ruptures` library); regime dummy variables; sliding-window training that forgets the old regime. |
| Leakage | Strictly causal features (rolling windows aligned to past only, `shift(1)` before rolling); fit all scalers/encoders on train fold only; temporal splits (see §2). |

---

## 2. Train / Dev / Test Splitting for Time Series

**Golden rule:** the model must never see the future. Every split must satisfy $\max(t_{\text{train}}) < \min(t_{\text{test}})$.

Let the series be $\{(x_t, y_t)\}_{t=1}^{T}$, forecast horizon $h$.

### 2.1 Simple temporal holdout (fixed-origin)

$$\mathcal{D}_{\text{train}} = \{t : 1 \le t \le T_1\},\quad \mathcal{D}_{\text{dev}} = \{t : T_1 < t \le T_2\},\quad \mathcal{D}_{\text{test}} = \{t : T_2 < t \le T\}$$

with typical proportions 70/15/15 or 80/10/10, chosen so dev and test each cover ≥ a few forecast horizons (and ideally ≥ 1 full seasonal cycle).

- ✅ **Pros:** dead simple; one training run; exactly mimics deployment ("train on everything up to today, predict tomorrow").
- ❌ **Cons:** a *single* evaluation window — the score has high variance and depends on whether the test window happened to be calm or chaotic (a test set landing on the 2021 rate spike tells a very different story than one landing on 2019). No estimate of score uncertainty.
- **Use when:** data is short, compute is limited, or as the final untouchable test set on top of one of the CV schemes below.

### 2.2 Expanding-window CV (rolling-origin, "walk-forward")

For folds $k = 1, \dots, K$ with initial size $n_0$ and step $s$:

$$\mathcal{T}_k^{\text{train}} = \{1, \dots, n_0 + (k-1)s\}, \qquad \mathcal{T}_k^{\text{val}} = \{n_0 + (k-1)s + 1, \dots, n_0 + (k-1)s + h\}$$

The train set start is fixed; the origin advances; old data is never dropped. This is what `sklearn.model_selection.TimeSeriesSplit` implements.

- ✅ **Pros:** uses maximum data per fold; multiple evaluation windows → mean ± std of the metric; matches deployment where you retrain on all history. "Usually more accurate for stationary processes" ([walk-forward CV guide](https://metricgate.com/docs/time-series-walk-forward-cv/)).
- ❌ **Cons:** later folds train on more data than earlier folds (folds aren't identically configured); retraining $K$ times is expensive; old, stale regimes stay in the training set forever.
- **Use when:** default choice for most forecasting problems; the process is reasonably stable and more history genuinely helps.

### 2.3 Sliding/rolling-window CV (fixed window size $w$)

$$\mathcal{T}_k^{\text{train}} = \{(k-1)s + 1, \dots, (k-1)s + w\}, \qquad \mathcal{T}_k^{\text{val}} = \{(k-1)s + w + 1, \dots, (k-1)s + w + h\}$$

Both endpoints slide; training size is constant.

- ✅ **Pros:** every fold has identical training size (comparable folds); automatically forgets stale regimes — "preferable when the data-generating process drifts, since old observations become irrelevant or actively misleading" ([expanding vs rolling](https://insightful-data-lab.com/2025/08/24/expanding-window-cross-validation/)); cheaper per fold.
- ❌ **Cons:** discards data; choosing $w$ is another hyperparameter; small $w$ → high-variance models.
- **Use when:** clear regime changes / concept drift — highly relevant for freight rates, where pre-2020 dynamics may actively hurt post-2020 prediction.

### 2.4 Blocked CV with gap (embargo / purged CV)

Partition into contiguous blocks and insert a **gap of $g$ points** between train and validation:

$$\mathcal{T}_k^{\text{val}} = \{a_k, \dots, b_k\}, \qquad \mathcal{T}_k^{\text{train}} \subseteq \{1, \dots, a_k - g - 1\}$$

The gap $g$ ≥ the maximum lag/rolling-window length used in features, so no training sample's features overlap the validation window. (In finance this is "purging and embargoing" — López de Prado, *Advances in Financial Machine Learning*.)

- ✅ **Pros:** eliminates the subtle leakage where a training row's lag-30 feature contains validation-period values; most honest error estimate for autocorrelated data ([blocked CV overview](https://www.numberanalytics.com/blog/ultimate-guide-time-series-cv)).
- ❌ **Cons:** loses $g$ points per fold; more bookkeeping.
- **Use when:** you use long lag/rolling features (you almost certainly do) and want an honest score; standard in quant finance.

### 2.5 Practical recipe

1. **Carve off the final test block first** (e.g., last 10–15% or the assessment's specified horizon) and never touch it.
2. On the remainder, run **expanding-window CV** (or sliding-window if regimes shift) with a **gap ≥ max feature lag** for model selection and hyperparameter tuning.
3. Report test performance **once**, from a model retrained on train+dev.
4. Never use random K-fold, and never shuffle. ([Analytics Vidhya overview](https://www.analyticsvidhya.com/blog/2026/03/time-series-cross-validation/))

---

## 3. Models for Time Series — Pros & Cons

**Always start with baselines.** If a model can't beat these, it's worthless:
- **Naive:** $\hat{y}_{t+h} = y_t$ (surprisingly hard to beat for random-walk-like series such as freight rates and FX).
- **Seasonal naive:** $\hat{y}_{t+h} = y_{t+h-s}$.
- **Drift:** $\hat{y}_{t+h} = y_t + h \cdot \frac{y_t - y_1}{t - 1}$.

### 3.1 Statistical models

| Model | Pros | Cons |
|---|---|---|
| **ARIMA / SARIMA** | Principled handling of autocorrelation; interpretable; confidence intervals for free; excellent on short, univariate, stationary-after-differencing series | Assumes linear dynamics; manual order selection (or auto_arima); struggles with multiple seasonalities, exogenous shocks, and many related series |
| **ETS (exponential smoothing, Holt-Winters)** | Robust, fast, tiny data requirements; great baseline; M-competition workhorse | Univariate only; linear; no exogenous variables in basic form |
| **Prophet** | Handles holidays/changepoints/missing data out of the box; easy to use; has been applied directly to container freight rates ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0967070X23000185)) | Often **underperforms** both simple statistical models and GBMs in benchmarks; curve-fitting approach ignores autocorrelation of errors |

### 3.2 Machine learning (tabularized) models

| Model | Pros | Cons |
|---|---|---|
| **Linear/Ridge/Lasso regression on lags** | Fast, interpretable, **can extrapolate trends** (unlike trees); strong with well-engineered features; Lasso does feature selection | Linear only; sensitive to outliers (use Huber loss); needs manual interaction features |
| **Random Forest** | Robust to outliers/noise (bagging averages them out); little tuning needed; hard to overfit badly; feature importances | **Cannot extrapolate beyond the training target range** (predictions are averages of training leaf values); usually beaten by boosting on accuracy; large memory footprint |
| **XGBoost / LightGBM / CatBoost** | **The dominant competition winner** — used by almost all top-50 M5 teams ([M5 results paper](https://www.sciencedirect.com/science/article/pii/S0169207021001874)); captures nonlinearities & interactions automatically; handles mixed feature types; native missing-value handling; fast (LightGBM especially); early stopping; quantile loss for prediction intervals | **Cannot extrapolate** — a tree's output is bounded by training targets, so a monotonically rising series gets forecast flat (fix: predict differences/ratios, or detrend first); needs careful CV to avoid overfitting; recursive multi-step degrades ([error decomposition paper](https://arxiv.org/pdf/2511.11461)) |

*RF vs. XGBoost intuition:* RF = **bagging** (parallel independent trees on bootstrap samples, variance reduction); XGBoost = **boosting** (sequential trees, each fitting the previous ensemble's residual gradient, bias reduction). Boosting is more accurate when tuned; RF is more forgiving when not.

### 3.3 Deep learning

| Model | Pros | Cons |
|---|---|---|
| **LSTM / GRU** | Learns temporal dependencies natively, no manual lag engineering; handles long sequences; multivariate naturally | Data-hungry; slow to train; sensitive to scaling & hyperparameters; frequently loses to GBMs on small/medium tabular-ish problems |
| **N-BEATS / N-HiTS** | Pure deep-learning architectures that beat statistical ensembles in M4; interpretable trend/seasonality stacks | Univariate focus (base version); heavy compute |
| **TFT (Temporal Fusion Transformer)** | State-of-the-art multivariate with static + known-future + observed covariates; attention gives interpretability | Complex, many hyperparameters, needs lots of data |
| **Foundation models (TimeGPT, Chronos, Moirai)** | Zero-shot forecasting; good when you have almost no history | Young field; can lose to a tuned LightGBM on domain-specific data; API/cost dependencies |

**Rule of thumb (Nixtla and others):** statistical ensembles first → boosted trees if you have tabular covariates → deep learning only if you have lots of data and the 3–10% extra accuracy justifies the cost ([Nixtla comparison](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/comparing_methods.html), [stats vs ML vs DL](https://blog.reachsumit.com/posts/2022/12/stats-vs-ml-for-ts/)).

For **freight rates specifically**, the literature uses ARIMA/VAR/VECM classically, and RF/XGBoost/MLP/LSTM/RBF-NN in recent work; hybrid and specialized models often beat standalone ones ([systematic review, Maritime Economics & Logistics](https://link.springer.com/article/10.1057/s41278-025-00334-3), [comparative evaluation](https://www.sciencedirect.com/science/article/pii/S209252122500015X)).

---

## 4. The Math: How Data Flows Into Each Model Type

### 4.1 Tabularization — the key trick for ML models

Tree/linear models expect a fixed-size matrix $X \in \mathbb{R}^{n \times d}$, not a sequence. You convert the series with a **sliding window**:

**Step 1 — build the supervised table.** For each time $t$, one row:

$$x_t = \big[\underbrace{y_{t-1}, y_{t-2}, \dots, y_{t-p}}_{\text{lag features}},\ \underbrace{\bar{y}_{t-1:t-w}, \sigma_{t-1:t-w}, \min, \max}_{\text{rolling stats (past-only!)}},\ \underbrace{\text{dow}_t, \text{month}_t, \dots}_{\text{calendar}},\ \underbrace{z^{(1)}_{t-1}, \dots}_{\text{exogenous lags}}\big], \qquad \text{target } y_t$$

A series of length $T$ with max lag $p$ yields a matrix of shape $(T - p,\ d)$. **Every feature must use only information available strictly before $t$** (in pandas: `shift(1)` before any rolling aggregation).

**Step 2 — choose a multi-step strategy** for horizon $h > 1$:

- **Recursive:** train one 1-step model $\hat{f}$; predict $\hat{y}_{t+1} = \hat{f}(x_t)$, then plug $\hat{y}_{t+1}$ back into the features to predict $\hat{y}_{t+2}$, etc. One model, but **errors compound** with horizon ([recursive vs direct](https://metricgate.com/blogs/recursive-vs-direct-multistep-forecasting/)).
- **Direct:** train $h$ separate models, $\hat{f}_j$ predicting $y_{t+j}$ from $x_t$ directly. No error feedback; more compute; "typically wins for nonlinear models at long horizons or under structural breaks" ([Let's Data Science](https://letsdatascience.com/blog/multi-step-time-series-forecasting-recursive-direct-and-hybrid-strategies)).
- **Multi-output:** one model emits the whole vector $(\hat{y}_{t+1}, \dots, \hat{y}_{t+h})$ (natural for neural nets).

**Step 3 — evaluate** with temporal splits (§2) using MAE, RMSE, MAPE/sMAPE, or scale-free **MASE** $= \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}$ (< 1 means you beat naive).

### 4.2 ARIMA$(p, d, q)$

Difference the series $d$ times: $y'_t = (1 - B)^d y_t$ where $B$ is the backshift operator ($By_t = y_{t-1}$). Then model:

$$y'_t = c + \underbrace{\phi_1 y'_{t-1} + \dots + \phi_p y'_{t-p}}_{\text{AR}(p)\text{: regression on own lags}} + \underbrace{\theta_1 \varepsilon_{t-1} + \dots + \theta_q \varepsilon_{t-q}}_{\text{MA}(q)\text{: regression on past errors}} + \varepsilon_t$$

**Flow:** raw 1-D series in → check stationarity (ADF) → pick $d$ → pick $p, q$ from ACF/PACF or minimize AIC → fit $\phi, \theta$ by maximum likelihood → forecast recursively, un-difference to return to the original scale. Input shape: just the vector $(y_1, \dots, y_T)$; no feature matrix.

### 4.3 Random Forest

Given the tabularized $(X, y)$, for $b = 1, \dots, B$: draw a bootstrap sample, grow a deep tree where each split chooses among a random subset of $m \approx d/3$ features the split minimizing MSE:

$$\min_{j,\, s}\ \sum_{i \in \text{left}} (y_i - \bar{y}_{\text{left}})^2 + \sum_{i \in \text{right}} (y_i - \bar{y}_{\text{right}})^2$$

Prediction is the average: $\hat{y} = \frac{1}{B}\sum_b T_b(x)$. Averaging decorrelated trees cuts variance — but the output is always an average of *training* target values, hence **no extrapolation**.

### 4.4 Gradient boosting (XGBoost / LightGBM)

Additive model built stagewise: $\hat{y}_i^{(m)} = \hat{y}_i^{(m-1)} + \eta\, f_m(x_i)$. XGBoost fits $f_m$ to the second-order Taylor expansion of the regularized objective:

$$\mathcal{L}^{(m)} = \sum_{i=1}^{n} \Big[ g_i f_m(x_i) + \tfrac{1}{2} h_i f_m(x_i)^2 \Big] + \gamma \tau + \tfrac{1}{2}\lambda \sum_{j=1}^{\tau} w_j^2$$

where $g_i = \partial_{\hat{y}} \ell(y_i, \hat{y}^{(m-1)})$, $h_i = \partial^2_{\hat{y}} \ell$, $\tau$ = number of leaves, $w_j$ = leaf weights. The optimal leaf weight and split gain have closed forms:

$$w_j^* = -\frac{\sum_{i \in \text{leaf } j} g_i}{\sum_{i \in \text{leaf } j} h_i + \lambda}, \qquad \text{Gain} = \tfrac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda}\right] - \gamma$$

**Flow:** tabularized $(X, y)$ of shape $(n, d)$ in → initialize $\hat{y}^{(0)} = \bar{y}$ → repeat: compute gradients per sample → grow a tree greedily by split gain (LightGBM: leaf-wise growth + histogram binning, which is why it's fast) → add scaled tree → early-stop when the *temporal* validation fold stops improving. Prediction: sum of leaf values across all trees.

### 4.5 LSTM (sequence models)

Input is 3-D: $(\text{batch},\ \text{sequence length } L,\ \text{features } d)$ — built by slicing windows $[x_{t-L+1}, \dots, x_t] \mapsto y_{t+1..t+h}$, with features **scaled** (fit the scaler on train only). Each cell updates:

$$\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}; x_t] + b_f) &\text{(forget gate)}\\
i_t &= \sigma(W_i [h_{t-1}; x_t] + b_i) &\text{(input gate)}\\
\tilde{c}_t &= \tanh(W_c [h_{t-1}; x_t] + b_c) &\text{(candidate state)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t &\text{(cell state)}\\
o_t &= \sigma(W_o [h_{t-1}; x_t] + b_o), \quad h_t = o_t \odot \tanh(c_t) &\text{(output)}
\end{aligned}$$

The final hidden state $h_L$ feeds a dense head producing the forecast vector. Trained by backpropagation through time with MSE/MAE loss.

### 4.6 Shape cheat-sheet

| Model | Input for training | Input at prediction |
|---|---|---|
| ARIMA/ETS | 1-D vector $(T,)$ | model state; forecasts recursively |
| Linear/RF/XGBoost | matrix $(T - p,\ d)$ from sliding window | one row $(1, d)$ per step (recursive) or per horizon model (direct) |
| LSTM/Transformer | tensor $(n_{\text{windows}},\ L,\ d)$ | latest window $(1, L, d)$ |

---

## 5. Practitioner Wisdom: What Actually Wins, and How to Tweak

### 5.1 What the best in the business converge on

The strongest empirical evidence comes from the M-competitions (the field's benchmark since 1982) and Kaggle:

1. **LightGBM dominates applied forecasting with covariates.** In M5 (Walmart sales, 42k series), the majority of winners and nearly all of the top 50 used LightGBM ([M5 findings, IJF](https://www.sciencedirect.com/science/article/pii/S0169207021001874)). Reasons: fast, minimal preprocessing, few knobs, handles categoricals & missing values ([Time Series Handbook ch. 8](https://phdinds-aim.github.io/time_series_handbook/08_WinningestMethods/lightgbm_m5_forecasting.html)).
2. **Cross-learning beats per-series models.** All M5 winning teams trained *one model over many related series* (pooled data with series-ID features) rather than one model per series ([Kaggle forecasting learnings](https://arxiv.org/pdf/2009.07701)). If you have multiple freight lanes, pool them.
3. **Ensembling is nearly free accuracy.** The M5 winner averaged **220 LightGBM models** (per-store, per-category, per-department, recursive + non-recursive variants). Even averaging 3–4 diverse models (statistical + GBM) reliably beats any single one.
4. **Simple baselines are shockingly strong.** In M5, exponential smoothing beat 92.5% of all submissions. Never report a model without its naive/seasonal-naive comparison.
5. **Deep learning earns its cost only sometimes** — an extra 3–10% at much higher compute, worthwhile in high-stakes domains ([TDS: DL vs statistics](https://towardsdatascience.com/time-series-forecasting-deep-learning-vs-statistics-who-wins-c568389d02df/)).
6. **Feature engineering > model choice > hyperparameter tuning**, in that order of impact. The ASHRAE competition postmortem: "gradient boosting machines and careful pre-processing work best" ([ASHRAE lessons](https://arxiv.org/pdf/2202.02898)).

### 5.2 Tweaks that improve unseen-set (test) performance

**Features (biggest lever):**
- Lags at meaningful horizons: $t-1$, $t-7$, $t-14$, $t-28$ for daily; align to the seasonality of your series.
- Rolling means/std/min/max over multiple windows — always `shift(1)` first.
- Calendar features (day-of-week, month, holidays); encode cyclically ($\sin/\cos(2\pi \cdot \text{month}/12)$) for linear/NN models.
- Domain exogenous drivers (for freight: bunker fuel price, port congestion indices, capacity/supply metrics, BDI/SCFI indices — port congestion materially improves rate forecasts ([Frontiers study](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1545471/full))) — but only **lagged** values you'd actually have at prediction time.
- Diverse feature *sets* across ensemble members beats one giant feature set.

**Target transformation (fixes tree extrapolation):**
- Predict $\Delta y_t = y_t - y_{t-1}$ or $\log(y_t / y_{t-1})$ instead of the level; reconstruct by cumulating. Trees can then handle trends outside the training range.
- $\log$ target for right-skewed, multiplicative series (freight rates qualify).

**Validation discipline:**
- Tune hyperparameters against expanding/sliding-window CV, *never* random K-fold.
- Make the dev set resemble the test regime (recent data), and use **early stopping on the temporal dev fold** — the single most effective GBM regularizer.
- Gap/embargo between train and validation ≥ max feature lag (§2.4).

**GBM hyperparameters that matter most (roughly in order):** learning rate (0.01–0.1) with many estimators + early stopping; `num_leaves`/`max_depth` (keep modest — time series are noisy); `min_child_samples` / `min_data_in_leaf` (raise it to resist noise); subsample & colsample (~0.7–0.9); L1/L2 regularization.

**Robustness to noise & outliers:**
- Train with MAE / Huber / quantile loss instead of MSE when outliers exist (MSE lets one COVID spike dominate the gradient).
- Quantile regression (LightGBM `objective="quantile"`) gives prediction intervals — valuable for a rate-negotiation use case.

**Multi-step strategy:** prefer **direct** (per-horizon models) for GBMs at longer horizons; recursive error compounding hits trees hard ([epistemic error decomposition](https://arxiv.org/pdf/2511.11461)). The M5 winner hedged by ensembling both.

**Final-model ritual:** after selecting everything on CV, retrain on train+dev with the chosen settings (freeze the early-stopped iteration count from CV), predict the untouched test block exactly once, and report against the naive baseline with MASE.

### 5.3 Anti-patterns checklist

- ❌ Random shuffling / random K-fold on time series.
- ❌ Scaling, imputing, or target-encoding using statistics from the full series.
- ❌ Centered rolling windows or un-shifted rolling features.
- ❌ Dropping the COVID-era spike as "outliers" when the test period contains a regime like it.
- ❌ Feeding raw trending levels to a tree model and wondering why forecasts go flat.
- ❌ Reporting a single holdout score without a naive baseline or CV variance.
- ❌ Tuning on the test set ("just checking one more config") — that's how leaderboard scores die on unseen data.

---

## Sources

**Lags & stationarity:** [statsmodels — Stationarity and detrending (ADF/KPSS)](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html) · [FPP3 ch. 9.1 — Stationarity and differencing](https://otexts.com/fpp3/stationarity.html) · [Statistical tests for stationarity — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/statistical-tests-to-check-stationarity-in-time-series-part-1/) · [When a time series only quacks like a duck (TDS)](https://towardsdatascience.com/when-a-time-series-only-quacks-like-a-duck-10de9e165e/) · [Test stationarity in R — ADF, KPSS, differencing](https://r-statistics.co/Test-Stationarity-in-R.html)

**Data problems & cleaning:** [FPP2 — Missing values and outliers](https://otexts.com/fpp2/missing-outliers.html) · [How to Deal with Time Series Outliers (TDS)](https://towardsdatascience.com/how-to-deal-with-time-series-outliers-28b217c7f6c2/) · [Finding outliers in time series (TDS)](https://towardsdatascience.com/the-ultimate-guide-to-finding-outliers-in-your-time-series-data-part-3-0ff73ce28ca3-2/) · [Improving Time Series Data Quality (ResearchGate)](https://www.researchgate.net/publication/391552163_Improving_Time_Series_Data_Quality_Identifying_Outliers_and_Handling_Missing_Values_in_a_Multilocation_Gas_and_Weather_Dataset) · [Handling Gaps in Time Series (TDS)](https://towardsdatascience.com/handling-gaps-in-time-series-dc47ae883990/)

**Splitting & CV:** [Walk-forward (rolling-origin) CV — MetricGate](https://metricgate.com/docs/time-series-walk-forward-cv/) · [Expanding-window CV](https://insightful-data-lab.com/2025/08/24/expanding-window-cross-validation/) · [Ultimate Guide to Time Series CV](https://www.numberanalytics.com/blog/ultimate-guide-time-series-cv) · [Time Series CV — Analytics Vidhya](https://www.analyticsvidhya.com/blog/2026/03/time-series-cross-validation/)

**Models & benchmarks:** [M5 Accuracy competition: results, findings, conclusions (IJF)](https://www.sciencedirect.com/science/article/pii/S0169207021001874) · [LightGBM & M5 — Time Series Handbook](https://phdinds-aim.github.io/time_series_handbook/08_WinningestMethods/lightgbm_m5_forecasting.html) · [Learnings from Kaggle's forecasting competitions (arXiv)](https://arxiv.org/pdf/2009.07701) · [Nixtla: comparing statistical, ML, and neural methods](https://nixtlaverse.nixtla.io/neuralforecast/docs/tutorials/comparing_methods.html) · [Stats vs ML vs DL for time series](https://blog.reachsumit.com/posts/2022/12/stats-vs-ml-for-ts/) · [DL vs statistics — who wins? (TDS)](https://towardsdatascience.com/time-series-forecasting-deep-learning-vs-statistics-who-wins-c568389d02df/) · [ASHRAE GEP-III lessons learned (arXiv)](https://arxiv.org/pdf/2202.02898)

**Multi-step strategies:** [Recursive vs Direct — MetricGate](https://metricgate.com/blogs/recursive-vs-direct-multistep-forecasting/) · [Recursive, direct & hybrid strategies](https://letsdatascience.com/blog/multi-step-time-series-forecasting-recursive-direct-and-hybrid-strategies) · [Epistemic error decomposition for multi-step forecasting (arXiv)](https://arxiv.org/pdf/2511.11461)

**Freight-rate domain:** [ML in freight rate forecasting — systematic review (Maritime Econ. & Logistics)](https://link.springer.com/article/10.1057/s41278-025-00334-3) · [Comparative evaluation of ML for container freight rates (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S209252122500015X) · [Prophet for container freight rates (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0967070X23000185) · [Port congestion & freight rate dynamics (Frontiers)](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1545471/full) · [ANNs vs conventional models for freight rates (Springer)](https://link.springer.com/article/10.1057/s41278-020-00156-5)
