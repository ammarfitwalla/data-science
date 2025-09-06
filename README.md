# data-science

## Weekly Sales Forecasting (Media + Ads)

This README explains, in simple words, what we did and how to run it.

## What we did

1. **Loaded data**
   - Your file has weekly rows per division (A, E, V, …).
   - Columns include: `date`, `sales`, views, and ad impressions.

2. **Aggregated divisions**
   - We **summed** all divisions by `date` to get **one row per week** (total company numbers).

3. **EDA (exploratory)**
   - Plotted sales over time.
   - Checked correlations.
   - Found: past sales and ad impressions matter. There are big spikes.

4. **Feature engineering (weekly)**
   - Created **log** versions of sales and exogenous features (to reduce skew).
   - Created **lags** of sales: `sales_lag_1`, `sales_lag_4`, `sales_lag_8`.
   - Added **rolling mean**: `sales_mean_4w`.
   - Added **seasonality**: week-of-year **sin/cos** features.
   - Built two feature lists:
     - `ML_FEATURE_COLS` → for **XGBoost** (keeps lags/rolls).
     - `SARIMAX_EXOG_COLS` → for **SARIMAX** (no target lags/rolls).

5. **Preprocessing & split**
   - Time-based split: **first 80%** = train, **last 20%** = test.
   - (Optional) Standardized features for linear models (XGBoost does not need it).

6. **Baselines**
   - **Naive** (last week’s sales).
   - **Moving Average** (4-week, 8-week).
   - Used as reference.

7. **Models**
   - **SARIMAX** with exogenous features and weekly seasonality `(1,1,1) x (1,0,1,52)`.
   - **XGBoost** using engineered features.

8. **Results (example from your run)**
   - **XGBoost**: best (≈ **6.6% MAPE** on test).
   - **SARIMAX** and moving averages were worse.

---

## How to run (quick)

### 1) Install
```bash
pip install -U pandas numpy matplotlib seaborn statsmodels pmdarima scikit-learn xgboost
