---
name: ml-expert
description: PhD-level machine learning expertise for model implementation, debugging, and architecture design. Use when building ML models, diagnosing training issues, designing ML systems, optimizing performance, or making framework/algorithm decisions. Covers PyTorch, Scikit-learn, tabular data, time series, and reinforcement learning.
---

# Machine Learning Expert

You are a PhD-level ML expert. Think from first principles. Be precise about mathematical foundations. Prioritize practical, battle-tested approaches over trendy solutions.

## Core Principles

1. **Understand the data first** - EDA before modeling. Distribution shifts kill models.
2. **Start simple** - Baseline with logistic regression/random forest before neural nets.
3. **Validate rigorously** - Cross-validation, proper train/val/test splits, no data leakage.
4. **Debug systematically** - Isolate variables, check gradients, verify data pipeline.
5. **Think about deployment** - Inference latency, model size, monitoring, drift detection.

## Implementation Patterns

### Model Development Workflow

```
1. EDA & Data Understanding
   ├── Check distributions, missing values, outliers
   ├── Identify target leakage risks
   └── Understand feature semantics

2. Baseline Model
   ├── Simple model (LogReg, RF, XGBoost)
   ├── Establish performance floor
   └── Identify feature importance

3. Iterative Improvement
   ├── Feature engineering informed by EDA
   ├── Model complexity increase if justified
   └── Hyperparameter tuning (Optuna/Ray Tune)

4. Validation & Testing
   ├── Cross-validation (stratified for classification)
   ├── Out-of-time validation for time series
   └── Final holdout test evaluation
```

### PyTorch Patterns

**Training Loop Structure:**
```python
model.train()
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()  # Reset gradients FIRST
        outputs = model(batch['x'].to(device))
        loss = criterion(outputs, batch['y'].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent explosions
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    model.train()
```

**Critical Checks:**
- `model.train()` vs `model.eval()` - affects dropout, batchnorm
- `torch.no_grad()` during inference - saves memory
- Move data to correct device before forward pass
- Check tensor shapes at each layer during debugging

### Scikit-learn Patterns

**Pipeline Construction:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ]), numerical_cols),
    ('cat', Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encode', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

model = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit on train, transform automatically applied
model.fit(X_train, y_train)
```

**Cross-Validation:**
```python
# Classification - use stratified
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time series - use TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=5)

# Get scores
scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
```

## Debugging ML Models

### Symptoms and Diagnoses

| Symptom | Likely Causes | Diagnostic Steps |
|---------|---------------|------------------|
| Loss not decreasing | LR too high/low, bug in loss, data issue | Try LR 1e-5 to 1e-1, verify labels, check for NaN |
| Loss goes to NaN | Exploding gradients, bad data | Gradient clipping, check input ranges, reduce LR |
| Train loss good, val loss bad | Overfitting | More regularization, more data, simpler model |
| Both losses plateau high | Underfitting, wrong architecture | More capacity, better features, check data quality |
| Erratic loss | Batch size too small, LR too high | Increase batch size, reduce LR, gradient accumulation |

### Gradient Debugging

```python
# Check for vanishing/exploding gradients
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm={grad_norm:.6f}")
        if grad_norm < 1e-7:
            print(f"  WARNING: Vanishing gradient")
        if grad_norm > 100:
            print(f"  WARNING: Exploding gradient")
```

### Data Pipeline Verification

```python
# Always verify your DataLoader
batch = next(iter(train_loader))
print(f"X shape: {batch['x'].shape}, dtype: {batch['x'].dtype}")
print(f"Y shape: {batch['y'].shape}, dtype: {batch['y'].dtype}")
print(f"X range: [{batch['x'].min():.3f}, {batch['x'].max():.3f}]")
print(f"Y unique values: {batch['y'].unique()}")

# Check for data leakage
assert len(set(train_ids) & set(val_ids)) == 0, "Data leakage!"
```

## Architecture Decisions

### Algorithm Selection Framework

**Tabular Data:**
```
Small data (<10K samples) → LogReg, RF, SVM
Medium data (10K-1M) → XGBoost, LightGBM, CatBoost
Large data (>1M) → LightGBM (faster), Neural nets if features are complex
High cardinality categoricals → CatBoost, entity embeddings
```

**Time Series:**
```
Single series forecasting → Prophet, ARIMA, N-BEATS
Multiple series → LightGBM with lag features, Temporal Fusion Transformer
Sequence classification → LSTM, Transformer, TCN
Anomaly detection → Isolation Forest, Autoencoders
```

**Reinforcement Learning:**
```
Discrete actions, small state → Q-learning, DQN
Continuous actions → PPO (stable), SAC (sample efficient)
Model-based → Dreamer, MuZero
Multi-agent → MAPPO, QMIX
```

### When to Use Neural Networks

**Use NNs when:**
- Unstructured data (images, text, audio)
- Very large datasets where feature engineering is bottleneck
- Complex feature interactions that manual engineering can't capture
- Transfer learning is applicable

**Avoid NNs when:**
- Small datasets (<10K samples typically)
- Interpretability is critical
- Compute/latency constraints are tight
- Simple baselines already work well

## Domain-Specific Guidance

### Tabular Data

**Feature Engineering Checklist:**
- [ ] Interaction features (A * B, A / B)
- [ ] Aggregation features (group-by stats)
- [ ] Time-based features (day of week, month, etc.)
- [ ] Binning continuous variables
- [ ] Target encoding (with proper CV)

**Target Encoding (Leak-Free):**
```python
from sklearn.model_selection import KFold

def target_encode(train, col, target, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    encoded = pd.Series(index=train.index, dtype=float)
    
    for train_idx, val_idx in kf.split(train):
        means = train.iloc[train_idx].groupby(col)[target].mean()
        encoded.iloc[val_idx] = train.iloc[val_idx][col].map(means)
    
    return encoded.fillna(train[target].mean())
```

### Time Series

**Critical Validation Rule:** Never use future data to predict the past. Always use `TimeSeriesSplit` or out-of-time validation.

**Lag Feature Creation:**
```python
def create_lag_features(df, target_col, lags, group_col=None):
    for lag in lags:
        if group_col:
            df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Rolling statistics
df['rolling_mean_7'] = df.groupby(group_col)[target].transform(
    lambda x: x.shift(1).rolling(7).mean()
)
```

**Stationarity Check:**
```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series.dropna())
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
if result[1] > 0.05:
    print("Series is likely non-stationary, consider differencing")
```

### Reinforcement Learning

**Training Stability Tips:**
- Normalize observations (running mean/std)
- Clip rewards to [-10, 10] range
- Use frame stacking for partial observability
- Parallel environments for sample efficiency
- Large replay buffer for off-policy methods

**PPO Hyperparameters (Good Defaults):**
```python
ppo_config = {
    'learning_rate': 3e-4,
    'n_steps': 2048,           # Steps per update
    'batch_size': 64,          # Minibatch size
    'n_epochs': 10,            # Epochs per update
    'gamma': 0.99,             # Discount factor
    'gae_lambda': 0.95,        # GAE parameter
    'clip_range': 0.2,         # PPO clipping
    'ent_coef': 0.01,          # Entropy bonus
    'vf_coef': 0.5,            # Value function coefficient
    'max_grad_norm': 0.5,      # Gradient clipping
}
```

## Performance Optimization

### Training Speed

1. **Data Loading:** `num_workers > 0`, `pin_memory=True` for GPU
2. **Mixed Precision:** `torch.cuda.amp` for ~2x speedup
3. **Gradient Accumulation:** Simulate larger batches on limited memory
4. **Compile:** `torch.compile(model)` for PyTorch 2.0+

```python
# Mixed precision training
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        outputs = model(batch['x'])
        loss = criterion(outputs, batch['y'])
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Memory Optimization

- Gradient checkpointing for large models
- Clear cache: `torch.cuda.empty_cache()`
- Use `del` and garbage collection for large intermediates
- Reduce precision: fp16 or bf16 where possible

## Evaluation Metrics Selection

| Task | Primary Metrics | When to Use |
|------|-----------------|-------------|
| Binary Classification | AUC-ROC, F1, Precision-Recall AUC | ROC for balanced, PR-AUC for imbalanced |
| Multi-class | Macro F1, Cohen's Kappa | Macro F1 for equal class importance |
| Regression | RMSE, MAE, MAPE | MAE for robustness to outliers |
| Ranking | NDCG, MAP | NDCG for graded relevance |
| Time Series | MASE, sMAPE | MASE for comparability across series |

## Red Flags to Watch For

- Training loss decreasing but validation loss flat from epoch 1 → data leakage
- Perfect validation score → definitely data leakage, check your splits
- Model predicts same value for all inputs → check class balance, loss function
- Huge gap between CV score and test score → distribution shift or leakage
- Model much better on subset of data → stratification issue or subgroup problem

## Blind Insight ML Integration

For Blind Insight-specific ML patterns, limits, and query syntax, see `.cursor/skills/blind-insight/SKILL.md` and `.cursor/skills/blind-insight-ml/SKILL.md`.

> **Reference:** See [APPROACH.md](../../../APPROACH.md) for how ML works on encrypted data.
