# Advanced ML Debugging Patterns

Reference guide for diagnosing and fixing complex ML issues.

## Neural Network Debugging

### Sanity Checks (Do These First)

```python
# 1. Overfit to single batch
model.train()
batch = next(iter(train_loader))
for i in range(1000):
    optimizer.zero_grad()
    loss = criterion(model(batch['x']), batch['y'])
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"Step {i}: loss = {loss.item():.6f}")
# If loss doesn't go to ~0, architecture or code is broken

# 2. Verify forward pass dimensions
x = torch.randn(2, *input_shape)
try:
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
except Exception as e:
    print(f"Forward pass failed: {e}")

# 3. Count parameters
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total:,} total, {trainable:,} trainable")
```

### Layer-by-Layer Activation Analysis

```python
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
for name, layer in model.named_modules():
    if isinstance(layer, (nn.Linear, nn.Conv2d, nn.LSTM)):
        layer.register_forward_hook(get_activation(name))

# Run forward pass
_ = model(sample_input)

# Analyze
for name, act in activations.items():
    print(f"{name}:")
    print(f"  shape: {act.shape}")
    print(f"  mean: {act.mean():.4f}, std: {act.std():.4f}")
    print(f"  min: {act.min():.4f}, max: {act.max():.4f}")
    dead_neurons = (act.abs() < 1e-6).float().mean()
    print(f"  dead neurons: {dead_neurons:.2%}")
```

### Learning Rate Finder

```python
def lr_finder(model, train_loader, criterion, optimizer, start_lr=1e-7, end_lr=10, num_iter=100):
    lrs, losses = [], []
    model.train()
    
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    iterator = iter(train_loader)
    for i in range(num_iter):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        
        optimizer.zero_grad()
        loss = criterion(model(batch['x']), batch['y'])
        loss.backward()
        optimizer.step()
        
        lrs.append(lr)
        losses.append(loss.item())
        
        lr *= lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if loss.item() > 4 * min(losses):
            break
    
    return lrs, losses
# Plot and pick LR where loss is still decreasing steeply
```

## Data Quality Debugging

### Distribution Shift Detection

```python
from scipy import stats

def detect_drift(train_col, test_col, threshold=0.05):
    """Detect distribution shift using KS test for continuous, chi2 for categorical"""
    if train_col.dtype in ['float64', 'float32', 'int64', 'int32']:
        stat, pval = stats.ks_2samp(train_col.dropna(), test_col.dropna())
        test_type = 'KS'
    else:
        train_counts = train_col.value_counts(normalize=True)
        test_counts = test_col.value_counts(normalize=True)
        all_cats = set(train_counts.index) | set(test_counts.index)
        train_freq = [train_counts.get(c, 0) for c in all_cats]
        test_freq = [test_counts.get(c, 0) for c in all_cats]
        stat, pval = stats.chisquare(test_freq, train_freq)
        test_type = 'Chi2'
    
    return {
        'test': test_type,
        'statistic': stat,
        'pvalue': pval,
        'drift_detected': pval < threshold
    }
```

### Feature Importance Sanity Check

```python
def suspicious_features(model, X, y, feature_names):
    """Identify features with suspiciously high importance (potential leakage)"""
    from sklearn.inspection import permutation_importance
    
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    suspicious = []
    mean_importance = result.importances_mean.mean()
    
    for i, (mean, std) in enumerate(zip(result.importances_mean, result.importances_std)):
        if mean > 5 * mean_importance:  # 5x average is suspicious
            suspicious.append({
                'feature': feature_names[i],
                'importance': mean,
                'std': std,
                'ratio_to_avg': mean / mean_importance
            })
    
    return sorted(suspicious, key=lambda x: -x['importance'])
```

## Common Bug Patterns

### 1. Incorrect Tensor Operations

```python
# BUG: In-place operation breaks autograd
x = x + 1  # OK
x += 1     # BREAKS GRADIENT

# BUG: Detaching accidentally
features = encoder(x).detach()  # No gradients flow back!
features = encoder(x)  # Correct

# BUG: Wrong dimension in softmax
probs = F.softmax(logits, dim=0)  # Wrong! Softmax over batch
probs = F.softmax(logits, dim=1)  # Correct: over classes
```

### 2. Data Type Mismatches

```python
# BUG: Float labels for classification
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels.float())  # WRONG
loss = criterion(logits, labels.long())   # Correct

# BUG: Wrong dtype for regression
criterion = nn.MSELoss()
loss = criterion(preds, targets.long())   # WRONG
loss = criterion(preds.float(), targets.float())  # Correct
```

### 3. BatchNorm/Dropout Mode Errors

```python
# BUG: Forgot to switch mode
model.train()
# ... training loop ...
val_loss = evaluate(model, val_loader)  # WRONG: still in train mode

# Correct
model.eval()
with torch.no_grad():
    val_loss = evaluate(model, val_loader)
model.train()
```

### 4. Learning Rate Schedule Bugs

```python
# BUG: Scheduler step in wrong place
for epoch in range(epochs):
    for batch in dataloader:
        scheduler.step()  # WRONG: steps every batch
        ...

# Correct: step per epoch
for epoch in range(epochs):
    for batch in dataloader:
        ...
    scheduler.step()  # Step after epoch
```

## Profiling Tools

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    model(sample_input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### Memory Profiling

```python
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

## Reproducibility Checklist

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call at start of script
set_seed(42)
```

**Additional reproducibility factors:**
- DataLoader with `worker_init_fn` for multi-worker consistency
- Same hardware (CPU vs GPU can give different results)
- Same library versions
- Deterministic algorithms: `torch.use_deterministic_algorithms(True)`
