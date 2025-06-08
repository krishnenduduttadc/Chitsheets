---
title: "Machine Learning Algorithms Deep Dive"
author: "Codeium"
date: "2025-02-15"
geometry: "margin=2cm"
output: 
  pdf_document:
    toc: true
    toc_depth: 3
---

# Machine Learning Algorithms Deep Dive

## 1. Support Vector Machine (SVM)

### Linear SVM
```
Decision Boundary: w·x + b = 0
Margin Constraints:
  Positive class: w·x + b ≥ 1
  Negative class: w·x + b ≤ -1

Optimization Problem:
  Minimize: (1/2)||w||² 
  Subject to: y_i(w·x_i + b) ≥ 1

Soft Margin SVM (C parameter):
  Minimize: (1/2)||w||² + C∑ξᵢ
  Subject to: y_i(w·x_i + b) ≥ 1 - ξᵢ
```

### Kernel SVM
```
Kernel Functions:
1. Linear: K(x,y) = x·y
2. Polynomial: K(x,y) = (γx·y + r)^d
3. RBF: K(x,y) = exp(-γ||x-y||²)
4. Sigmoid: K(x,y) = tanh(γx·y + r)

Decision Function:
f(x) = sign(∑(α_i y_i K(x_i,x)) + b)
```

### SVM Hyperparameters
```
C: Regularization parameter
  Small C → Larger margin, more violations
  Large C → Smaller margin, fewer violations

γ (gamma): Kernel coefficient
  Small γ → Larger influence radius
  Large γ → Smaller influence radius
```

## 2. Gradient Descent

### Types of Gradient Descent

#### Batch Gradient Descent
```
For all parameters θ:
θ = θ - α × (∂J/∂θ)

Update using entire dataset
Memory: O(n)
```

#### Stochastic Gradient Descent (SGD)
```
For each training example i:
θ = θ - α × (∂J_i/∂θ)

Update using single example
Memory: O(1)
```

#### Mini-batch Gradient Descent
```
For each mini-batch B:
θ = θ - α × (∂J_B/∂θ)

Update using batch of b examples
Memory: O(b)
```

### Learning Rate Schedules
```
1. Time-based decay:
   α(t) = α₀/(1 + kt)

2. Step decay:
   α(t) = α₀ × 0.1^⌊t/d⌋

3. Exponential decay:
   α(t) = α₀ × e^(-kt)
```

### Gradient Descent Variants
```
1. Momentum:
   v(t) = βv(t-1) + (1-β)∇J(θ)
   θ = θ - αv(t)

2. RMSprop:
   s(t) = βs(t-1) + (1-β)(∇J(θ))²
   θ = θ - α∇J(θ)/√(s(t) + ε)

3. Adam:
   m(t) = β₁m(t-1) + (1-β₁)∇J(θ)
   v(t) = β₂v(t-1) + (1-β₂)(∇J(θ))²
   θ = θ - α × m(t)/(√v(t) + ε)
```

## 3. Naive Bayes

### Types of Naive Bayes

#### Gaussian Naive Bayes
```
P(x_i|y) = (1/√(2πσ²_y))exp(-(x_i-μ_y)²/(2σ²_y))

For continuous features:
μ_y = mean of x for class y
σ²_y = variance of x for class y
```

#### Multinomial Naive Bayes
```
P(x_i|y) = (count(x_i,y) + α)/(count(y) + αn)

For discrete features (e.g., text):
α = smoothing parameter (Laplace smoothing)
n = number of features
```

#### Bernoulli Naive Bayes
```
P(x_i|y) = P(i|y)^x_i × (1-P(i|y))^(1-x_i)

For binary features:
P(i|y) = probability of feature i appearing in class y
```

### Naive Bayes Decision Rule
```
ŷ = argmax_y P(y)∏P(x_i|y)

In log space (to prevent underflow):
ŷ = argmax_y log(P(y)) + ∑log(P(x_i|y))
```

## 4. K-Means Clustering

### Algorithm Steps
```
1. Initialize k centroids randomly
2. Repeat until convergence:
   a. Assign points to nearest centroid
   b. Update centroids as mean of assigned points

Assignment step:
c_i = argmin_j ||x_i - μ_j||²

Update step:
μ_j = (1/|S_j|)∑(x_i) for x_i in cluster j
```

### Initialization Methods
```
1. Random Initialization:
   Select k points randomly

2. K-means++:
   a. Choose first centroid randomly
   b. For remaining k-1 centroids:
      P(x) ∝ min(D(x)²) to all centroids
```

### Choosing K
```
Elbow Method:
Plot inertia vs k
Inertia = ∑min||x_i - μ_j||²

Silhouette Score:
s(i) = (b(i) - a(i))/max(a(i), b(i))
where:
a(i) = mean intra-cluster distance
b(i) = mean nearest-cluster distance
```

## 5. Polynomial Regression

### Model Form
```
y = β₀ + β₁x + β₂x² + ... + βₙxⁿ + ε

Matrix form:
X = [1  x  x²  ...  xⁿ]
β = [β₀ β₁ β₂ ... βₙ]ᵀ
y = Xβ + ε
```

### Feature Generation
```
Original: x
Polynomial: [1, x, x², x³, ..., xⁿ]

Example (degree=2):
x = [1, 2, 3]
X = [[1, 1, 1],
     [1, 2, 4],
     [1, 3, 9]]
```

### Regularization
```
Ridge (L2):
min ||y - Xβ||² + λ||β||²

Lasso (L1):
min ||y - Xβ||² + λ|β|
```

### Avoiding Overfitting
1. Cross-validation for degree selection
2. Feature scaling crucial
   ```
   x_scaled = (x - μ)/σ
   ```
3. Regularization parameter tuning

## 6. Common Implementation Tips

### Feature Scaling
```
For all algorithms except Naive Bayes:
- StandardScaler
- MinMaxScaler
- RobustScaler
```

### Hyperparameter Selection
```
SVM:
- C: [0.1, 1, 10, 100]
- gamma: ['scale', 'auto', 0.1, 0.01]
- kernel: ['rbf', 'linear', 'poly']

K-Means:
- n_clusters: [2-10]
- init: ['k-means++', 'random']
- n_init: [10, 20, 30]

Polynomial Regression:
- degree: [1-5]
- alpha (regularization): [0.001, 0.01, 0.1, 1]
```

### Performance Metrics
```
Clustering:
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

Regression:
- R² Score
- MSE/RMSE
- MAE

Classification:
- Accuracy
- Precision/Recall
- F1 Score
- ROC-AUC
```

Remember:
- Always scale features (except for Naive Bayes)
- Use cross-validation
- Consider computational complexity
- Monitor for overfitting
- Validate assumptions
