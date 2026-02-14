# Appendix B: ML.NET Algorithm Cheat Sheet

This appendix serves as a quick reference guide for selecting and configuring ML.NET algorithms. Use it when deciding which trainer to use, tuning hyperparameters, or optimizing model performance.

---

## Algorithm Selection Flowchart

Use this decision tree to quickly identify the right algorithm family for your problem:

```
START: What are you trying to predict?
│
├─► A continuous number (price, temperature, score)?
│   └─► Use REGRESSION
│       ├─► Linear relationship? → FastTree, LightGbm, or Sdca
│       ├─► Complex patterns? → FastForest or LightGbm
│       └─► Fast training needed? → OnlineGradientDescent
│
├─► A category from 2 options (yes/no, spam/not-spam)?
│   └─► Use BINARY CLASSIFICATION
│       ├─► Interpretability matters? → SdcaLogisticRegression
│       ├─► Best accuracy needed? → LightGbm or FastTree
│       └─► Streaming data? → AveragedPerceptron
│
├─► A category from 3+ options (color, animal type, sentiment)?
│   └─► Use MULTICLASS CLASSIFICATION
│       ├─► Few classes (<10)? → SdcaMaximumEntropy
│       ├─► Many classes (10+)? → LightGbm or OneVersusAll
│       └─► Hierarchical classes? → OneVersusAll with custom base
│
├─► Groups or segments (customer segments, document clusters)?
│   └─► Use CLUSTERING
│       └─► KMeans (only option, works well)
│
├─► Unusual patterns or outliers (fraud, defects)?
│   └─► Use ANOMALY DETECTION
│       ├─► One-class problem? → RandomizedPca
│       └─► Time series? → SrCnn or SsaChangePointDetector
│
└─► Personalized suggestions (products, movies)?
    └─► Use RECOMMENDATION
        └─► MatrixFactorization (collaborative filtering)
```

---

## Regression Trainers

Use regression when predicting continuous numeric values.

| Trainer | Use Case | Key Hyperparameters | Pros | Cons |
|---------|----------|---------------------|------|------|
| **FastTreeRegressionTrainer** | General-purpose regression with nonlinear patterns | `NumberOfLeaves` (20-100), `NumberOfTrees` (100-500), `MinimumExampleCountPerLeaf` (10-50), `LearningRate` (0.1-0.3) | Fast training, handles missing values, feature importance built-in | Can overfit with too many trees |
| **FastForestRegressionTrainer** | When you need stable predictions and reduced overfitting | `NumberOfTrees` (100-500), `NumberOfLeaves` (20-50), `MinimumExampleCountPerLeaf` (1-20) | Robust to outliers, less prone to overfitting than FastTree | Slightly slower, less interpretable |
| **LightGbmRegressionTrainer** | Large datasets, best-in-class accuracy | `NumberOfLeaves` (31-127), `NumberOfIterations` (100-1000), `LearningRate` (0.01-0.1), `MinimumExampleCountPerLeaf` (20-100) | State-of-the-art accuracy, handles categorical features natively, memory efficient | Requires more tuning, can overfit |
| **SdcaRegressionTrainer** | Linear relationships, sparse features | `L1Regularization` (0-1), `L2Regularization` (0.0001-1), `MaximumNumberOfIterations` (10-100) | Very fast, works well with high-dimensional sparse data | Only captures linear relationships |
| **OnlineGradientDescentTrainer** | Streaming data, incremental learning | `LearningRate` (0.01-0.1), `L2Regularization` (0-1), `NumberOfIterations` (1-10) | Supports online learning, memory efficient | Requires feature normalization, linear only |
| **LbfgsPoissonRegressionTrainer** | Count data (events per time period) | `L1Regularization` (0-1), `L2Regularization` (0.0001-1), `OptimizationTolerance` (1e-7) | Natural for count predictions, prevents negative predictions | Assumes Poisson distribution |
| **GamRegressionTrainer** | Interpretable models with smooth functions | `NumberOfIterations` (100-9999), `MaximumBinCountPerFeature` (16-256), `LearningRate` (0.001-0.1) | Highly interpretable, captures nonlinear effects | Slower training, limited interactions |
| **OlsTrainer** | Simple linear regression, baseline model | `L2Regularization` (0-1) | Very fast, provides coefficients and p-values | Linear only, sensitive to outliers |

### Regression Code Example

```csharp
// LightGbm with tuned hyperparameters
var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
    .Append(mlContext.Regression.Trainers.LightGbm(new LightGbmRegressionTrainer.Options
    {
        NumberOfLeaves = 64,
        NumberOfIterations = 500,
        LearningRate = 0.05,
        MinimumExampleCountPerLeaf = 20,
        LabelColumnName = "Price",
        FeatureColumnName = "Features"
    }));
```

---

## Binary Classification Trainers

Use binary classification when predicting one of two possible outcomes.

| Trainer | Use Case | Key Hyperparameters | Pros | Cons |
|---------|----------|---------------------|------|------|
| **LightGbmBinaryTrainer** | Best accuracy, large datasets | `NumberOfLeaves` (31-127), `NumberOfIterations` (100-500), `LearningRate` (0.01-0.1), `MinimumExampleCountPerLeaf` (20-100) | Excellent accuracy, handles imbalanced data, fast | Complex to tune |
| **FastTreeBinaryTrainer** | General-purpose, interpretable | `NumberOfLeaves` (20-100), `NumberOfTrees` (100-300), `LearningRate` (0.1-0.3), `MinimumExampleCountPerLeaf` (10-50) | Good accuracy, provides feature importance | Can overfit on small datasets |
| **FastForestBinaryTrainer** | Robust classification, reduced variance | `NumberOfTrees` (100-500), `NumberOfLeaves` (20-50) | Stable predictions, handles noise well | Lower accuracy than boosting methods |
| **SdcaLogisticRegressionBinaryTrainer** | Linear problems, probability calibration | `L1Regularization` (0-1), `L2Regularization` (1e-4 to 1), `MaximumNumberOfIterations` (10-100) | Fast, well-calibrated probabilities, interpretable | Linear decision boundary only |
| **SdcaNonCalibratedBinaryTrainer** | When raw scores suffice | Same as SdcaLogisticRegression | Faster than calibrated version | Scores aren't true probabilities |
| **AveragedPerceptronTrainer** | Online learning, streaming | `LearningRate` (0.01-1), `NumberOfIterations` (1-20), `L2Regularization` (0-1) | Very fast, supports incremental updates | Linear only, requires normalization |
| **LinearSvmTrainer** | Maximum margin classification | `NumberOfIterations` (1-100), `Lambda` (regularization) | Good generalization on small datasets | Binary only, linear boundary |
| **LbfgsLogisticRegressionBinaryTrainer** | Probability estimates, sparse data | `L1Regularization` (0-1), `L2Regularization` (1e-4 to 1), `OptimizationTolerance` (1e-7) | Well-calibrated, handles high dimensions | Linear only |
| **PriorTrainer** | Baseline model (predicts majority class) | None | Instant training | Useless for actual predictions |
| **GamBinaryTrainer** | Interpretable + nonlinear | Same as GamRegression | Explainable smooth functions per feature | Slower, limited feature interactions |
| **SymbolicSgdLogisticRegressionBinaryTrainer** | Privacy-preserving, distributed | `NumberOfIterations`, `LearningRate` | Differentially private option | Experimental, linear only |

### Binary Classification Code Example

```csharp
// FastTree with class balancing
var pipeline = mlContext.BinaryClassification.Trainers.FastTree(
    new FastTreeBinaryTrainer.Options
    {
        NumberOfLeaves = 50,
        NumberOfTrees = 200,
        LearningRate = 0.2,
        MinimumExampleCountPerLeaf = 20,
        LabelColumnName = "IsSpam",
        FeatureColumnName = "Features",
        UnbalancedSets = true // Important for imbalanced data
    });
```

---

## Multiclass Classification Trainers

Use multiclass classification when predicting one of three or more categories.

| Trainer | Use Case | Key Hyperparameters | Pros | Cons |
|---------|----------|---------------------|------|------|
| **LightGbmMulticlassTrainer** | Best accuracy, any number of classes | `NumberOfLeaves` (31-127), `NumberOfIterations` (100-500), `LearningRate` (0.01-0.1) | State-of-the-art, handles many classes efficiently | Memory intensive with many classes |
| **SdcaMaximumEntropyTrainer** | Moderate class count, linear boundaries | `L1Regularization`, `L2Regularization`, `MaximumNumberOfIterations` | Fast training, good probability estimates | Linear only |
| **SdcaNonCalibratedMulticlassTrainer** | When probabilities not needed | Same as SdcaMaximumEntropy | Faster than calibrated | Raw scores only |
| **LbfgsMaximumEntropyTrainer** | Text classification, sparse features | `L1Regularization`, `L2Regularization`, `OptimizationTolerance` | Handles high dimensions, well-calibrated | Linear only |
| **NaiveBayesMulticlassTrainer** | Text classification baseline | None (parameter-free) | Extremely fast, works with little data | Assumes feature independence |
| **OneVersusAllTrainer** | Wrapper for any binary classifier | Depends on base learner | Flexible, use any binary classifier | Slower with many classes |
| **PairwiseCouplingTrainer** | When OneVsAll struggles | Depends on base learner | Better calibration than OvA | O(n²) classifiers for n classes |
| **KMeansTrainer** + classification | Semi-supervised scenarios | See clustering section | Can leverage unlabeled data | Two-stage process |

### Multiclass Code Example

```csharp
// OneVersusAll with LightGbm base learner
var lightGbmBinary = mlContext.BinaryClassification.Trainers.LightGbm(
    new LightGbmBinaryTrainer.Options
    {
        NumberOfIterations = 200,
        LearningRate = 0.1
    });

var pipeline = mlContext.MulticlassClassification.Trainers.OneVersusAll(
    binaryEstimator: lightGbmBinary,
    labelColumnName: "Category");
```

---

## Clustering Trainers

Use clustering to discover natural groupings in unlabeled data.

| Trainer | Use Case | Key Hyperparameters | Pros | Cons |
|---------|----------|---------------------|------|------|
| **KMeansTrainer** | Customer segmentation, document grouping | `NumberOfClusters` (2-20), `MaximumNumberOfIterations` (100-1000), `OptimizationTolerance` (1e-4) | Fast, scalable, well-understood | Must specify k, sensitive to initialization |

### Choosing the Number of Clusters

```csharp
// Elbow method - try different k values
var results = new List<(int k, double avgDistance)>();

for (int k = 2; k <= 10; k++)
{
    var trainer = mlContext.Clustering.Trainers.KMeans(
        numberOfClusters: k,
        featureColumnName: "Features");
    
    var model = trainer.Fit(data);
    var predictions = model.Transform(data);
    var metrics = mlContext.Clustering.Evaluate(predictions);
    
    results.Add((k, metrics.AverageDistance));
}

// Plot results - look for "elbow" where improvement slows
```

---

## Anomaly Detection Trainers

Use anomaly detection to identify unusual patterns or outliers.

| Trainer | Use Case | Key Hyperparameters | Pros | Cons |
|---------|----------|---------------------|------|------|
| **RandomizedPcaTrainer** | Detecting outliers in high-dimensional data | `Rank` (10-100), `Oversampling` (20), `EnsureZeroMean` (true) | Fast, works with many features | Assumes linear subspace |
| **SsaSpikeDetector** | Sudden spikes in time series | `PvalueHistoryLength`, `Confidence`, `TrainingWindowSize`, `SeasonalityWindowSize` | Real-time detection, no training phase | Time series only |
| **SsaChangePointDetector** | Regime changes in time series | `Confidence` (80-99), `ChangeHistoryLength`, `TrainingWindowSize` | Detects trend shifts | Sensitive to seasonality |
| **SrCnnAnomalyDetector** | Complex time series anomalies | `BatchSize`, `Threshold` (0.1-0.5), `Sensitivity` (0-100) | Deep learning accuracy, handles complex patterns | Requires more data |

### Anomaly Detection Code Example

```csharp
// Randomized PCA for fraud detection
var pipeline = mlContext.Transforms.Concatenate("Features", featureColumns)
    .Append(mlContext.AnomalyDetection.Trainers.RandomizedPca(
        featureColumnName: "Features",
        rank: 20,
        oversampling: 20,
        ensureZeroMean: true));

// Time series spike detection
var spikeDetector = mlContext.Transforms.DetectSpikeBySsa(
    outputColumnName: "Prediction",
    inputColumnName: "Value",
    confidence: 95,
    pvalueHistoryLength: 30,
    trainingWindowSize: 120,
    seasonalityWindowSize: 24);
```

---

## Recommendation Trainers

Use recommendation to suggest items based on user-item interactions.

| Trainer | Use Case | Key Hyperparameters | Pros | Cons |
|---------|----------|---------------------|------|------|
| **MatrixFactorizationTrainer** | Collaborative filtering (user-item ratings) | `NumberOfIterations` (20-100), `ApproximationRank` (8-256), `LearningRate` (0.01-0.1), `Lambda` (regularization, 0.01-0.1) | Handles sparse matrices, discovers latent factors | Cold start problem, needs ratings |

### Matrix Factorization Options Explained

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| `ApproximationRank` | 8-256 | Number of latent factors. Higher = more expressive but slower |
| `NumberOfIterations` | 20-100 | Training epochs. Monitor for convergence |
| `LearningRate` | 0.01-0.1 | Step size. Lower = stable but slow |
| `Lambda` | 0.01-0.1 | L2 regularization. Higher = less overfitting |
| `Alpha` | 1-40 | Confidence weight for implicit feedback |
| `C` | 0.0001-0.01 | Offset for confidence calculation |

### Recommendation Code Example

```csharp
// Explicit ratings (1-5 stars)
var trainer = mlContext.Recommendation().Trainers.MatrixFactorization(
    new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "UserIdEncoded",
        MatrixRowIndexColumnName = "MovieIdEncoded",
        LabelColumnName = "Rating",
        NumberOfIterations = 50,
        ApproximationRank = 64,
        LearningRate = 0.05,
        Lambda = 0.025
    });

// Implicit feedback (views, clicks)
var implicitTrainer = mlContext.Recommendation().Trainers.MatrixFactorization(
    new MatrixFactorizationTrainer.Options
    {
        MatrixColumnIndexColumnName = "UserIdEncoded",
        MatrixRowIndexColumnName = "ProductIdEncoded",
        LabelColumnName = "Interactions",
        LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
        Alpha = 10,
        Lambda = 0.01,
        ApproximationRank = 32
    });
```

---

## Performance Optimization Tips

### Data Loading Optimization

| Technique | When to Use | Impact |
|-----------|-------------|--------|
| `LoadFromEnumerable` with streaming | Large datasets that don't fit in memory | Reduces memory 10x+ |
| Specify column types explicitly | Always | Faster parsing, catches errors early |
| Use `CreateDatabaseLoader` | SQL data sources | Streaming from database |
| Cache after transforms | Multiple passes over same data | 2-5x faster training |

```csharp
// Cache transformed data for multi-pass algorithms
var cachedData = mlContext.Data.Cache(transformedData);
```

### Training Speed Optimization

| Technique | Applicable Trainers | Speedup |
|-----------|---------------------|---------|
| Reduce `NumberOfIterations` | All iterative trainers | Linear |
| Reduce `NumberOfTrees` | Tree-based trainers | Linear |
| Increase `MinimumExampleCountPerLeaf` | Tree-based trainers | 20-50% |
| Use `FeatureSelectionUsingMutualInformation` | All | Depends on reduction |
| Enable multi-threading | Most trainers | 2-8x on multi-core |

```csharp
// Configure parallelism
mlContext.Options.NumberOfThreads = Environment.ProcessorCount;
```

### Memory Optimization

| Technique | When to Use | Memory Reduction |
|-----------|-------------|------------------|
| `KeyToValueMappingEstimator` | Categorical features | Varies |
| `NormalizeMeanVariance` | Numerical features | None, but improves convergence |
| Reduce `ApproximationRank` | Matrix factorization | Proportional to reduction |
| Use sparse vectors | High-dimensional sparse data | 10-100x |
| `PreviewFeaturizedData` then filter | Feature engineering | Debug only |

### Hyperparameter Tuning Strategy

1. **Start with defaults** - ML.NET defaults are reasonable
2. **Tune learning rate first** - Most impactful parameter
3. **Adjust model complexity** - Trees/leaves for tree-based, rank for matrix factorization
4. **Add regularization** - Only if overfitting observed
5. **Use cross-validation** - Essential for reliable evaluation

```csharp
// Cross-validation for hyperparameter selection
var cvResults = mlContext.Regression.CrossValidate(
    data: trainData,
    estimator: pipeline,
    numberOfFolds: 5,
    labelColumnName: "Target");

var avgRSquared = cvResults.Average(r => r.Metrics.RSquared);
```

---

## Quick Reference: Trainer Selection by Scenario

| Scenario | Recommended Trainer | Runner-Up |
|----------|---------------------|-----------|
| House price prediction | LightGbmRegression | FastTreeRegression |
| Spam detection | FastTreeBinary | LightGbmBinary |
| Sentiment analysis (pos/neg) | SdcaLogisticRegression | LightGbmBinary |
| Document classification | LbfgsMaximumEntropy | NaiveBayes |
| Customer segmentation | KMeans | — |
| Fraud detection | RandomizedPca | — |
| Movie recommendations | MatrixFactorization | — |
| Stock anomaly detection | SrCnnAnomalyDetector | SsaSpikeDetector |
| Image classification | ImageClassification (transfer) | — |
| Time series forecasting | SsaForecasting | — |

---

## Version Compatibility

This cheat sheet covers ML.NET 5.0.0 trainers. Key changes from previous versions:

- **ML.NET 2.0+**: Added `SrCnnAnomalyDetector` for time series
- **ML.NET 3.0+**: Improved `MatrixFactorization` with implicit feedback
- **ML.NET 4.0+**: Enhanced `LightGbm` categorical handling
- **ML.NET 5.0**: Stability improvements, .NET 9 support

Always check the [ML.NET API documentation](https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml) for the latest parameter options and deprecated trainers.

---

*Use this appendix as your go-to reference when building ML.NET solutions. Bookmark it, print it, keep it handy.*
