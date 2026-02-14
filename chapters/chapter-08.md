# Chapter 8: Regression — Predicting Numbers

When a ride-share app estimates your fare, when a real estate website predicts home values, or when a financial model forecasts next quarter's revenue—they're all using regression. Unlike classification, which assigns categories, regression predicts continuous numerical values. It's one of the most practical and widely-used techniques in machine learning, and understanding it deeply will serve you well across countless domains.

In this chapter, we'll build regression intuition from the ground up, explore the mathematics that make it work (without getting lost in academic notation), and implement a complete taxi fare prediction system using ML.NET. By the end, you'll understand not just *how* to build regression models, but *why* they work and when they fail.

## The Essence of Regression

Regression finds the mathematical relationship between input features and a numerical output. At its core, it answers the question: "Given what I know about this situation, what number should I predict?"

Consider predicting a home's sale price. You might know the square footage, number of bedrooms, neighborhood, and age. Regression learns the weight each factor contributes:

```
Price ≈ $150 × SquareFeet + $10,000 × Bedrooms + $50,000 × NeighborhoodScore - $1,000 × Age + $75,000
```

That final constant ($75,000) is the **intercept** or **bias term**—the baseline prediction when all features are zero. The multipliers ($150, $10,000, etc.) are the **coefficients** or **weights**—they tell us how much each feature influences the prediction.

[FIGURE: Scatter plot showing house prices vs. square footage with a fitted regression line. The slope represents the coefficient, and the y-intercept represents the bias term. Points scatter around the line, illustrating that regression finds the "best fit" through noisy data.]

### Simple Linear Regression

Simple linear regression uses a single feature to predict an outcome:

```
ŷ = wx + b
```

Where:
- `ŷ` (y-hat) is our prediction
- `x` is the input feature
- `w` is the weight (slope)
- `b` is the bias (intercept)

This is the equation of a line—you learned it in algebra as `y = mx + b`. We're simply fitting a line through our data points.

```csharp
// Conceptual implementation - how simple linear regression works
public class SimpleLinearRegression
{
    public double Weight { get; private set; }
    public double Bias { get; private set; }
    
    public void Fit(double[] x, double[] y)
    {
        // Calculate means
        double xMean = x.Average();
        double yMean = y.Average();
        
        // Calculate weight using least squares formula
        double numerator = 0, denominator = 0;
        for (int i = 0; i < x.Length; i++)
        {
            numerator += (x[i] - xMean) * (y[i] - yMean);
            denominator += (x[i] - xMean) * (x[i] - xMean);
        }
        
        Weight = numerator / denominator;
        Bias = yMean - Weight * xMean;
    }
    
    public double Predict(double x) => Weight * x + Bias;
}
```

The formula calculates the slope that minimizes the total squared distance between the line and all data points—the "least squares" solution.

### Multiple Linear Regression

Real predictions require multiple features. Multiple linear regression extends our equation:

```
ŷ = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b
```

Or in vector notation:

```
ŷ = w · x + b
```

Each feature gets its own weight, and we sum their contributions. For our house price example:

```csharp
public class HousePriceModel
{
    // Each coefficient represents dollars per unit of that feature
    public double SquareFootCoef { get; set; }     // $/sqft
    public double BedroomCoef { get; set; }        // $ per bedroom
    public double BathroomCoef { get; set; }       // $ per bathroom
    public double AgeCoef { get; set; }            // $ per year (usually negative)
    public double Intercept { get; set; }          // Base price
    
    public double Predict(House house)
    {
        return SquareFootCoef * house.SquareFeet
             + BedroomCoef * house.Bedrooms
             + BathroomCoef * house.Bathrooms
             + AgeCoef * house.Age
             + Intercept;
    }
}
```

[FIGURE: 3D visualization showing a plane fitted through data points in three dimensions (two features + output). Caption: "With two features, we're fitting a plane. With more features, we fit a hyperplane in n-dimensional space."]

### Interpreting Coefficients

One of regression's greatest strengths is **interpretability**. Each coefficient tells a story:

- **Positive coefficient**: Feature increases the prediction
- **Negative coefficient**: Feature decreases the prediction
- **Magnitude**: How much impact per unit change

If `SquareFootCoef = 150`, adding one square foot adds $150 to the predicted price. If `AgeCoef = -1000`, each year of age reduces the price by $1,000.

But beware: **coefficients are only directly comparable when features are on the same scale**. A coefficient of 0.001 for a feature measured in millions might be more impactful than a coefficient of 100 for a feature measured in units. This is why feature scaling matters, as we'll see shortly.

### Feature Scaling: Making Coefficients Meaningful

Consider two features: square footage (typically 500-5000) and number of bedrooms (typically 1-6). Without scaling, square footage dominates simply because its values are larger. This causes two problems:

1. **Training instability**: Gradient descent takes uneven steps, oscillating on some dimensions
2. **Misleading coefficients**: You can't compare feature importance directly

**Min-Max scaling** transforms features to [0, 1]:

```csharp
public static double[] MinMaxScale(double[] values)
{
    double min = values.Min();
    double max = values.Max();
    double range = max - min;
    
    return values.Select(v => (v - min) / range).ToArray();
}
```

**Z-score standardization** centers features at mean=0, std=1:

```csharp
public static double[] Standardize(double[] values)
{
    double mean = values.Average();
    double std = Math.Sqrt(values.Average(v => Math.Pow(v - mean, 2)));
    
    return values.Select(v => (v - mean) / std).ToArray();
}
```

After standardization, coefficients directly indicate relative importance. A coefficient of 2.0 means "one standard deviation increase in this feature increases the prediction by 2 units."

**When to use which**:
- **Min-Max**: When you need bounded values (e.g., neural networks)
- **Z-score**: When you need to compare coefficient magnitudes or handle outliers better
- **No scaling needed**: Tree-based models (they don't care about scale)

## The Mathematics Behind Learning

Understanding *how* regression learns its coefficients transforms you from a library user into a practitioner who can diagnose problems and optimize solutions.

### The Loss Function

Learning requires a definition of "good." In regression, we typically use **Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

For each training example, we calculate the squared difference between the actual value (`y`) and our prediction (`ŷ`), then average these squared errors.

```csharp
public static double CalculateMSE(double[] actual, double[] predicted)
{
    if (actual.Length != predicted.Length)
        throw new ArgumentException("Arrays must have same length");
    
    double sumSquaredErrors = 0;
    for (int i = 0; i < actual.Length; i++)
    {
        double error = actual[i] - predicted[i];
        sumSquaredErrors += error * error;
    }
    
    return sumSquaredErrors / actual.Length;
}
```

Why squared errors? Three reasons:

1. **Penalizes large errors severely**: A prediction off by 10 contributes 100 to the loss; off by 2 contributes only 4
2. **Differentiable everywhere**: Smooth curves are easier to optimize than absolute values (which have a kink at zero)
3. **Mathematical convenience**: Has a closed-form solution for linear regression

[FIGURE: Parabolic curve showing MSE as a function of a single weight. The minimum of the curve is labeled "Optimal weight." Caption: "The loss function creates a 'bowl'—we want to find the bottom."]

### Gradient Descent: Finding the Minimum

For complex models, we can't solve for optimal weights directly. Instead, we use **gradient descent**—an iterative algorithm that nudges weights toward the minimum loss.

The **gradient** is a vector of partial derivatives—it points "uphill" in the steepest direction. We move opposite to the gradient to go downhill:

```
w_new = w_old - α × ∂Loss/∂w
```

Where `α` (alpha) is the **learning rate**—how big a step we take.

```csharp
public class GradientDescentRegression
{
    private double[] _weights;
    private double _bias;
    private readonly double _learningRate;
    private readonly int _iterations;
    
    public GradientDescentRegression(double learningRate = 0.01, int iterations = 1000)
    {
        _learningRate = learningRate;
        _iterations = iterations;
    }
    
    public void Fit(double[][] X, double[] y)
    {
        int n = X.Length;           // Number of samples
        int features = X[0].Length; // Number of features
        
        // Initialize weights to small random values
        _weights = new double[features];
        _bias = 0;
        
        for (int iter = 0; iter < _iterations; iter++)
        {
            double[] predictions = X.Select(Predict).ToArray();
            
            // Calculate gradients
            double[] weightGradients = new double[features];
            double biasGradient = 0;
            
            for (int i = 0; i < n; i++)
            {
                double error = predictions[i] - y[i];
                
                for (int j = 0; j < features; j++)
                {
                    weightGradients[j] += (2.0 / n) * error * X[i][j];
                }
                biasGradient += (2.0 / n) * error;
            }
            
            // Update weights and bias
            for (int j = 0; j < features; j++)
            {
                _weights[j] -= _learningRate * weightGradients[j];
            }
            _bias -= _learningRate * biasGradient;
        }
    }
    
    public double Predict(double[] x)
    {
        double result = _bias;
        for (int i = 0; i < _weights.Length; i++)
        {
            result += _weights[i] * x[i];
        }
        return result;
    }
}
```

### Learning Rate: The Goldilocks Parameter

The learning rate critically affects training:

- **Too large**: Overshoots the minimum, oscillates wildly or diverges
- **Too small**: Converges painfully slowly
- **Just right**: Steady progress toward the minimum

[FIGURE: Three gradient descent paths on a loss surface. Left shows wild oscillation with large learning rate, middle shows slow creeping with tiny learning rate, right shows smooth convergence with appropriate learning rate.]

In practice, ML.NET handles learning rate selection automatically, but understanding this helps you debug training problems.

### Variants of Gradient Descent

The basic algorithm processes all training examples before updating weights. This is called **batch gradient descent**. For large datasets, this is slow—you might have millions of examples.

**Stochastic Gradient Descent (SGD)** updates weights after each individual example:

```csharp
// Stochastic: update after each sample
for (int epoch = 0; epoch < epochs; epoch++)
{
    Shuffle(trainingData);  // Random order each epoch
    foreach (var sample in trainingData)
    {
        double prediction = Predict(sample.Features);
        double error = prediction - sample.Label;
        
        // Immediate update
        for (int j = 0; j < _weights.Length; j++)
        {
            _weights[j] -= _learningRate * error * sample.Features[j];
        }
        _bias -= _learningRate * error;
    }
}
```

SGD is noisier but converges faster on large datasets. The noise can actually help escape local minima.

**Mini-batch gradient descent** splits the difference—update weights after processing a small batch (typically 32-256 samples). This combines SGD's speed with batch descent's stability. Most modern libraries, including ML.NET's SDCA, use mini-batch approaches.

### The Normal Equation: Solving Directly

For linear regression specifically, we can skip gradient descent entirely. The **normal equation** gives the optimal weights in closed form:

```
w = (XᵀX)⁻¹Xᵀy
```

```csharp
// Using MathNet.Numerics for matrix operations
public double[] SolveDirect(double[,] X, double[] y)
{
    var XMatrix = Matrix<double>.Build.DenseOfArray(X);
    var yVector = Vector<double>.Build.DenseOfArray(y);
    
    // w = (X'X)^(-1) X' y
    var Xt = XMatrix.Transpose();
    var XtX = Xt * XMatrix;
    var XtXInverse = XtX.Inverse();
    var Xty = Xt * yVector;
    
    return (XtXInverse * Xty).ToArray();
}
```

The normal equation is fast for small datasets (< 10,000 features) but becomes impractical as the matrix inversion costs O(n³) time. For large-scale problems, gradient descent variants win.

## Regularization: Preventing Overfitting

A model can learn its training data *too* well, memorizing noise rather than learning true patterns. This is **overfitting**—excellent training performance but poor generalization.

[FIGURE: Three polynomial fits to the same data. Left shows underfitting (straight line through curved data), middle shows good fit (smooth curve following the trend), right shows overfitting (wiggly curve passing through every point).]

Regularization adds a penalty for model complexity, discouraging extreme coefficient values.

### L2 Regularization (Ridge)

L2 regularization adds the sum of squared weights to the loss:

```
Loss = MSE + λ × Σ(wᵢ²)
```

The hyperparameter `λ` (lambda) controls regularization strength. Higher values force smaller weights.

```csharp
public static double RidgeLoss(double[] actual, double[] predicted, 
                                double[] weights, double lambda)
{
    double mse = CalculateMSE(actual, predicted);
    double l2Penalty = weights.Sum(w => w * w);
    return mse + lambda * l2Penalty;
}
```

L2 regularization:
- Shrinks all coefficients toward zero
- Keeps all features (doesn't set coefficients exactly to zero)
- Works well when most features are relevant

### L1 Regularization (Lasso)

L1 regularization uses the sum of absolute values:

```
Loss = MSE + λ × Σ|wᵢ|
```

```csharp
public static double LassoLoss(double[] actual, double[] predicted, 
                                double[] weights, double lambda)
{
    double mse = CalculateMSE(actual, predicted);
    double l1Penalty = weights.Sum(w => Math.Abs(w));
    return mse + lambda * l1Penalty;
}
```

L1 regularization:
- Drives some coefficients to exactly zero
- Performs automatic feature selection
- Creates sparse models (fewer active features)
- Works well when many features are irrelevant

### Elastic Net: Best of Both Worlds

Elastic Net combines L1 and L2:

```
Loss = MSE + λ₁ × Σ|wᵢ| + λ₂ × Σ(wᵢ²)
```

This provides both feature selection (from L1) and stable coefficient estimation (from L2).

### The Geometric Intuition

Why does L1 drive coefficients to exactly zero while L2 only shrinks them?

Imagine the loss function as a bowl-shaped surface. Regularization adds a constraint region—solutions must stay within this region. The optimal point is where the loss surface touches the constraint boundary.

[FIGURE: Two side-by-side plots. Left shows L2 constraint as a circle with an elliptical loss contour touching it—the touch point is on the curved edge, giving non-zero values for both weights. Right shows L1 constraint as a diamond with the same loss contour—the touch point is at a corner where one weight equals zero.]

- **L2 constraint** is a sphere. Loss contours almost always touch the smooth surface, giving non-zero values.
- **L1 constraint** is a diamond with sharp corners. Loss contours often touch at corners, where some coordinates are exactly zero.

This geometric property makes L1 a natural feature selector—it automatically discovers which features don't contribute.

### Choosing Lambda: Cross-Validation

The regularization strength λ is a hyperparameter you must choose. Too small, and you don't prevent overfitting. Too large, and you underfit.

```csharp
public double FindOptimalLambda(IDataView data, IEstimator<ITransformer> basePipeline)
{
    var lambdaValues = new[] { 0.001, 0.01, 0.1, 1.0, 10.0 };
    var bestLambda = 0.0;
    var bestScore = double.MinValue;
    
    foreach (var lambda in lambdaValues)
    {
        var pipeline = basePipeline.Append(
            _mlContext.Regression.Trainers.Sdca(
                l2Regularization: (float)lambda));
        
        var cvResults = _mlContext.Regression.CrossValidate(
            data, pipeline, numberOfFolds: 5);
        
        var avgR2 = cvResults.Average(r => r.Metrics.RSquared);
        
        Console.WriteLine($"λ = {lambda}: R² = {avgR2:P2}");
        
        if (avgR2 > bestScore)
        {
            bestScore = avgR2;
            bestLambda = lambda;
        }
    }
    
    return bestLambda;
}
```

A common approach is to try values on a logarithmic scale (0.001, 0.01, 0.1, 1, 10, 100) and refine around the best performer.

### When to Use Each

| Situation | Recommended Approach |
|-----------|---------------------|
| Many relevant features | L2 (Ridge) |
| Many irrelevant features | L1 (Lasso) |
| Feature groups with correlation | Elastic Net |
| Need feature selection | L1 or Elastic Net |
| Interpretability matters | L1 (fewer features) |

## Regression with ML.NET

ML.NET provides production-ready regression algorithms with sensible defaults. Let's explore the major options.

### Setting Up a Regression Pipeline

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 42);

// Define data schema
public class HouseData
{
    [LoadColumn(0)] public float SquareFeet { get; set; }
    [LoadColumn(1)] public float Bedrooms { get; set; }
    [LoadColumn(2)] public float Bathrooms { get; set; }
    [LoadColumn(3)] public float Age { get; set; }
    [LoadColumn(4)] public float Price { get; set; }  // Label
}

public class HousePrediction
{
    [ColumnName("Score")]
    public float PredictedPrice { get; set; }
}

// Load and split data
var data = mlContext.Data.LoadFromTextFile<HouseData>("houses.csv", hasHeader: true, separatorChar: ',');
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
```

### SDCA (Stochastic Dual Coordinate Ascent)

SDCA is ML.NET's go-to linear regression algorithm. It's fast, memory-efficient, and handles large datasets well:

```csharp
var pipeline = mlContext.Transforms.Concatenate("Features", 
        "SquareFeet", "Bedrooms", "Bathrooms", "Age")
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.Regression.Trainers.Sdca(
        labelColumnName: "Price",
        maximumNumberOfIterations: 100,
        l2Regularization: 0.1f,
        l1Regularization: 0f));  // Pure L2 regularization

var model = pipeline.Fit(split.TrainSet);
```

SDCA works well for:
- Linear relationships
- Large datasets (scales efficiently)
- When interpretability matters (linear coefficients)

### FastTree (Gradient Boosted Trees)

FastTree builds an ensemble of decision trees, capturing non-linear patterns:

```csharp
var fastTreePipeline = mlContext.Transforms.Concatenate("Features", 
        "SquareFeet", "Bedrooms", "Bathrooms", "Age")
    .Append(mlContext.Regression.Trainers.FastTree(
        labelColumnName: "Price",
        numberOfLeaves: 31,
        numberOfTrees: 100,
        minimumExampleCountPerLeaf: 10,
        learningRate: 0.2));

var treeModel = fastTreePipeline.Fit(split.TrainSet);
```

FastTree excels when:
- Relationships are non-linear
- Feature interactions matter (trees naturally capture these)
- You have enough data (trees need more samples than linear models)

### FastForest (Random Forest)

Random forests train multiple independent trees and average their predictions:

```csharp
var forestPipeline = mlContext.Transforms.Concatenate("Features", 
        "SquareFeet", "Bedrooms", "Bathrooms", "Age")
    .Append(mlContext.Regression.Trainers.FastForest(
        labelColumnName: "Price",
        numberOfTrees: 100,
        numberOfLeaves: 31));

var forestModel = forestPipeline.Fit(split.TrainSet);
```

FastForest is:
- More robust to overfitting than single trees
- Good for exploring feature importance
- Less sensitive to hyperparameters

### LightGBM

For serious performance, LightGBM is often the best choice:

```csharp
// Requires Microsoft.ML.LightGbm NuGet package
var lgbmPipeline = mlContext.Transforms.Concatenate("Features", 
        "SquareFeet", "Bedrooms", "Bathrooms", "Age")
    .Append(mlContext.Regression.Trainers.LightGbm(
        labelColumnName: "Price",
        numberOfLeaves: 31,
        numberOfIterations: 100,
        learningRate: 0.1,
        minimumExampleCountPerLeaf: 20));

var lgbmModel = lgbmPipeline.Fit(split.TrainSet);
```

LightGBM advantages:
- Handles categorical features efficiently
- Memory-efficient histogram-based training
- Often wins Kaggle competitions

### Choosing Between Linear and Tree-Based Models

The choice between SDCA (linear) and tree-based models (FastTree, LightGBM) depends on your priorities:

| Factor | Linear Models | Tree-Based Models |
|--------|---------------|-------------------|
| **Interpretability** | Excellent—coefficients tell the story | Poor—complex ensemble logic |
| **Non-linear patterns** | Requires manual feature engineering | Captures automatically |
| **Feature interactions** | Must be explicitly added | Discovered automatically |
| **Training speed** | Very fast | Moderate to slow |
| **Data requirements** | Works with less data | Needs more samples |
| **Overfitting risk** | Lower | Higher (tune carefully) |

**My recommendation**: Start with SDCA for a baseline. If performance is inadequate, try LightGBM. If LightGBM dramatically outperforms SDCA, non-linear patterns or interactions are important—consider adding polynomial features to your linear model if interpretability matters.

```csharp
// Creating polynomial features for linear models
public static void AddPolynomialFeatures(MLContext mlContext)
{
    var pipeline = mlContext.Transforms.CustomMapping<TaxiTrip, PolynomialFeatures>(
        (input, output) =>
        {
            output.DistanceSquared = input.TripDistance * input.TripDistance;
            output.DistanceHour = input.TripDistance * DateTime.Parse(input.PickupDateTime).Hour;
            // Interaction: distance matters more in rush hour
        },
        contractName: "PolynomialFeatures");
}
```

## Model Evaluation: Understanding the Metrics

Evaluating regression requires understanding several metrics, each telling a different story.

### Root Mean Squared Error (RMSE)

```
RMSE = √(MSE) = √[(1/n) × Σ(yᵢ - ŷᵢ)²]
```

RMSE is in the same units as your target variable. If predicting prices in dollars, RMSE is in dollars.

```csharp
var predictions = model.Transform(split.TestSet);
var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Price");

Console.WriteLine($"RMSE: ${metrics.RootMeanSquaredError:N2}");
// Output: RMSE: $45,230.50
```

**Interpretation**: On average, predictions are off by about $45,230. Whether this is good depends on context—acceptable for million-dollar homes, terrible for $100k homes.

### Mean Absolute Error (MAE)

```
MAE = (1/n) × Σ|yᵢ - ŷᵢ|
```

MAE is more intuitive—the average absolute error without squaring.

```csharp
Console.WriteLine($"MAE: ${metrics.MeanAbsoluteError:N2}");
// Output: MAE: $32,100.00
```

**RMSE vs MAE**: RMSE penalizes large errors more severely. If RMSE >> MAE, you have some predictions that are *way* off. If they're similar, errors are fairly consistent.

### R² (Coefficient of Determination)

```
R² = 1 - (Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²)
```

R² measures the proportion of variance explained by the model:

```csharp
Console.WriteLine($"R²: {metrics.RSquared:P2}");
// Output: R²: 87.50%
```

**Interpretation**:
- R² = 1.0: Perfect predictions
- R² = 0.0: Model is no better than predicting the mean
- R² < 0: Model is worse than predicting the mean (something's wrong)
- R² = 0.875: Model explains 87.5% of the variance

### Which Metric to Use?

| Use Case | Preferred Metric |
|----------|-----------------|
| Communicate to stakeholders | MAE (most intuitive) |
| Penalize large errors | RMSE |
| Compare across datasets | R² (scale-independent) |
| Outlier-sensitive domains | RMSE |
| When all errors matter equally | MAE |

Always report multiple metrics—they reveal different aspects of model performance.

### Additional Metrics Worth Knowing

**Mean Absolute Percentage Error (MAPE)** expresses error as a percentage of the actual value:

```csharp
public static double CalculateMAPE(double[] actual, double[] predicted)
{
    double sumPercentageErrors = 0;
    int validCount = 0;
    
    for (int i = 0; i < actual.Length; i++)
    {
        if (Math.Abs(actual[i]) > 0.001)  // Avoid division by zero
        {
            sumPercentageErrors += Math.Abs((actual[i] - predicted[i]) / actual[i]);
            validCount++;
        }
    }
    
    return 100.0 * sumPercentageErrors / validCount;
}
```

MAPE is intuitive—"predictions are off by 8% on average"—but fails when actual values approach zero.

**Adjusted R²** penalizes adding features that don't improve the model:

```
Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - p - 1)]
```

Where `n` is sample count and `p` is feature count. Use this when comparing models with different numbers of features.

### Metrics for Business Decisions

Sometimes statistical metrics miss business reality. Consider:

- **A $5 error on a $10 fare** (50% error) is worse than
- **A $5 error on a $100 fare** (5% error)

But both contribute equally to MAE.

Custom metrics align with your specific use case:

```csharp
public static double CalculateBusinessLoss(IEnumerable<(double Actual, double Predicted)> results)
{
    double totalLoss = 0;
    
    foreach (var (actual, predicted) in results)
    {
        double error = predicted - actual;
        
        if (error > 0)
        {
            // Overestimation: customer sees high price, may not book
            totalLoss += error * 2.0;  // Penalize more heavily
        }
        else
        {
            // Underestimation: we lose revenue but customer is happy
            totalLoss += Math.Abs(error) * 1.0;
        }
    }
    
    return totalLoss / results.Count();
}
```

This asymmetric loss function reflects that overestimating fares costs you customers, while underestimating just reduces margin.

## Detecting Overfitting

Overfitting is regression's constant threat. Here's how to detect and prevent it.

### Training vs. Test Performance Gap

```csharp
var trainMetrics = mlContext.Regression.Evaluate(
    model.Transform(split.TrainSet), labelColumnName: "Price");
var testMetrics = mlContext.Regression.Evaluate(
    model.Transform(split.TestSet), labelColumnName: "Price");

Console.WriteLine($"Train R²: {trainMetrics.RSquared:P2}");
Console.WriteLine($"Test R²:  {testMetrics.RSquared:P2}");

// Good: Train R²: 89.5%, Test R²: 87.2%
// Overfitting: Train R²: 99.8%, Test R²: 72.1%
```

A large gap between training and test performance signals overfitting.

### Cross-Validation

Instead of a single train/test split, cross-validation provides more robust estimates:

```csharp
var cvResults = mlContext.Regression.CrossValidate(
    data, 
    pipeline, 
    numberOfFolds: 5,
    labelColumnName: "Price");

var avgR2 = cvResults.Average(r => r.Metrics.RSquared);
var stdR2 = Math.Sqrt(cvResults.Average(r => 
    Math.Pow(r.Metrics.RSquared - avgR2, 2)));

Console.WriteLine($"R² = {avgR2:P2} ± {stdR2:P2}");
// Output: R² = 85.30% ± 2.10%
```

High variance across folds suggests the model is sensitive to which data it sees—a sign of potential overfitting.

### Residual Analysis

**Residuals** are the differences between actual and predicted values. Plotting them reveals problems:

```csharp
public class ResidualAnalysis
{
    public static void Analyze(IEnumerable<(float Actual, float Predicted)> results)
    {
        var residuals = results.Select(r => r.Actual - r.Predicted).ToList();
        
        // Check for patterns
        Console.WriteLine($"Mean Residual: {residuals.Average():F2}");  // Should be ~0
        Console.WriteLine($"Std Dev: {StandardDeviation(residuals):F2}");
        
        // Check for heteroscedasticity (non-constant variance)
        var lowPredictions = results.Where(r => r.Predicted < results.Average(x => x.Predicted));
        var highPredictions = results.Where(r => r.Predicted >= results.Average(x => x.Predicted));
        
        Console.WriteLine($"Low-range residual std: {StandardDeviation(lowPredictions.Select(r => r.Actual - r.Predicted))}");
        Console.WriteLine($"High-range residual std: {StandardDeviation(highPredictions.Select(r => r.Actual - r.Predicted))}");
    }
}
```

[FIGURE: Four residual plots. Top-left: Good (random scatter around zero). Top-right: Non-linear pattern (curved residuals suggest missing non-linear terms). Bottom-left: Heteroscedasticity (fan shape—variance increases with prediction). Bottom-right: Outliers (most points clustered with a few extreme values).]

**Healthy residuals should**:
- Scatter randomly around zero
- Have constant variance across predictions
- Show no patterns

## Project: NYC Taxi Fare Prediction

Let's build a complete regression system to predict New York City taxi fares. This project demonstrates feature engineering, algorithm comparison, and proper evaluation.

### Understanding the Dataset

The NYC Taxi dataset contains millions of yellow taxi trips. We'll use a representative sample with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `vendor_id` | string | Taxi company identifier |
| `pickup_datetime` | DateTime | When the trip started |
| `passenger_count` | int | Number of passengers |
| `trip_distance` | float | Miles traveled |
| `pickup_longitude` | float | Pickup location longitude |
| `pickup_latitude` | float | Pickup location latitude |
| `dropoff_longitude` | float | Dropoff location longitude |
| `dropoff_latitude` | float | Dropoff location latitude |
| `fare_amount` | float | **Target**: Fare in dollars |

### Exploring the Data

Before building models, explore your data. Real datasets have issues:

```csharp
public class DataExplorer
{
    public static void AnalyzeTaxiData(MLContext mlContext, IDataView data)
    {
        // Convert to enumerable for analysis
        var trips = mlContext.Data.CreateEnumerable<TaxiTrip>(data, reuseRowObject: false).ToList();
        
        Console.WriteLine($"Total trips: {trips.Count:N0}");
        Console.WriteLine();
        
        // Fare distribution
        var fares = trips.Select(t => t.FareAmount).OrderBy(f => f).ToList();
        Console.WriteLine("Fare Distribution:");
        Console.WriteLine($"  Min: ${fares.First():F2}");
        Console.WriteLine($"  25th percentile: ${fares[(int)(fares.Count * 0.25)]:F2}");
        Console.WriteLine($"  Median: ${fares[(int)(fares.Count * 0.5)]:F2}");
        Console.WriteLine($"  75th percentile: ${fares[(int)(fares.Count * 0.75)]:F2}");
        Console.WriteLine($"  Max: ${fares.Last():F2}");
        Console.WriteLine();
        
        // Data quality issues
        var negativeFares = trips.Count(t => t.FareAmount <= 0);
        var extremeFares = trips.Count(t => t.FareAmount > 200);
        var zeroDistance = trips.Count(t => t.TripDistance <= 0);
        var invalidCoords = trips.Count(t => 
            t.PickupLatitude == 0 || t.PickupLongitude == 0);
        
        Console.WriteLine("Data Quality Issues:");
        Console.WriteLine($"  Negative/zero fares: {negativeFares}");
        Console.WriteLine($"  Extreme fares (>$200): {extremeFares}");
        Console.WriteLine($"  Zero distance trips: {zeroDistance}");
        Console.WriteLine($"  Invalid coordinates: {invalidCoords}");
    }
}
```

Typical findings reveal:
- **Negative fares**: Refunds or data errors—remove these
- **Extreme outliers**: $500+ fares are rare and skew the model—consider capping or removing
- **Zero-distance trips**: Driver cancellations or GPS errors—filter out
- **Invalid coordinates**: (0, 0) is in the Atlantic Ocean—something went wrong

Clean your data before training:

```csharp
data = mlContext.Data.FilterRowsByColumn(data, "FareAmount", lowerBound: 2.5, upperBound: 200);
data = mlContext.Data.FilterRowsByColumn(data, "TripDistance", lowerBound: 0.1, upperBound: 50);
data = mlContext.Data.FilterRowsByColumn(data, "PickupLatitude", lowerBound: 40.5, upperBound: 41.0);
data = mlContext.Data.FilterRowsByColumn(data, "PickupLongitude", lowerBound: -74.3, upperBound: -73.7);
```

These bounds keep trips within Manhattan and nearby boroughs, with reasonable fare and distance ranges.

### Data Classes

```csharp
public class TaxiTrip
{
    [LoadColumn(0)] public string VendorId { get; set; }
    [LoadColumn(1)] public string PickupDateTime { get; set; }
    [LoadColumn(2)] public float PassengerCount { get; set; }
    [LoadColumn(3)] public float TripDistance { get; set; }
    [LoadColumn(4)] public float PickupLongitude { get; set; }
    [LoadColumn(5)] public float PickupLatitude { get; set; }
    [LoadColumn(6)] public float DropoffLongitude { get; set; }
    [LoadColumn(7)] public float DropoffLatitude { get; set; }
    [LoadColumn(8)] public float FareAmount { get; set; }
}

public class TaxiFarePrediction
{
    [ColumnName("Score")]
    public float FareAmount { get; set; }
}
```

### Feature Engineering

Raw data rarely predicts well. Feature engineering transforms it into signals the model can learn from:

```csharp
public class TaxiFeatureEngineering
{
    public static IEstimator<ITransformer> CreateFeaturePipeline(MLContext mlContext)
    {
        return mlContext.Transforms
            // Parse datetime and extract components
            .CustomMapping<TaxiTrip, EnrichedTaxiTrip>(
                (input, output) =>
                {
                    var dt = DateTime.Parse(input.PickupDateTime);
                    output.Hour = dt.Hour;
                    output.DayOfWeek = (int)dt.DayOfWeek;
                    output.Month = dt.Month;
                    output.IsWeekend = dt.DayOfWeek == DayOfWeek.Saturday || 
                                       dt.DayOfWeek == DayOfWeek.Sunday ? 1f : 0f;
                    output.IsRushHour = (dt.Hour >= 7 && dt.Hour <= 9) || 
                                        (dt.Hour >= 16 && dt.Hour <= 19) ? 1f : 0f;
                    output.IsNight = dt.Hour >= 20 || dt.Hour <= 5 ? 1f : 0f;
                    
                    // Copy through other features
                    output.PassengerCount = input.PassengerCount;
                    output.TripDistance = input.TripDistance;
                    output.PickupLongitude = input.PickupLongitude;
                    output.PickupLatitude = input.PickupLatitude;
                    output.DropoffLongitude = input.DropoffLongitude;
                    output.DropoffLatitude = input.DropoffLatitude;
                    
                    // Calculate direct distance (Haversine approximation for short distances)
                    double latDiff = input.DropoffLatitude - input.PickupLatitude;
                    double lonDiff = input.DropoffLongitude - input.PickupLongitude;
                    output.DirectDistance = (float)Math.Sqrt(latDiff * latDiff + lonDiff * lonDiff) * 69f; // Rough miles
                    
                    // Detect airport trips (JFK approximate coordinates)
                    bool isJFKPickup = input.PickupLatitude > 40.63 && input.PickupLatitude < 40.66 &&
                                       input.PickupLongitude > -73.82 && input.PickupLongitude < -73.76;
                    bool isJFKDropoff = input.DropoffLatitude > 40.63 && input.DropoffLatitude < 40.66 &&
                                        input.DropoffLongitude > -73.82 && input.DropoffLongitude < -73.76;
                    output.IsAirportTrip = isJFKPickup || isJFKDropoff ? 1f : 0f;
                },
                contractName: "TaxiFeatureEngineering")
            
            // Encode categorical features
            .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorEncoded", "VendorId"))
            
            // Combine all features
            .Append(mlContext.Transforms.Concatenate("Features",
                "VendorEncoded", "PassengerCount", "TripDistance", "DirectDistance",
                "Hour", "DayOfWeek", "Month", "IsWeekend", "IsRushHour", "IsNight", "IsAirportTrip",
                "PickupLongitude", "PickupLatitude", "DropoffLongitude", "DropoffLatitude"))
            
            // Normalize features for linear models
            .Append(mlContext.Transforms.NormalizeMinMax("Features"));
    }
}

public class EnrichedTaxiTrip
{
    public float PassengerCount { get; set; }
    public float TripDistance { get; set; }
    public float PickupLongitude { get; set; }
    public float PickupLatitude { get; set; }
    public float DropoffLongitude { get; set; }
    public float DropoffLatitude { get; set; }
    public float Hour { get; set; }
    public int DayOfWeek { get; set; }
    public int Month { get; set; }
    public float IsWeekend { get; set; }
    public float IsRushHour { get; set; }
    public float IsNight { get; set; }
    public float DirectDistance { get; set; }
    public float IsAirportTrip { get; set; }
}
```

**Key engineered features**:

1. **Time decomposition**: Hour, day of week, month—each captures different demand patterns
2. **Rush hour indicator**: Peak times command higher fares due to traffic (longer trips)
3. **Night indicator**: Late-night surcharges and different traffic patterns
4. **Direct distance**: Straight-line miles between points (supplements actual trip distance)
5. **Airport detection**: JFK trips have flat rates (~$52 to Manhattan)

### Why Feature Engineering Beats Raw Data

Consider predicting a 6 AM Tuesday trip versus a 6 PM Friday trip. The raw hour (6 vs. 18) and day (2 vs. 5) don't capture the behavioral difference. But engineered features like `IsRushHour` and `IsWeekend` encode domain knowledge directly.

Some additional features worth considering:

```csharp
// Weather integration (if you have weather data)
output.IsRaining = weatherService.IsRaining(input.PickupDateTime);
output.Temperature = weatherService.GetTemperature(input.PickupDateTime);

// Geographic features
output.IsManhattan = input.PickupLatitude > 40.7 && input.PickupLatitude < 40.82 &&
                     input.PickupLongitude > -74.02 && input.PickupLongitude < -73.93;

// Trip direction (uptown vs downtown)
output.IsNorthbound = input.DropoffLatitude > input.PickupLatitude ? 1f : 0f;

// Distance ratio (actual vs direct - indicates route complexity)
if (output.DirectDistance > 0.1f)
{
    output.DistanceRatio = input.TripDistance / output.DirectDistance;
}
else
{
    output.DistanceRatio = 1f;
}
```

The distance ratio is particularly clever: a ratio near 1.0 means a direct route, while a high ratio indicates traffic detours or a scenic driver. High ratios often correlate with higher fares.

### Building and Comparing Models

```csharp
public class TaxiFarePredictor
{
    private readonly MLContext _mlContext;
    
    public TaxiFarePredictor()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public void TrainAndEvaluate(string dataPath)
    {
        // Load data
        var data = _mlContext.Data.LoadFromTextFile<TaxiTrip>(
            dataPath, hasHeader: true, separatorChar: ',');
        
        // Filter invalid data
        data = _mlContext.Data.FilterRowsByColumn(data, "FareAmount", lowerBound: 1, upperBound: 500);
        data = _mlContext.Data.FilterRowsByColumn(data, "TripDistance", lowerBound: 0.1, upperBound: 100);
        
        var split = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        
        // Build feature pipeline
        var featurePipeline = TaxiFeatureEngineering.CreateFeaturePipeline(_mlContext);
        
        // Define trainers to compare
        var trainers = new Dictionary<string, IEstimator<ITransformer>>
        {
            ["SDCA"] = _mlContext.Regression.Trainers.Sdca(
                labelColumnName: "FareAmount",
                l2Regularization: 0.1f),
                
            ["FastTree"] = _mlContext.Regression.Trainers.FastTree(
                labelColumnName: "FareAmount",
                numberOfLeaves: 31,
                numberOfTrees: 100,
                learningRate: 0.2),
                
            ["FastForest"] = _mlContext.Regression.Trainers.FastForest(
                labelColumnName: "FareAmount",
                numberOfTrees: 100),
                
            ["LightGBM"] = _mlContext.Regression.Trainers.LightGbm(
                labelColumnName: "FareAmount",
                numberOfLeaves: 31,
                numberOfIterations: 100)
        };
        
        Console.WriteLine("Algorithm Comparison");
        Console.WriteLine(new string('=', 60));
        Console.WriteLine($"{"Algorithm",-15} {"RMSE",10} {"MAE",10} {"R²",10}");
        Console.WriteLine(new string('-', 60));
        
        RegressionMetrics bestMetrics = null;
        string bestAlgorithm = null;
        ITransformer bestModel = null;
        
        foreach (var (name, trainer) in trainers)
        {
            var pipeline = featurePipeline.Append(trainer);
            var model = pipeline.Fit(split.TrainSet);
            var predictions = model.Transform(split.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "FareAmount");
            
            Console.WriteLine($"{name,-15} {metrics.RootMeanSquaredError,10:F2} " +
                            $"{metrics.MeanAbsoluteError,10:F2} {metrics.RSquared,10:P1}");
            
            if (bestMetrics == null || metrics.RSquared > bestMetrics.RSquared)
            {
                bestMetrics = metrics;
                bestAlgorithm = name;
                bestModel = model;
            }
        }
        
        Console.WriteLine(new string('=', 60));
        Console.WriteLine($"Best: {bestAlgorithm} with R² = {bestMetrics.RSquared:P2}");
        
        // Perform residual analysis on best model
        AnalyzeResiduals(bestModel, split.TestSet);
        
        // Save best model
        _mlContext.Model.Save(bestModel, data.Schema, "taxi_fare_model.zip");
    }
    
    private void AnalyzeResiduals(ITransformer model, IDataView testData)
    {
        Console.WriteLine("\nResidual Analysis");
        Console.WriteLine(new string('=', 60));
        
        var predictions = model.Transform(testData);
        var predictionData = _mlContext.Data.CreateEnumerable<TaxiPredictionWithActual>(
            predictions, reuseRowObject: false).ToList();
        
        var residuals = predictionData.Select(p => p.FareAmount - p.Score).ToList();
        
        Console.WriteLine($"Mean Residual: ${residuals.Average():F2}");
        Console.WriteLine($"Residual Std Dev: ${StandardDeviation(residuals):F2}");
        
        // Check by fare ranges
        var lowFares = predictionData.Where(p => p.FareAmount < 10);
        var midFares = predictionData.Where(p => p.FareAmount >= 10 && p.FareAmount < 30);
        var highFares = predictionData.Where(p => p.FareAmount >= 30);
        
        Console.WriteLine($"\nMAE by fare range:");
        Console.WriteLine($"  Low (<$10):   ${lowFares.Average(p => Math.Abs(p.FareAmount - p.Score)):F2}");
        Console.WriteLine($"  Mid ($10-30): ${midFares.Average(p => Math.Abs(p.FareAmount - p.Score)):F2}");
        Console.WriteLine($"  High (>$30):  ${highFares.Average(p => Math.Abs(p.FareAmount - p.Score)):F2}");
        
        // Check for outliers (residuals > 3 std dev)
        double stdDev = StandardDeviation(residuals);
        var outliers = residuals.Count(r => Math.Abs(r) > 3 * stdDev);
        Console.WriteLine($"\nOutlier predictions (>3σ): {outliers} ({100.0 * outliers / residuals.Count:F2}%)");
    }
    
    private static double StandardDeviation(IEnumerable<double> values)
    {
        var list = values.ToList();
        double mean = list.Average();
        double sumSquares = list.Sum(v => Math.Pow(v - mean, 2));
        return Math.Sqrt(sumSquares / list.Count);
    }
}

public class TaxiPredictionWithActual
{
    public float FareAmount { get; set; }
    public float Score { get; set; }
}
```

### Sample Output

```
Algorithm Comparison
============================================================
Algorithm             RMSE        MAE         R²
------------------------------------------------------------
SDCA                  4.82       3.21      81.2%
FastTree              3.94       2.67      87.4%
FastForest            4.12       2.81      86.2%
LightGBM              3.87       2.58      88.1%
============================================================
Best: LightGBM with R² = 88.10%

Residual Analysis
============================================================
Mean Residual: $0.08
Residual Std Dev: $3.87

MAE by fare range:
  Low (<$10):   $1.45
  Mid ($10-30): $2.82
  High (>$30):  $5.21

Outlier predictions (>3σ): 847 (1.69%)
```

### Making Predictions

```csharp
public class TaxiFarePredictionService
{
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<TaxiTrip, TaxiFarePrediction> _predictionEngine;
    
    public TaxiFarePredictionService(string modelPath)
    {
        _mlContext = new MLContext();
        var model = _mlContext.Model.Load(modelPath, out _);
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiFarePrediction>(model);
    }
    
    public decimal PredictFare(TaxiTrip trip)
    {
        var prediction = _predictionEngine.Predict(trip);
        return Math.Max(2.50m, Math.Round((decimal)prediction.FareAmount, 2));
    }
}

// Usage
var service = new TaxiFarePredictionService("taxi_fare_model.zip");

var trip = new TaxiTrip
{
    VendorId = "VTS",
    PickupDateTime = "2024-03-15 18:30:00",  // Friday rush hour
    PassengerCount = 2,
    TripDistance = 3.5f,
    PickupLongitude = -73.99f,
    PickupLatitude = 40.73f,
    DropoffLongitude = -73.97f,
    DropoffLatitude = 40.76f
};

var fare = service.PredictFare(trip);
Console.WriteLine($"Estimated fare: ${fare}");
// Output: Estimated fare: $14.50
```

### Insights from Our Model

After training, we can extract feature importance from tree-based models:

```csharp
var treeModel = model.LastTransformer as FastTreeRegressionModelParameters;
if (treeModel != null)
{
    var importance = treeModel.FeatureContribution(/* feature vector */);
    // Top features: TripDistance, DirectDistance, IsAirportTrip, Hour, IsRushHour
}
```

Key findings from taxi fare prediction:

1. **Distance dominates**: Trip distance explains most variance—no surprise
2. **Time matters**: Rush hour and late-night trips command 10-15% higher fares
3. **Airport trips are special**: JFK trips have nearly flat rates, reducing prediction error
4. **Passenger count barely matters**: Taxis don't charge per person
5. **Location helps at the margins**: Pickup/dropoff coordinates help with neighborhood-specific patterns

### Troubleshooting Common Issues

**Problem: Model predicts negative fares**

The model extrapolated beyond training data. Fix with post-processing:

```csharp
public decimal PredictFare(TaxiTrip trip)
{
    var prediction = _predictionEngine.Predict(trip);
    // Enforce minimum fare ($2.50 base fare in NYC)
    return Math.Max(2.50m, Math.Round((decimal)prediction.FareAmount, 2));
}
```

**Problem: High variance across cross-validation folds**

Your model is sensitive to the specific training data, indicating overfitting. Try:
- Increase regularization
- Reduce model complexity (fewer trees, fewer leaves)
- Add more training data if available

**Problem: Predictions cluster around the mean**

Your model is underfitting—not capturing the signal. Try:
- Add more features (feature engineering)
- Reduce regularization
- Use a more powerful model (switch from SDCA to FastTree)

**Problem: Systematic bias in residuals**

If residuals show a pattern (e.g., always underpredict high fares), your model is missing something:
- Check for missing features
- Consider non-linear transformations (log of fare, polynomial features)
- Try a more flexible model

### From Model to Production

A trained model is just the beginning. Production considerations include:

```csharp
public class TaxiFareService
{
    private readonly PredictionEngine<TaxiTrip, TaxiFarePrediction> _engine;
    private readonly ILogger<TaxiFareService> _logger;
    
    public FareEstimate GetFareEstimate(TaxiTripRequest request)
    {
        var trip = MapToTaxiTrip(request);
        
        // Validate inputs
        if (!IsValidCoordinate(trip.PickupLatitude, trip.PickupLongitude) ||
            !IsValidCoordinate(trip.DropoffLatitude, trip.DropoffLongitude))
        {
            _logger.LogWarning("Invalid coordinates in fare request");
            return FareEstimate.Unknown();
        }
        
        var prediction = _engine.Predict(trip);
        
        // Apply business rules
        var baseFare = Math.Max(2.50f, prediction.FareAmount);
        var lowEstimate = baseFare * 0.85f;
        var highEstimate = baseFare * 1.15f;
        
        return new FareEstimate
        {
            BaseFare = baseFare,
            LowEstimate = lowEstimate,
            HighEstimate = highEstimate,
            Currency = "USD",
            Confidence = CalculateConfidence(trip)
        };
    }
    
    private float CalculateConfidence(TaxiTrip trip)
    {
        // Lower confidence for unusual trips
        if (trip.TripDistance > 20) return 0.7f;  // Long trips are harder to predict
        if (trip.PassengerCount > 4) return 0.8f;  // Less common
        return 0.9f;
    }
}
```

Key production concerns:
- **Input validation**: Don't trust user input
- **Confidence intervals**: Give ranges, not point estimates
- **Monitoring**: Track prediction distribution over time to detect drift
- **Fallbacks**: Have a rule-based backup if the model fails

## Summary

Regression transforms data into numerical predictions through learned relationships. In this chapter, you've learned:

- **Linear regression** fits lines (or hyperplanes) to data, with coefficients that tell you exactly how each feature contributes
- **Gradient descent** iteratively finds optimal weights by following the loss landscape downhill
- **Regularization** (L1/L2) prevents overfitting by penalizing complexity—L1 for feature selection, L2 for stable estimates
- **ML.NET provides** production-ready algorithms: SDCA for linear models, FastTree and LightGBM for non-linear power
- **Evaluation requires multiple metrics**: RMSE penalizes outliers, MAE is intuitive, R² is scale-independent
- **Residual analysis** reveals whether your model has systematic problems

The taxi fare project demonstrated end-to-end regression: feature engineering that captures domain knowledge (rush hour, airports), algorithm comparison that finds the best fit, and residual analysis that validates our model behaves sensibly.

In the next chapter, we'll tackle a different challenge: when your data has structure that standard algorithms ignore. We'll explore how to handle time series data, where the order of observations carries critical information.

---

**Key Takeaways**

- Regression predicts continuous values; coefficients tell you the impact of each feature
- MSE is the standard loss; squaring errors penalizes large mistakes heavily
- Regularization is not optional—overfitting is the default failure mode
- Tree-based methods (FastTree, LightGBM) usually outperform linear models on complex data
- Always check residuals: random scatter is healthy, patterns mean missed opportunities
- Feature engineering often matters more than algorithm choice
