# Chapter 7: Introduction to ML.NET

Machine learning has transformed from an academic curiosity into an essential tool for modern software development. From recommendation engines to fraud detection, from image recognition to predictive maintenance, ML powers countless applications we interact with daily. For C# developers, **ML.NET** represents Microsoft's answer to a critical question: how can .NET developers leverage machine learning without abandoning their language, tools, and ecosystem?

In this chapter, we'll explore ML.NET from the ground up. You'll learn its core architecture, understand why it was designed the way it was, and build your first complete machine learning model. By the end, you'll have the foundation to tackle real-world ML problems using familiar C# patterns and practices.

## Why ML.NET?

Before diving into code, let's address the elephant in the room: why use ML.NET when Python dominates the machine learning landscape?

The answer lies in **integration and deployment**. While Python excels at research and experimentation, production systems often run on .NET. ML.NET lets you:

- **Stay in one ecosystem**: No need to maintain separate Python services or manage cross-language deployments
- **Leverage existing skills**: Use familiar C# patterns, debugging tools, and IDE features
- **Deploy anywhere .NET runs**: From Azure to on-premises servers to mobile devices via .NET MAUI
- **Achieve production performance**: ML.NET is optimized for high-throughput inference in production scenarios

ML.NET isn't trying to replace Python for research—it's designed for .NET developers who need to integrate machine learning into production applications.

## ML.NET Architecture Overview

ML.NET's architecture reflects lessons learned from decades of machine learning development at Microsoft. Let's examine its core components.

[DIAGRAM: ML.NET Architecture Overview
- Top layer: "Your Application Code (C#)"
- Middle layer: "ML.NET API" containing boxes for "MLContext", "IDataView", "Transforms", "Trainers"
- Bottom layer: "Native Libraries" showing "Intel MKL", "ONNX Runtime", "TensorFlow", "LightGBM"
- Arrows showing data flow from application through API to native libraries]

### MLContext: The Entry Point

Every ML.NET application begins with `MLContext`. Think of it as your machine learning session manager—similar to how `DbContext` works in Entity Framework. The `MLContext` provides access to all ML.NET operations and maintains the state needed for reproducible experiments.

```csharp
using Microsoft.ML;

// Create the ML.NET context
var mlContext = new MLContext(seed: 42);
```

The optional `seed` parameter ensures reproducibility. Machine learning algorithms often involve random initialization, and setting a seed guarantees you'll get the same results across runs—essential for debugging and experimentation.

`MLContext` exposes several important properties:

- **Data**: Methods for loading and creating data
- **Transforms**: Feature engineering and data preprocessing
- **BinaryClassification, MulticlassClassification, Regression, etc.**: Task-specific trainers and evaluators
- **Model**: Operations for saving and loading trained models

### IDataView: The Data Abstraction

`IDataView` is ML.NET's core data abstraction, and understanding it is crucial for effective ML.NET development. Unlike `DataTable` or `List<T>`, `IDataView` was designed from the ground up for machine learning workloads.

[DIAGRAM: IDataView Characteristics
- Box labeled "IDataView" with four extending arrows pointing to:
  - "Lazy Evaluation" - "Data processed only when needed"
  - "Immutable" - "Transforms create new views, never modify"
  - "Streaming" - "Can process data larger than memory"
  - "Columnar" - "Optimized for column-wise operations"]

**Why not just use `List<T>`?**

Consider a dataset with millions of rows. Loading it entirely into memory as objects would be wasteful—machine learning algorithms typically process data column by column, not row by row. `IDataView` provides:

1. **Lazy evaluation**: Data is processed only when needed, enabling efficient pipelines
2. **Columnar storage**: Operations on single columns are optimized
3. **Streaming capability**: Process datasets larger than available memory
4. **Immutability**: Every transformation creates a new view, maintaining data lineage

Here's how you typically create an `IDataView`:

```csharp
// Define your data class
public class HouseData
{
    public float Size { get; set; }
    public float Bedrooms { get; set; }
    public float Price { get; set; }
}

// Load from an in-memory collection
var houses = new List<HouseData>
{
    new() { Size = 1500, Bedrooms = 3, Price = 250000 },
    new() { Size = 2000, Bedrooms = 4, Price = 320000 },
    // ... more data
};

IDataView dataView = mlContext.Data.LoadFromEnumerable(houses);

// Or load directly from a file
IDataView csvData = mlContext.Data.LoadFromTextFile<HouseData>(
    path: "houses.csv",
    hasHeader: true,
    separatorChar: ',');
```

### Schema and Column Types

Every `IDataView` has a schema describing its columns. ML.NET supports various data types, but the most common for machine learning are:

| C# Type | ML.NET Type | Usage |
|---------|-------------|-------|
| `float` | R4 (Single) | Features, regression targets |
| `bool` | Boolean | Binary labels |
| `string` | Text | Categorical features, text data |
| `float[]` | Vector<R4> | Multi-dimensional features |

You can inspect a schema programmatically:

```csharp
foreach (var column in dataView.Schema)
{
    Console.WriteLine($"{column.Name}: {column.Type}");
}
```

## The Pipeline Pattern

ML.NET's most distinctive feature is its **pipeline pattern**. A pipeline chains together data transformations and a training algorithm into a single, reproducible workflow.

[DIAGRAM: ML.NET Pipeline Flow
Linear flow diagram showing:
"Raw Data" → [Transform 1: Normalize] → [Transform 2: Encode Categories] → [Transform 3: Concatenate Features] → [Trainer: Algorithm] → "Trained Model"
With annotation: "Estimator Chain (before fitting)" above the transforms
And annotation: "Transformer Chain (after fitting)" below]

### Estimators and Transformers

Understanding the distinction between **Estimators** and **Transformers** is key to mastering ML.NET:

- **Estimator** (`IEstimator<T>`): An operation that *learns* from data. It has a `Fit()` method that examines data and produces a Transformer. Example: a normalizer needs to learn the min/max values from your data.

- **Transformer** (`ITransformer`): An operation that *applies* a learned transformation. It has a `Transform()` method that processes data. Example: applying the learned normalization to new data.

This separation enables the critical distinction between training and inference:

```csharp
// Define an estimator pipeline
var pipeline = mlContext.Transforms.NormalizeMinMax("Size")
    .Append(mlContext.Transforms.NormalizeMinMax("Bedrooms"))
    .Append(mlContext.Transforms.Concatenate("Features", "Size", "Bedrooms"))
    .Append(mlContext.Regression.Trainers.Sdca(
        labelColumnName: "Price",
        featureColumnName: "Features"));

// Fit the pipeline to training data - this produces a transformer
ITransformer model = pipeline.Fit(trainingData);

// The model (transformer) can now transform new data
IDataView predictions = model.Transform(testData);
```

### Common Transforms

ML.NET provides a rich library of transforms for feature engineering:

**Normalization** scales numeric features to comparable ranges:

```csharp
// Min-Max normalization (scales to 0-1)
mlContext.Transforms.NormalizeMinMax("Price")

// Mean-variance normalization (zero mean, unit variance)
mlContext.Transforms.NormalizeMeanVariance("Price")

// Log normalization (useful for skewed distributions)
mlContext.Transforms.NormalizeLogMeanVariance("Price")
```

**Categorical Encoding** converts text categories to numbers:

```csharp
// One-hot encoding: "Red", "Blue", "Green" → [1,0,0], [0,1,0], [0,0,1]
mlContext.Transforms.Categorical.OneHotEncoding("Color")

// Hash encoding: maps to fixed-size bucket (good for high cardinality)
mlContext.Transforms.Categorical.OneHotHashEncoding("ProductId", numberOfBits: 10)
```

**Text Processing** handles string features:

```csharp
// Featurize text into numeric vectors
mlContext.Transforms.Text.FeaturizeText(
    outputColumnName: "TextFeatures",
    inputColumnName: "Description")
```

**Feature Concatenation** combines multiple columns into one feature vector:

```csharp
// ML algorithms expect a single feature column
mlContext.Transforms.Concatenate("Features", 
    "Size", "Bedrooms", "Bathrooms", "YearBuilt")
```

### Building a Complete Pipeline

Let's build a realistic pipeline for house price prediction:

```csharp
public class HouseData
{
    [LoadColumn(0)] public float Size { get; set; }
    [LoadColumn(1)] public float Bedrooms { get; set; }
    [LoadColumn(2)] public float Bathrooms { get; set; }
    [LoadColumn(3)] public string Neighborhood { get; set; }
    [LoadColumn(4)] public float YearBuilt { get; set; }
    [LoadColumn(5)] public float Price { get; set; }
}

var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood")
    .Append(mlContext.Transforms.NormalizeMinMax("Size"))
    .Append(mlContext.Transforms.NormalizeMinMax("Bedrooms"))
    .Append(mlContext.Transforms.NormalizeMinMax("Bathrooms"))
    .Append(mlContext.Transforms.NormalizeMinMax("YearBuilt"))
    .Append(mlContext.Transforms.Concatenate("Features",
        "Size", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt"))
    .Append(mlContext.Regression.Trainers.FastTree(
        labelColumnName: "Price",
        featureColumnName: "Features",
        numberOfLeaves: 20,
        minimumExampleCountPerLeaf: 10,
        learningRate: 0.2));
```

Notice how the pipeline reads almost like a recipe: encode categories, normalize numeric features, combine everything, then train. This declarative style makes ML.NET pipelines readable and maintainable.

## Training vs. Inference Workflows

Understanding the separation between training and inference is fundamental to deploying machine learning in production.

[DIAGRAM: Training vs Inference Workflows
Two parallel flows:

TRAINING (Development/Batch):
"Historical Data" → "Pipeline.Fit()" → "Trained Model (.zip)" → "Save to Disk"

INFERENCE (Production/Real-time):
"Load Model" → "New Data" → "Model.Transform()" → "Predictions"]

### The Training Workflow

Training happens during development or as a scheduled batch process:

```csharp
public class ModelTrainer
{
    public void TrainAndSaveModel(string dataPath, string modelPath)
    {
        var mlContext = new MLContext(seed: 42);
        
        // 1. Load data
        var data = mlContext.Data.LoadFromTextFile<HouseData>(
            dataPath, hasHeader: true, separatorChar: ',');
        
        // 2. Split into training and test sets
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        
        // 3. Define the pipeline
        var pipeline = BuildPipeline(mlContext);
        
        // 4. Train the model
        Console.WriteLine("Training model...");
        var model = pipeline.Fit(split.TrainSet);
        
        // 5. Evaluate on test set
        var predictions = model.Transform(split.TestSet);
        var metrics = mlContext.Regression.Evaluate(predictions, "Price");
        
        Console.WriteLine($"R² Score: {metrics.RSquared:F4}");
        Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F2}");
        
        // 6. Save the model
        mlContext.Model.Save(model, data.Schema, modelPath);
        Console.WriteLine($"Model saved to {modelPath}");
    }
}
```

### The Inference Workflow

Inference happens in production, often in a web API or background service:

```csharp
public class PredictionService
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    private readonly PredictionEngine<HouseData, HousePrediction> _predictionEngine;
    
    public PredictionService(string modelPath)
    {
        _mlContext = new MLContext();
        
        // Load the trained model
        _model = _mlContext.Model.Load(modelPath, out var modelSchema);
        
        // Create a prediction engine for single predictions
        _predictionEngine = _mlContext.Model
            .CreatePredictionEngine<HouseData, HousePrediction>(_model);
    }
    
    public float PredictPrice(HouseData house)
    {
        var prediction = _predictionEngine.Predict(house);
        return prediction.Price;
    }
}

public class HousePrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}
```

### PredictionEngine vs. Transform

ML.NET offers two ways to make predictions:

**`PredictionEngine`** is optimized for single predictions:
- Creates internal object pools for efficiency
- Thread-safe when properly configured
- Ideal for web APIs handling one request at a time

```csharp
var engine = mlContext.Model.CreatePredictionEngine<Input, Output>(model);
var prediction = engine.Predict(singleInput);
```

**`Transform`** is better for batch predictions:
- Processes entire `IDataView` at once
- More efficient for large batches
- Maintains the streaming/lazy evaluation benefits

```csharp
var predictions = model.Transform(batchDataView);
var results = mlContext.Data.CreateEnumerable<Output>(predictions, false);
```

> **Production Tip**: `PredictionEngine` is not thread-safe by default. In ASP.NET Core, use `PredictionEnginePool` from the `Microsoft.Extensions.ML` package for proper dependency injection and thread safety.

## Model Evaluation

A model is only useful if it generalizes to new data. ML.NET provides comprehensive evaluation tools to measure model performance.

### Train/Test Split

The simplest evaluation approach splits your data into training and test sets:

```csharp
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

// Train on 80% of data
var model = pipeline.Fit(split.TrainSet);

// Evaluate on held-out 20%
var predictions = model.Transform(split.TestSet);
```

This approach has a limitation: your evaluation depends on which random 20% ended up in the test set. For small datasets, this variance can be significant.

### Cross-Validation

Cross-validation provides a more robust evaluation by training and testing multiple times:

[DIAGRAM: 5-Fold Cross-Validation
Five horizontal bars representing the full dataset, each divided into 5 segments.
In each bar, one segment is shaded (test) while others are unshaded (train):
- Fold 1: [TEST][train][train][train][train]
- Fold 2: [train][TEST][train][train][train]
- Fold 3: [train][train][TEST][train][train]
- Fold 4: [train][train][train][TEST][train]
- Fold 5: [train][train][train][train][TEST]
Arrow pointing to "Average metrics across all folds"]

```csharp
// Perform 5-fold cross-validation
var cvResults = mlContext.Regression.CrossValidate(
    data, 
    pipeline, 
    numberOfFolds: 5,
    labelColumnName: "Price");

// Aggregate results
var avgR2 = cvResults.Average(r => r.Metrics.RSquared);
var stdR2 = Math.Sqrt(cvResults.Average(r => 
    Math.Pow(r.Metrics.RSquared - avgR2, 2)));

Console.WriteLine($"Average R²: {avgR2:F4} (±{stdR2:F4})");
```

Cross-validation is more computationally expensive but gives you confidence intervals on your metrics.

### Evaluation Metrics by Task

Different ML tasks use different metrics:

**Regression** (predicting continuous values):
```csharp
var metrics = mlContext.Regression.Evaluate(predictions, "Price");
Console.WriteLine($"R² Score: {metrics.RSquared:F4}");           // 0-1, higher is better
Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:F2}");       // Average error magnitude
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F2}");   // Penalizes large errors
```

**Binary Classification** (yes/no predictions):
```csharp
var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");
Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");       // 0.5-1, higher is better
Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");            // Balance of precision/recall
```

**Multiclass Classification** (multiple categories):
```csharp
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label");
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:P2}");
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:P2}");
Console.WriteLine($"Log Loss: {metrics.LogLoss:F4}");            // Lower is better
```

### Confusion Matrix

For classification tasks, the confusion matrix reveals where your model makes mistakes:

```csharp
var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Species");
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
```

Output:
```
Confusion table
          ||========================
PREDICTED ||  setosa | versicolor | virginica
ACTUAL    ||========================
setosa    ||      10 |          0 |         0
versicolor||       0 |          9 |         1
virginica ||       0 |          2 |         8
          ||========================
```

This shows that the model occasionally confuses versicolor with virginica—valuable insight for improving the model.

## AutoML with ML.NET Model Builder

While understanding the fundamentals is important, ML.NET also offers AutoML capabilities that can dramatically accelerate development. **Model Builder** is a Visual Studio extension that provides a guided, visual experience for creating ML models.

### Installing Model Builder

Model Builder comes included with Visual Studio 2022. To verify it's installed:

1. Open Visual Studio
2. Go to **Extensions → Manage Extensions**
3. Search for "ML.NET Model Builder"
4. Install if not present (requires VS restart)

### Using Model Builder

Model Builder walks you through the complete ML workflow:

[DIAGRAM: Model Builder Workflow
Horizontal flow with 5 connected boxes:
"1. Choose Scenario" → "2. Select Data" → "3. Train Model" → "4. Evaluate Results" → "5. Generate Code"
With icons: lightbulb, database, gear, chart, code brackets]

**Step 1: Add Machine Learning**
Right-click your project → **Add → Machine Learning Model**. This launches the Model Builder wizard.

**Step 2: Choose a Scenario**
Model Builder offers pre-configured scenarios:
- **Value prediction** (regression)
- **Data classification** (binary/multiclass)
- **Image classification**
- **Object detection**
- **Text classification**
- **Recommendation**

**Step 3: Select Training Environment**
- **Local (CPU)**: Good for small datasets and quick experimentation
- **Local (GPU)**: Faster training if you have a compatible NVIDIA GPU
- **Azure**: Scale to larger datasets using Azure ML

**Step 4: Load and Configure Data**
Point Model Builder to your CSV or database. It automatically detects column types and suggests the prediction target.

**Step 5: Train**
Model Builder's AutoML engine tries multiple algorithms and hyperparameters, showing real-time progress. Training time is configurable—longer training explores more options.

**Step 6: Evaluate and Consume**
After training, Model Builder shows metrics and generates production-ready C# code including:
- Data model classes
- A training project (to retrain with new data)
- A consumption project (for integration)

### When to Use Model Builder vs. Code-First

| Scenario | Recommendation |
|----------|----------------|
| Learning ML.NET | Start with code-first for understanding |
| Rapid prototyping | Model Builder saves significant time |
| Custom preprocessing | Code-first for complex feature engineering |
| Production systems | Either; Model Builder generates production code |
| Experimentation | Model Builder's AutoML explores more options |

Model Builder generates standard ML.NET code, so you can always start with Model Builder and customize the generated code later.

## Project: Your First ML.NET Classification Model

Let's put everything together by building a complete classification model using the famous Iris dataset. This dataset contains measurements of iris flowers from three species, and our goal is to predict the species from the measurements.

### Project Setup

Create a new console application:

```bash
dotnet new console -n IrisClassification
cd IrisClassification
dotnet add package Microsoft.ML
```

### The Data

The Iris dataset contains 150 samples with four features each:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

And three possible species (labels):
- Iris setosa
- Iris versicolor
- Iris virginica

Create a file named `iris.csv` with sample data (or download the full dataset):

```csv
SepalLength,SepalWidth,PetalLength,PetalWidth,Species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
```

### Complete Implementation

Here's the full program:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IrisClassification;

// Input data class
public class IrisData
{
    [LoadColumn(0)]
    public float SepalLength { get; set; }
    
    [LoadColumn(1)]
    public float SepalWidth { get; set; }
    
    [LoadColumn(2)]
    public float PetalLength { get; set; }
    
    [LoadColumn(3)]
    public float PetalWidth { get; set; }
    
    [LoadColumn(4)]
    public string Species { get; set; } = string.Empty;
}

// Prediction output class
public class IrisPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedSpecies { get; set; } = string.Empty;
    
    public float[] Score { get; set; } = Array.Empty<float>();
}

class Program
{
    private static readonly string DataPath = "iris.csv";
    private static readonly string ModelPath = "IrisModel.zip";
    
    static void Main(string[] args)
    {
        Console.WriteLine("=== Iris Classification with ML.NET ===\n");
        
        // Create ML.NET context
        var mlContext = new MLContext(seed: 42);
        
        // Step 1: Load the data
        Console.WriteLine("Loading data...");
        var data = mlContext.Data.LoadFromTextFile<IrisData>(
            path: DataPath,
            hasHeader: true,
            separatorChar: ',');
        
        // Inspect the data
        var preview = data.Preview(maxRows: 5);
        Console.WriteLine($"Loaded {preview.RowView.Length} sample rows");
        Console.WriteLine($"Columns: {string.Join(", ", preview.Schema.Select(c => c.Name))}\n");
        
        // Step 2: Split into training and test sets
        Console.WriteLine("Splitting data (80% train, 20% test)...");
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        
        // Step 3: Build the training pipeline
        Console.WriteLine("Building training pipeline...\n");
        var pipeline = BuildTrainingPipeline(mlContext);
        
        // Step 4: Train the model
        Console.WriteLine("Training model...");
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var model = pipeline.Fit(split.TrainSet);
        stopwatch.Stop();
        Console.WriteLine($"Training completed in {stopwatch.ElapsedMilliseconds}ms\n");
        
        // Step 5: Evaluate the model
        Console.WriteLine("Evaluating model on test set...");
        EvaluateModel(mlContext, model, split.TestSet);
        
        // Step 6: Perform cross-validation for robust evaluation
        Console.WriteLine("\nPerforming 5-fold cross-validation...");
        CrossValidate(mlContext, pipeline, data);
        
        // Step 7: Save the model
        Console.WriteLine($"\nSaving model to {ModelPath}...");
        mlContext.Model.Save(model, data.Schema, ModelPath);
        Console.WriteLine("Model saved successfully!");
        
        // Step 8: Load and use the model for predictions
        Console.WriteLine("\n=== Making Predictions ===\n");
        MakePredictions(mlContext);
        
        Console.WriteLine("\nDone!");
    }
    
    static IEstimator<ITransformer> BuildTrainingPipeline(MLContext mlContext)
    {
        // The pipeline:
        // 1. Map the string label to a numeric key (required for classification)
        // 2. Concatenate all features into a single vector
        // 3. Normalize the features (optional but often improves results)
        // 4. Apply the classification trainer
        // 5. Map the predicted key back to the original string label
        
        return mlContext.Transforms.Conversion.MapValueToKey(
                inputColumnName: "Species",
                outputColumnName: "Label")
            .Append(mlContext.Transforms.Concatenate(
                "Features",
                nameof(IrisData.SepalLength),
                nameof(IrisData.SepalWidth),
                nameof(IrisData.PetalLength),
                nameof(IrisData.PetalWidth)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: "Label",
                featureColumnName: "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                inputColumnName: "PredictedLabel",
                outputColumnName: "PredictedLabel"));
    }
    
    static void EvaluateModel(MLContext mlContext, ITransformer model, IDataView testData)
    {
        var predictions = model.Transform(testData);
        var metrics = mlContext.MulticlassClassification.Evaluate(
            predictions,
            labelColumnName: "Label",
            predictedLabelColumnName: "PredictedLabel");
        
        Console.WriteLine($"  Micro Accuracy:    {metrics.MicroAccuracy:P2}");
        Console.WriteLine($"  Macro Accuracy:    {metrics.MacroAccuracy:P2}");
        Console.WriteLine($"  Log Loss:          {metrics.LogLoss:F4}");
        Console.WriteLine($"  Log Loss Reduction:{metrics.LogLossReduction:F4}");
        
        Console.WriteLine("\nConfusion Matrix:");
        Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
    }
    
    static void CrossValidate(MLContext mlContext, IEstimator<ITransformer> pipeline, IDataView data)
    {
        var cvResults = mlContext.MulticlassClassification.CrossValidate(
            data,
            pipeline,
            numberOfFolds: 5,
            labelColumnName: "Label");
        
        var microAccuracies = cvResults.Select(r => r.Metrics.MicroAccuracy).ToArray();
        var avgAccuracy = microAccuracies.Average();
        var stdDev = Math.Sqrt(microAccuracies.Average(a => Math.Pow(a - avgAccuracy, 2)));
        
        Console.WriteLine($"  Cross-validation results:");
        for (int i = 0; i < cvResults.Count(); i++)
        {
            Console.WriteLine($"    Fold {i + 1}: {microAccuracies[i]:P2}");
        }
        Console.WriteLine($"  Average Accuracy: {avgAccuracy:P2} (±{stdDev:P2})");
    }
    
    static void MakePredictions(MLContext mlContext)
    {
        // Load the saved model
        var model = mlContext.Model.Load(ModelPath, out var modelSchema);
        
        // Create a prediction engine
        var predictionEngine = mlContext.Model
            .CreatePredictionEngine<IrisData, IrisPrediction>(model);
        
        // Test samples
        var samples = new[]
        {
            new IrisData 
            { 
                SepalLength = 5.1f, SepalWidth = 3.5f, 
                PetalLength = 1.4f, PetalWidth = 0.2f 
            },
            new IrisData 
            { 
                SepalLength = 6.3f, SepalWidth = 2.8f, 
                PetalLength = 5.1f, PetalWidth = 1.5f 
            },
            new IrisData 
            { 
                SepalLength = 7.2f, SepalWidth = 3.0f, 
                PetalLength = 5.8f, PetalWidth = 1.6f 
            }
        };
        
        foreach (var sample in samples)
        {
            var prediction = predictionEngine.Predict(sample);
            
            Console.WriteLine($"Flower: [{sample.SepalLength}, {sample.SepalWidth}, " +
                            $"{sample.PetalLength}, {sample.PetalWidth}]");
            Console.WriteLine($"  Predicted: {prediction.PredictedSpecies}");
            Console.WriteLine($"  Confidence scores: " +
                            $"[{string.Join(", ", prediction.Score.Select(s => $"{s:F3}"))}]");
            Console.WriteLine();
        }
    }
}
```

### Understanding the Code

Let's walk through the key sections:

**Data Classes**: `IrisData` uses `[LoadColumn]` attributes to map CSV columns to properties. `IrisPrediction` captures the model's output, including the predicted label and confidence scores for each class.

**Pipeline Construction**: The pipeline performs five operations:
1. `MapValueToKey`: Converts string species names to numeric keys (ML algorithms need numbers)
2. `Concatenate`: Combines all four feature columns into one vector
3. `NormalizeMinMax`: Scales features to 0-1 range for better training
4. `SdcaMaximumEntropy`: The actual classification algorithm (a form of logistic regression)
5. `MapKeyToValue`: Converts the predicted numeric key back to the species name

**Evaluation**: We evaluate on both a held-out test set and using cross-validation. The confusion matrix helps identify which species are confused with each other.

**Model Persistence**: `mlContext.Model.Save()` serializes the entire pipeline—including learned normalization parameters and model weights—to a ZIP file. This file contains everything needed for inference.

### Running the Project

Execute the project:

```bash
dotnet run
```

Expected output:

```
=== Iris Classification with ML.NET ===

Loading data...
Loaded 5 sample rows
Columns: SepalLength, SepalWidth, PetalLength, PetalWidth, Species

Splitting data (80% train, 20% test)...
Building training pipeline...

Training model...
Training completed in 127ms

Evaluating model on test set...
  Micro Accuracy:    96.67%
  Macro Accuracy:    96.30%
  Log Loss:          0.1247
  Log Loss Reduction:0.8865

Confusion Matrix:
          ||========================
PREDICTED ||  setosa | versicolor | virginica
ACTUAL    ||========================
setosa    ||      10 |          0 |         0
versicolor||       0 |          9 |         1
virginica ||       0 |          0 |        10
          ||========================

Performing 5-fold cross-validation...
  Cross-validation results:
    Fold 1: 96.67%
    Fold 2: 100.00%
    Fold 3: 93.33%
    Fold 4: 96.67%
    Fold 5: 96.67%
  Average Accuracy: 96.67% (±2.11%)

Saving model to IrisModel.zip...
Model saved successfully!

=== Making Predictions ===

Flower: [5.1, 3.5, 1.4, 0.2]
  Predicted: setosa
  Confidence scores: [0.982, 0.015, 0.003]

Flower: [6.3, 2.8, 5.1, 1.5]
  Predicted: virginica
  Confidence scores: [0.001, 0.294, 0.705]

Flower: [7.2, 3.0, 5.8, 1.6]
  Predicted: virginica
  Confidence scores: [0.000, 0.127, 0.873]

Done!
```

### Experimenting with Different Algorithms

One of ML.NET's strengths is easy algorithm switching. Try replacing the trainer:

```csharp
// Decision tree-based approach
.Append(mlContext.MulticlassClassification.Trainers.LightGbm(
    labelColumnName: "Label",
    featureColumnName: "Features"))

// Or a support vector machine
.Append(mlContext.MulticlassClassification.Trainers.LinearSvm(
    labelColumnName: "Label",
    featureColumnName: "Features"))

// Or a neural network approach with averaged perceptron
.Append(mlContext.MulticlassClassification.Trainers
    .OneVersusAll(mlContext.BinaryClassification.Trainers.AveragedPerceptron()))
```

The pipeline pattern makes it trivial to swap algorithms while keeping all preprocessing intact.

## Summary

In this chapter, you've built a solid foundation in ML.NET:

- **MLContext** is your entry point to all ML.NET functionality
- **IDataView** provides efficient, lazy, columnar data handling optimized for ML workloads
- **Pipelines** chain transforms and trainers into reproducible workflows
- **Estimators** learn from data; **Transformers** apply learned operations
- **Train/test split** and **cross-validation** help you evaluate model generalization
- **Model serialization** enables the critical separation between training and inference
- **Model Builder** offers a visual, AutoML-powered approach for rapid development

The Iris classification project demonstrated the complete workflow: loading data, building a pipeline, training, evaluating, saving, and making predictions. This same pattern applies whether you're predicting flower species, customer churn, or house prices.

In the next chapter, we'll dive deeper into regression problems, exploring feature engineering techniques and advanced evaluation strategies. You'll build on today's foundation to tackle more complex real-world prediction tasks.

## Exercises

1. **Experiment with algorithms**: Replace SDCA with LightGBM or another trainer. How do the metrics change?

2. **Feature engineering**: Add polynomial features (e.g., SepalLength² or SepalLength × PetalLength). Does it improve accuracy?

3. **Handle missing data**: Modify the pipeline to handle rows with missing values using `ReplaceMissingValues`.

4. **Build a web API**: Create an ASP.NET Core minimal API that loads the trained model and exposes a prediction endpoint.

5. **Use Model Builder**: Recreate this project using Visual Studio's Model Builder. Compare the generated code to your hand-written version.

## Further Reading

- [ML.NET Documentation](https://docs.microsoft.com/dotnet/machine-learning/)
- [ML.NET Samples Repository](https://github.com/dotnet/machinelearning-samples)
- [ML.NET API Reference](https://docs.microsoft.com/dotnet/api/microsoft.ml)
