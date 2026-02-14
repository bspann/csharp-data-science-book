# Appendix D: Python to C# Translation Guide

If you've spent any time learning data science, you've encountered Python tutorials. They're everywhere—on Medium, Stack Overflow, Kaggle, and in countless textbooks. The Python data science ecosystem is mature and well-documented, which means many of the best learning resources assume you're working in Python.

This appendix is your Rosetta Stone. It provides direct translations from Python's data science stack (NumPy, Pandas, Scikit-learn) to C#'s equivalent tools (Microsoft.Data.Analysis, ML.NET). Keep this chapter bookmarked—you'll reference it often when adapting Python examples to your C# projects.

## The Translation Mindset

Before diving into specific translations, understand the philosophical differences:

**Python** favors concise, dynamic syntax. Operations often happen in-place, types are inferred, and the REPL encourages experimentation. Code reads almost like pseudocode.

**C#** favors explicit, type-safe syntax. Operations typically return new objects, types are declared, and the compiler catches errors before runtime. Code is more verbose but more robust.

Neither approach is "better"—they serve different purposes. When translating, resist the urge to write C# that looks like Python. Embrace C#'s strengths: strong typing catches bugs early, IDE support is exceptional, and the compiled code runs faster.

---

## Core Library Mappings

| Python Library | C# Equivalent | NuGet Package |
|---------------|---------------|---------------|
| NumPy | Microsoft.Data.Analysis | Microsoft.Data.Analysis |
| Pandas | Microsoft.Data.Analysis | Microsoft.Data.Analysis |
| Scikit-learn | ML.NET | Microsoft.ML |
| Matplotlib | ScottPlot, OxyPlot | ScottPlot / OxyPlot.Core |
| Jupyter | .NET Interactive | Microsoft.DotNet.Interactive |

---

## Data Structures: Arrays and DataFrames

### NumPy Arrays → PrimitiveDataFrameColumn

**Python (NumPy):**
```python
import numpy as np

# Create array
arr = np.array([1, 2, 3, 4, 5])

# Array operations
arr_squared = arr ** 2
arr_normalized = (arr - arr.mean()) / arr.std()

# Element access
first = arr[0]
subset = arr[1:4]
```

**C# (Microsoft.Data.Analysis):**
```csharp
using Microsoft.Data.Analysis;

// Create column (similar to 1D array)
var col = new PrimitiveDataFrameColumn<double>("values", 
    new double[] { 1, 2, 3, 4, 5 });

// Column operations
var colSquared = col * col; // Element-wise multiplication
var mean = col.Mean();
var std = col.StandardDeviation();
var colNormalized = (col - mean) / std;

// Element access
var first = col[0];
var subset = col.Clone(new long[] { 1, 2, 3 }); // Indices 1-3
```

### Pandas DataFrame → DataFrame

**Python (Pandas):**
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
})

# Access columns
ages = df['Age']
names = df['Name']

# Access rows
first_row = df.iloc[0]
subset = df.iloc[1:3]
```

**C# (Microsoft.Data.Analysis):**
```csharp
using Microsoft.Data.Analysis;

// Create DataFrame
var df = new DataFrame(
    new StringDataFrameColumn("Name", new[] { "Alice", "Bob", "Charlie" }),
    new PrimitiveDataFrameColumn<int>("Age", new[] { 25, 30, 35 }),
    new PrimitiveDataFrameColumn<double>("Salary", new[] { 50000.0, 60000.0, 70000.0 })
);

// Access columns
var ages = df["Age"];
var names = df["Name"];

// Access rows
var firstRow = df.Rows[0];
var subset = df.Head(2).Tail(1); // Or use filtering
```

---

## Data Loading and I/O

### Loading CSV Files

**Python:**
```python
import pandas as pd

# Basic load
df = pd.read_csv('data.csv')

# With options
df = pd.read_csv('data.csv', 
                  sep=',',
                  header=0,
                  na_values=['NA', 'NULL'])
```

**C#:**
```csharp
using Microsoft.Data.Analysis;

// Basic load
var df = DataFrame.LoadCsv("data.csv");

// With options
var df = DataFrame.LoadCsv("data.csv",
    separator: ',',
    header: true,
    dataTypes: new Type[] { typeof(string), typeof(int), typeof(double) });
```

### Saving to CSV

**Python:**
```python
df.to_csv('output.csv', index=False)
```

**C#:**
```csharp
DataFrame.SaveCsv(df, "output.csv");
// Or using a stream
using var stream = File.Create("output.csv");
DataFrame.SaveCsv(df, stream);
```

---

## Data Filtering and Selection

### Boolean Filtering

**Python:**
```python
# Filter rows where Age > 25
filtered = df[df['Age'] > 25]

# Multiple conditions
filtered = df[(df['Age'] > 25) & (df['Salary'] < 70000)]

# Using query (string-based)
filtered = df.query('Age > 25 and Salary < 70000')
```

**C#:**
```csharp
// Filter rows where Age > 25
var ageCol = df["Age"] as PrimitiveDataFrameColumn<int>;
var mask = ageCol.ElementwiseGreaterThan(25);
var filtered = df.Filter(mask);

// Multiple conditions
var salaryCol = df["Salary"] as PrimitiveDataFrameColumn<double>;
var ageMask = ageCol.ElementwiseGreaterThan(25);
var salaryMask = salaryCol.ElementwiseLessThan(70000);
var combinedMask = ageMask.And(salaryMask);
var filtered = df.Filter(combinedMask);

// Using LINQ (convert to enumerable first)
var filtered = df.Rows
    .Where(row => (int)row["Age"] > 25 && (double)row["Salary"] < 70000);
```

### Selecting Columns

**Python:**
```python
# Single column
ages = df['Age']

# Multiple columns
subset = df[['Name', 'Age']]

# Drop columns
df_dropped = df.drop(['Salary'], axis=1)
```

**C#:**
```csharp
// Single column
var ages = df["Age"];

// Multiple columns
var subset = df[new[] { "Name", "Age" }];

// Drop columns
var df_dropped = df.Clone();
df_dropped.Columns.Remove("Salary");
```

---

## Aggregations and Grouping

### Basic Statistics

**Python:**
```python
# Single column stats
mean = df['Salary'].mean()
std = df['Salary'].std()
min_val = df['Salary'].min()
max_val = df['Salary'].max()

# Descriptive statistics
df.describe()
```

**C#:**
```csharp
// Single column stats
var salaryCol = df["Salary"] as PrimitiveDataFrameColumn<double>;
var mean = salaryCol.Mean();
var std = salaryCol.StandardDeviation();
var min_val = salaryCol.Min();
var max_val = salaryCol.Max();

// Descriptive statistics
var description = df.Description();
```

### GroupBy Operations

**Python:**
```python
# Group and aggregate
grouped = df.groupby('Department')['Salary'].mean()

# Multiple aggregations
grouped = df.groupby('Department').agg({
    'Salary': ['mean', 'sum', 'count'],
    'Age': 'mean'
})
```

**C#:**
```csharp
// Group and aggregate
var grouped = df.GroupBy("Department")
    .Mean("Salary");

// Multiple aggregations (chain operations)
var grouped = df.GroupBy("Department");
var meanSalary = grouped.Mean("Salary");
var sumSalary = grouped.Sum("Salary");
var count = grouped.Count();

// Or use LINQ for complex aggregations
var results = df.Rows
    .GroupBy(row => row["Department"])
    .Select(g => new {
        Department = g.Key,
        AvgSalary = g.Average(r => (double)r["Salary"]),
        AvgAge = g.Average(r => (int)r["Age"]),
        Count = g.Count()
    });
```

---

## Data Transformation

### Adding and Modifying Columns

**Python:**
```python
# Add new column
df['Bonus'] = df['Salary'] * 0.1

# Modify existing
df['Salary'] = df['Salary'] * 1.05

# Apply function
df['Name_Upper'] = df['Name'].apply(lambda x: x.upper())
```

**C#:**
```csharp
// Add new column
var salaryCol = df["Salary"] as PrimitiveDataFrameColumn<double>;
var bonusCol = salaryCol * 0.1;
bonusCol.SetName("Bonus");
df.Columns.Add(bonusCol);

// Modify existing
df["Salary"] = salaryCol * 1.05;

// Apply function (create new column from transformation)
var nameCol = df["Name"] as StringDataFrameColumn;
var upperNames = new StringDataFrameColumn("Name_Upper", 
    nameCol.Select(n => n?.ToUpper()));
df.Columns.Add(upperNames);
```

### Handling Missing Values

**Python:**
```python
# Check for nulls
df.isnull().sum()

# Drop rows with nulls
df_clean = df.dropna()

# Fill nulls
df_filled = df.fillna(0)
df_filled = df.fillna(df.mean())  # Fill with column mean
```

**C#:**
```csharp
// Check for nulls
foreach (var col in df.Columns)
{
    var nullCount = col.NullCount;
    Console.WriteLine($"{col.Name}: {nullCount} nulls");
}

// Drop rows with nulls
var cleanDf = df.DropNulls();

// Fill nulls with value
var col = df["Salary"] as PrimitiveDataFrameColumn<double>;
col.FillNulls(0, inPlace: true);

// Fill with mean
var mean = col.Mean();
col.FillNulls(mean.Value, inPlace: true);
```

### Sorting

**Python:**
```python
# Sort by single column
df_sorted = df.sort_values('Age')

# Descending
df_sorted = df.sort_values('Age', ascending=False)

# Multiple columns
df_sorted = df.sort_values(['Department', 'Salary'], 
                           ascending=[True, False])
```

**C#:**
```csharp
// Sort by single column
var dfSorted = df.OrderBy("Age");

// Descending
var dfSorted = df.OrderByDescending("Age");

// Multiple columns (chain or use custom comparer)
var dfSorted = df.OrderBy("Department")
    .ThenByDescending("Salary"); // If available
    
// Or use LINQ
var sorted = df.Rows
    .OrderBy(r => r["Department"])
    .ThenByDescending(r => (double)r["Salary"]);
```

---

## Scikit-learn to ML.NET

This section maps the most common Scikit-learn patterns to their ML.NET equivalents.

### The Pipeline Pattern

**Python (Scikit-learn):**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**C# (ML.NET):**
```csharp
using Microsoft.ML;

var mlContext = new MLContext();

var pipeline = mlContext.Transforms.NormalizeMinMax("Features")
    .Append(mlContext.BinaryClassification.Trainers
        .SdcaLogisticRegression(labelColumnName: "Label", 
                                 featureColumnName: "Features"));

var model = pipeline.Fit(trainData);
var predictions = model.Transform(testData);
```

### Train/Test Split

**Python:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

**C#:**
```csharp
var mlContext = new MLContext(seed: 42);

var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
var trainData = splitData.TrainSet;
var testData = splitData.TestSet;
```

### Feature Normalization

**Python:**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Z-score normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

**C#:**
```csharp
// Z-score normalization (mean=0, std=1)
var normalizer = mlContext.Transforms.NormalizeMeanVariance("Features");

// Min-Max scaling (0 to 1)
var normalizer = mlContext.Transforms.NormalizeMinMax("Features");

// Apply to data
var normalizedData = normalizer.Fit(data).Transform(data);
```

### One-Hot Encoding

**Python:**
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
encoded = encoder.fit_transform(df[['Category']])
```

**C#:**
```csharp
var encoder = mlContext.Transforms.Categorical
    .OneHotEncoding("CategoryEncoded", "Category");

var encodedData = encoder.Fit(data).Transform(data);
```

### Linear Regression

**Python:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
```

**C#:**
```csharp
// Build pipeline
var pipeline = mlContext.Transforms
    .Concatenate("Features", featureColumns)
    .Append(mlContext.Regression.Trainers.Sdca(
        labelColumnName: "Label",
        featureColumnName: "Features"));

// Train
var model = pipeline.Fit(trainData);

// Predict and evaluate
var predictions = model.Transform(testData);
var metrics = mlContext.Regression.Evaluate(predictions, "Label");

Console.WriteLine($"MSE: {metrics.MeanSquaredError}");
Console.WriteLine($"R²: {metrics.RSquared}");
```

### Logistic Regression (Binary Classification)

**Python:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
```

**C#:**
```csharp
var pipeline = mlContext.Transforms
    .Concatenate("Features", featureColumns)
    .Append(mlContext.BinaryClassification.Trainers
        .SdcaLogisticRegression(labelColumnName: "Label"));

var model = pipeline.Fit(trainData);
var predictions = model.Transform(testData);

var metrics = mlContext.BinaryClassification
    .Evaluate(predictions, "Label");

Console.WriteLine($"Accuracy: {metrics.Accuracy}");
Console.WriteLine($"Precision: {metrics.PositivePrecision}");
Console.WriteLine($"Recall: {metrics.PositiveRecall}");
Console.WriteLine($"F1: {metrics.F1Score}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve}");
```

### Random Forest

**Python:**
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**C#:**
```csharp
var trainer = mlContext.BinaryClassification.Trainers
    .FastForest(
        labelColumnName: "Label",
        featureColumnName: "Features",
        numberOfTrees: 100,
        maximumDepth: 10);

var pipeline = mlContext.Transforms
    .Concatenate("Features", featureColumns)
    .Append(trainer);

var model = pipeline.Fit(trainData);
```

### K-Means Clustering

**Python:**
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
```

**C#:**
```csharp
var pipeline = mlContext.Transforms
    .Concatenate("Features", featureColumns)
    .Append(mlContext.Clustering.Trainers.KMeans(
        featureColumnName: "Features",
        numberOfClusters: 3));

var model = pipeline.Fit(data);
var predictions = model.Transform(data);
```

### Cross-Validation

**Python:**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

**C#:**
```csharp
var cvResults = mlContext.BinaryClassification
    .CrossValidate(data, pipeline, numberOfFolds: 5);

var avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
var stdDev = Math.Sqrt(cvResults.Average(r => 
    Math.Pow(r.Metrics.Accuracy - avgAccuracy, 2)));

Console.WriteLine($"Accuracy: {avgAccuracy:F3} (+/- {stdDev * 2:F3})");
```

### Model Persistence

**Python:**
```python
import joblib

# Save
joblib.dump(model, 'model.pkl')

# Load
model = joblib.load('model.pkl')
```

**C#:**
```csharp
// Save
mlContext.Model.Save(model, trainData.Schema, "model.zip");

// Load
var loadedModel = mlContext.Model.Load("model.zip", out var schema);
```

---

## Quick Reference Tables

### NumPy Array Operations

| NumPy | C# (PrimitiveDataFrameColumn) |
|-------|-------------------------------|
| `arr.shape` | `col.Length` |
| `arr.dtype` | `col.DataType` |
| `arr.sum()` | `col.Sum()` |
| `arr.mean()` | `col.Mean()` |
| `arr.std()` | `col.StandardDeviation()` |
| `arr.min()` | `col.Min()` |
| `arr.max()` | `col.Max()` |
| `arr * 2` | `col * 2` |
| `arr + arr2` | `col.Add(col2)` |
| `arr ** 2` | `col * col` |
| `np.sqrt(arr)` | Custom transform needed |
| `arr[arr > 0]` | `col.Filter(col.ElementwiseGreaterThan(0))` |

### Pandas DataFrame Operations

| Pandas | C# (DataFrame) |
|--------|----------------|
| `df.shape` | `(df.Rows.Count, df.Columns.Count)` |
| `df.columns` | `df.Columns.Select(c => c.Name)` |
| `df.head(n)` | `df.Head(n)` |
| `df.tail(n)` | `df.Tail(n)` |
| `df.info()` | `df.Info()` |
| `df.describe()` | `df.Description()` |
| `df['col']` | `df["col"]` |
| `df[['a','b']]` | `df[new[] {"a", "b"}]` |
| `df.iloc[0]` | `df.Rows[0]` |
| `df.dropna()` | `df.DropNulls()` |
| `df.fillna(v)` | `col.FillNulls(v, true)` |
| `df.groupby('x')` | `df.GroupBy("x")` |
| `df.merge(df2)` | `df.Merge(df2, ...)` |
| `df.sort_values('x')` | `df.OrderBy("x")` |
| `pd.read_csv()` | `DataFrame.LoadCsv()` |
| `df.to_csv()` | `DataFrame.SaveCsv()` |

### Scikit-learn to ML.NET Trainers

| Scikit-learn | ML.NET |
|--------------|--------|
| `LinearRegression` | `Regression.Trainers.Sdca` |
| `Ridge` | `Regression.Trainers.Sdca` (with L2) |
| `Lasso` | `Regression.Trainers.Sdca` (with L1) |
| `LogisticRegression` | `BinaryClassification.Trainers.SdcaLogisticRegression` |
| `RandomForestClassifier` | `BinaryClassification.Trainers.FastForest` |
| `RandomForestRegressor` | `Regression.Trainers.FastForest` |
| `GradientBoostingClassifier` | `BinaryClassification.Trainers.FastTree` |
| `DecisionTreeClassifier` | `BinaryClassification.Trainers.FastTree` (depth=1) |
| `KMeans` | `Clustering.Trainers.KMeans` |
| `PCA` | `Transforms.ProjectToPrincipalComponents` |
| `StandardScaler` | `Transforms.NormalizeMeanVariance` |
| `MinMaxScaler` | `Transforms.NormalizeMinMax` |
| `OneHotEncoder` | `Transforms.Categorical.OneHotEncoding` |
| `LabelEncoder` | `Transforms.Conversion.MapValueToKey` |

---

## Common Gotchas

### 1. Null Handling
Python treats missing values as `NaN` (a special float). C# uses nullable types (`double?`, `int?`). Always check for nulls before arithmetic.

### 2. Zero-Based vs One-Based
Both languages use zero-based indexing, but be careful with `iloc` translations—C#'s row access is straightforward but less flexible.

### 3. Chained Operations
Pandas encourages chaining (`df.filter().groupby().mean()`). C# supports this too, but you may need intermediate variables for complex chains.

### 4. In-Place vs Copy
Pandas often modifies in-place by default. Microsoft.Data.Analysis usually returns new objects. Check the `inPlace` parameter when available.

### 5. Type Casting
Python is dynamically typed; C# requires explicit casts. When accessing DataFrame columns, cast to the appropriate column type:
```csharp
var col = df["Salary"] as PrimitiveDataFrameColumn<double>;
```

### 6. Feature Column Names
ML.NET is strict about column names. Always ensure your feature column is named "Features" (or specify the name explicitly in trainer options).

---

## Translating a Complete Example

Let's translate a typical Python ML workflow end-to-end.

**Python:**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('customers.csv')

# Prepare features and labels
X = df[['Age', 'Income', 'SpendingScore']]
y = df['Churned']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate
predictions = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
```

**C#:**
```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 42);

// Define data schema
public class CustomerData
{
    [LoadColumn(0)] public float Age { get; set; }
    [LoadColumn(1)] public float Income { get; set; }
    [LoadColumn(2)] public float SpendingScore { get; set; }
    [LoadColumn(3)] public bool Churned { get; set; }
}

public class ChurnPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
}

// Load data
var data = mlContext.Data.LoadFromTextFile<CustomerData>(
    "customers.csv", hasHeader: true, separatorChar: ',');

// Split
var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

// Build pipeline (concatenate features, scale, train)
var pipeline = mlContext.Transforms.Concatenate("Features", 
        "Age", "Income", "SpendingScore")
    .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
        labelColumnName: "Churned",
        featureColumnName: "Features"));

// Train
var model = pipeline.Fit(splitData.TrainSet);

// Evaluate
var predictions = model.Transform(splitData.TestSet);
var metrics = mlContext.BinaryClassification.Evaluate(
    predictions, labelColumnName: "Churned");

Console.WriteLine($"Accuracy: {metrics.Accuracy:F3}");
```

The C# version is longer but more explicit. Each step is clear, types are defined upfront, and the pipeline encapsulates all transformations. Once you're comfortable with this pattern, translating Python tutorials becomes mechanical.

---

## Text Processing and Feature Extraction

Text data requires special handling in both ecosystems. Here's how common text operations translate.

### Text Vectorization

**Python:**
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bag of words
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform(documents)

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X = tfidf.fit_transform(documents)
```

**C#:**
```csharp
// Featurize text (includes tokenization, n-grams, and TF-IDF)
var textPipeline = mlContext.Transforms.Text
    .FeaturizeText("Features", new TextFeaturizingEstimator.Options
    {
        WordFeatureExtractor = new WordBagEstimator.Options 
        { 
            NgramLength = 2,
            MaximumNgramsCount = new[] { 1000 }
        },
        CharFeatureExtractor = null // Disable character n-grams
    }, "TextColumn");

// Or step-by-step for more control
var pipeline = mlContext.Transforms.Text
    .NormalizeText("NormalizedText", "TextColumn")
    .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedText"))
    .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("CleanTokens", "Tokens"))
    .Append(mlContext.Transforms.Text.ProduceNgrams("Ngrams", "CleanTokens", 
        ngramLength: 2))
    .Append(mlContext.Transforms.Text.LatentDirichletAllocation("Features", "Ngrams"));
```

### Sentiment Analysis Pipeline

**Python:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

**C#:**
```csharp
var pipeline = mlContext.Transforms.Text
    .FeaturizeText("Features", "ReviewText")
    .Append(mlContext.BinaryClassification.Trainers
        .SdcaLogisticRegression(labelColumnName: "Sentiment"));

var model = pipeline.Fit(trainData);
var predictions = model.Transform(testData);
```

---

## Multi-Class Classification

When you have more than two categories, the patterns shift slightly.

**Python:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report

# Multi-class logistic regression
model = LogisticRegression(multi_class='multinomial')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
```

**C#:**
```csharp
// ML.NET handles multi-class automatically
var pipeline = mlContext.Transforms
    .Conversion.MapValueToKey("Label", "Category")  // Convert string to key
    .Append(mlContext.Transforms.Concatenate("Features", featureColumns))
    .Append(mlContext.MulticlassClassification.Trainers
        .SdcaMaximumEntropy(labelColumnName: "Label"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

var model = pipeline.Fit(trainData);
var predictions = model.Transform(testData);

var metrics = mlContext.MulticlassClassification.Evaluate(predictions);
Console.WriteLine($"Macro Accuracy: {metrics.MacroAccuracy:F3}");
Console.WriteLine($"Micro Accuracy: {metrics.MicroAccuracy:F3}");
Console.WriteLine($"Log Loss: {metrics.LogLoss:F3}");

// Per-class metrics
foreach (var classMetric in metrics.PerClassLogLoss)
{
    Console.WriteLine($"  Class Log Loss: {classMetric:F3}");
}
```

---

## Time Series and Date Handling

Date columns require special attention during translation.

**Python:**
```python
import pandas as pd

df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])

# Lag features
df['Value_Lag1'] = df['Value'].shift(1)
df['Value_Lag7'] = df['Value'].shift(7)

# Rolling statistics
df['Value_Rolling7Mean'] = df['Value'].rolling(window=7).mean()
```

**C#:**
```csharp
// After loading, transform date columns manually or via LINQ
var processedData = rawData.Select(row => new ProcessedRow
{
    Date = DateTime.Parse(row.DateString),
    Year = DateTime.Parse(row.DateString).Year,
    Month = DateTime.Parse(row.DateString).Month,
    DayOfWeek = (int)DateTime.Parse(row.DateString).DayOfWeek,
    IsWeekend = DateTime.Parse(row.DateString).DayOfWeek == DayOfWeek.Saturday 
             || DateTime.Parse(row.DateString).DayOfWeek == DayOfWeek.Sunday,
    Value = row.Value
}).ToList();

// For lag features, process in order
for (int i = 1; i < processedData.Count; i++)
{
    processedData[i].ValueLag1 = processedData[i - 1].Value;
    if (i >= 7)
        processedData[i].ValueLag7 = processedData[i - 7].Value;
}

// Rolling mean
for (int i = 6; i < processedData.Count; i++)
{
    processedData[i].ValueRolling7Mean = processedData
        .Skip(i - 6).Take(7)
        .Average(r => r.Value);
}
```

For time series forecasting specifically, ML.NET provides the Forecasting API:

```csharp
// Single Spectrum Analysis (SSA) forecasting
var forecastingPipeline = mlContext.Forecasting
    .ForecastBySsa(
        outputColumnName: "ForecastedValues",
        inputColumnName: "Value",
        windowSize: 7,
        seriesLength: 30,
        trainSize: 365,
        horizon: 7);

var model = forecastingPipeline.Fit(trainData);
var forecast = model.Transform(testData);
```

---

## Image Data Loading

Both ecosystems support image processing, though the approaches differ.

**Python:**
```python
from sklearn.datasets import load_digits
from PIL import Image
import numpy as np

# Load built-in dataset
digits = load_digits()
X, y = digits.data, digits.target

# Load custom images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        img_array = np.array(img.resize((64, 64))).flatten()
        images.append(img_array)
    return np.array(images)
```

**C#:**
```csharp
// Define image data class
public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath { get; set; }
    
    [LoadColumn(1)]
    public string Label { get; set; }
}

public class ImagePrediction
{
    public float[] Score { get; set; }
    public string PredictedLabel { get; set; }
}

// Image classification pipeline
var pipeline = mlContext.Transforms
    .Conversion.MapValueToKey("LabelKey", "Label")
    .Append(mlContext.Transforms.LoadImages("ImageObject", 
        imageFolder: imageFolder, inputColumnName: "ImagePath"))
    .Append(mlContext.Transforms.ResizeImages("ResizedImage", 
        imageWidth: 224, imageHeight: 224, inputColumnName: "ImageObject"))
    .Append(mlContext.Transforms.ExtractPixels("Pixels", "ResizedImage"))
    .Append(mlContext.MulticlassClassification.Trainers
        .ImageClassification(featureColumnName: "Pixels", 
                             labelColumnName: "LabelKey"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
```

---

## Hyperparameter Tuning

Python's GridSearchCV has no direct equivalent in ML.NET, but you can implement similar functionality.

**Python:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

**C#:**
```csharp
// Manual grid search implementation
var parameterSets = new[]
{
    new { Trees = 50, Depth = 5 },
    new { Trees = 50, Depth = 10 },
    new { Trees = 100, Depth = 5 },
    new { Trees = 100, Depth = 10 },
    new { Trees = 200, Depth = 5 },
    new { Trees = 200, Depth = 10 },
};

var bestAccuracy = 0.0;
var bestParams = parameterSets[0];

foreach (var parameters in parameterSets)
{
    var trainer = mlContext.BinaryClassification.Trainers.FastForest(
        numberOfTrees: parameters.Trees,
        maximumBinCountPerFeature: parameters.Depth);
    
    var pipeline = mlContext.Transforms
        .Concatenate("Features", featureColumns)
        .Append(trainer);
    
    var cvResults = mlContext.BinaryClassification
        .CrossValidate(data, pipeline, numberOfFolds: 5);
    
    var avgAccuracy = cvResults.Average(r => r.Metrics.Accuracy);
    
    if (avgAccuracy > bestAccuracy)
    {
        bestAccuracy = avgAccuracy;
        bestParams = parameters;
    }
    
    Console.WriteLine($"Trees={parameters.Trees}, Depth={parameters.Depth}: " +
                      $"Accuracy={avgAccuracy:F4}");
}

Console.WriteLine($"\nBest: Trees={bestParams.Trees}, Depth={bestParams.Depth}");
Console.WriteLine($"Best Accuracy: {bestAccuracy:F4}");
```

For more sophisticated tuning, consider using ML.NET's AutoML:

```csharp
// AutoML handles hyperparameter tuning automatically
var experimentResult = mlContext.Auto()
    .CreateBinaryClassificationExperiment(maxExperimentTimeInSeconds: 120)
    .Execute(trainData, labelColumnName: "Label");

Console.WriteLine($"Best trainer: {experimentResult.BestRun.TrainerName}");
Console.WriteLine($"Accuracy: {experimentResult.BestRun.ValidationMetrics.Accuracy}");

var bestModel = experimentResult.BestRun.Model;
```

---

## Confusion Matrix and Detailed Metrics

**Python:**
```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, predictions)
print(cm)
print(classification_report(y_test, predictions))
```

**C#:**
```csharp
var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Label");

// Confusion matrix
Console.WriteLine($"True Positives: {metrics.ConfusionMatrix.GetCountForClassPair(1, 1)}");
Console.WriteLine($"True Negatives: {metrics.ConfusionMatrix.GetCountForClassPair(0, 0)}");
Console.WriteLine($"False Positives: {metrics.ConfusionMatrix.GetCountForClassPair(0, 1)}");
Console.WriteLine($"False Negatives: {metrics.ConfusionMatrix.GetCountForClassPair(1, 0)}");

// Detailed metrics
Console.WriteLine($"\nAccuracy: {metrics.Accuracy:F4}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
Console.WriteLine($"Precision: {metrics.PositivePrecision:F4}");
Console.WriteLine($"Recall: {metrics.PositiveRecall:F4}");
Console.WriteLine($"Negative Precision: {metrics.NegativePrecision:F4}");
Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F4}");

// Print formatted confusion matrix
Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
```

---

## Making Single Predictions

After training, you often need to predict on individual samples.

**Python:**
```python
# Single prediction
sample = [[25, 50000, 75]]  # Age, Income, SpendingScore
prediction = model.predict(sample)
probability = model.predict_proba(sample)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

**C#:**
```csharp
// Create prediction engine for efficient single predictions
var predEngine = mlContext.Model
    .CreatePredictionEngine<CustomerData, ChurnPrediction>(model);

var sample = new CustomerData
{
    Age = 25,
    Income = 50000,
    SpendingScore = 75
};

var prediction = predEngine.Predict(sample);

Console.WriteLine($"Prediction: {prediction.Prediction}");
Console.WriteLine($"Probability: {prediction.Probability:F4}");
```

The PredictionEngine is not thread-safe. For production scenarios with concurrent requests, use `PredictionEnginePool`:

```csharp
// In Startup.cs or Program.cs
services.AddPredictionEnginePool<CustomerData, ChurnPrediction>()
    .FromFile("model.zip");

// In your controller/service
public class PredictionService
{
    private readonly PredictionEnginePool<CustomerData, ChurnPrediction> _pool;
    
    public PredictionService(PredictionEnginePool<CustomerData, ChurnPrediction> pool)
    {
        _pool = pool;
    }
    
    public ChurnPrediction Predict(CustomerData input)
    {
        return _pool.Predict(input);
    }
}
```

---

## Final Tips

1. **Start with the pipeline.** Identify what transforms and trainers the Python code uses, then find the ML.NET equivalents.

2. **Define your data classes first.** C#'s type system requires explicit schemas—this is actually an advantage for catching data issues early.

3. **Use the ML.NET Model Builder.** Visual Studio's Model Builder can generate starter code from your data, giving you a template to customize.

4. **Check the ML.NET samples repository.** Microsoft maintains extensive examples at github.com/dotnet/machinelearning-samples.

5. **When stuck, search for the ML.NET equivalent.** The API names are often similar—`StandardScaler` becomes `NormalizeMeanVariance`, `OneHotEncoder` becomes `OneHotEncoding`.

6. **Embrace the verbosity.** C# code is longer than Python, but that explicitness helps you understand exactly what's happening at each step. It also makes debugging easier.

7. **Use AutoML for exploration.** When you're not sure which algorithm to use, let ML.NET's AutoML explore the space and suggest the best approach for your data.

8. **Profile your data loading.** In production, data loading is often the bottleneck. Use IDataView's lazy loading and streaming capabilities to handle large datasets efficiently.

With practice, you'll develop an intuition for these translations. The patterns repeat, and soon you'll be reading Python tutorials while writing C# code in your head. Keep this appendix bookmarked—it's your bridge between the two ecosystems.
