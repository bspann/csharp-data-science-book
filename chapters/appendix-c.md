# Appendix C: Dataset Sources

Finding quality datasets is essential for practicing data science techniques and building real-world applications. This appendix provides a curated collection of dataset sources organized by task type, with practical guidance on accessing and loading each in C#.

## Public Dataset Repositories

### UCI Machine Learning Repository

The UCI Machine Learning Repository (https://archive.ics.uci.edu) is one of the oldest and most respected sources for machine learning datasets. Maintained by the University of California, Irvine, it hosts over 600 datasets covering classification, regression, clustering, and other tasks.

**Key Characteristics:**
- Free and open access
- Well-documented datasets with academic citations
- Clean, preprocessed data ideal for learning
- CSV and ARFF formats predominate

**Loading UCI Datasets in C#:**

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

// Download and load the Iris dataset
var mlContext = new MLContext();
string dataPath = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";

// Download to local file first (recommended for large datasets)
using var client = new HttpClient();
var content = await client.GetStringAsync(dataPath);
await File.WriteAllTextAsync("iris.data", content);

// Load into ML.NET
var data = mlContext.Data.LoadFromTextFile<IrisData>(
    "iris.data",
    separatorChar: ',',
    hasHeader: false
);
```

### Scikit-learn Datasets

While scikit-learn is a Python library, its classic datasets are available in various formats and are excellent for learning. These datasets are small, well-understood, and perfect for algorithm comparison.

**Classic Datasets Available:**

| Dataset | Samples | Features | Task | Difficulty |
|---------|---------|----------|------|------------|
| Iris | 150 | 4 | Classification | Beginner |
| Wine | 178 | 13 | Classification | Beginner |
| Breast Cancer | 569 | 30 | Classification | Beginner |
| Digits | 1,797 | 64 | Classification | Intermediate |
| Boston Housing | 506 | 13 | Regression | Beginner |
| Diabetes | 442 | 10 | Regression | Beginner |
| California Housing | 20,640 | 8 | Regression | Intermediate |

**Accessing in C# via ML.NET:**

ML.NET provides built-in sample datasets through NuGet packages:

```csharp
// Install: Microsoft.ML.SamplesUtils (if available)
// Or download from: https://github.com/dotnet/machinelearning/tree/main/test/data

// The ML.NET samples repository contains these datasets in CSV format
string irisUrl = "https://raw.githubusercontent.com/dotnet/machinelearning/main/test/data/iris.txt";
```

### Azure Open Datasets

Microsoft's Azure Open Datasets (https://azure.microsoft.com/services/open-datasets/) provides curated datasets optimized for Azure integration but accessible to all developers.

**Notable Datasets:**

- **US Population by ZIP Code** – Demographics data, updated annually
- **Public Holidays** – Global holiday calendars
- **NOAA Weather Data** – Historical weather observations
- **NYC Taxi Data** – Billions of trip records for time series analysis
- **COVID-19 Data** – Pandemic tracking statistics

**Accessing Azure Open Datasets in C#:**

```csharp
using Azure.Storage.Blobs;

// Azure Open Datasets are stored in Azure Blob Storage
// Many are accessible without authentication
string containerUrl = "https://azureopendatastorage.blob.core.windows.net/nyctlc";
var blobServiceClient = new BlobServiceClient(new Uri(containerUrl));

// For authenticated access with your Azure subscription
using Azure.Identity;
var credential = new DefaultAzureCredential();
var client = new BlobServiceClient(
    new Uri("https://azureopendatastorage.blob.core.windows.net"),
    credential
);
```

### Kaggle

Kaggle (https://kaggle.com) hosts the world's largest data science community with thousands of datasets and competitions. Datasets range from toy examples to enterprise-scale challenges.

**Key Benefits:**
- Massive variety (50,000+ datasets)
- Community notebooks showing usage patterns
- Competition datasets with known benchmarks
- Active discussion forums

**License Considerations:** Each Kaggle dataset has its own license. Check before commercial use. Common licenses include CC0 (public domain), CC BY (attribution required), and custom competition licenses.

**Kaggle API Integration in C#:**

```csharp
using System.Diagnostics;

// First, install Kaggle CLI: pip install kaggle
// Set up ~/.kaggle/kaggle.json with your API credentials

public async Task DownloadKaggleDataset(string dataset, string outputPath)
{
    var process = new Process
    {
        StartInfo = new ProcessStartInfo
        {
            FileName = "kaggle",
            Arguments = $"datasets download -d {dataset} -p {outputPath} --unzip",
            RedirectStandardOutput = true,
            UseShellExecute = false
        }
    };
    
    process.Start();
    await process.WaitForExitAsync();
}

// Usage
await DownloadKaggleDataset("uciml/iris", "./data");
```

---

## Datasets by Task Type

### Classification Datasets

**Beginner Level:**

| Dataset | Source | Samples | Classes | License |
|---------|--------|---------|---------|---------|
| Iris | UCI | 150 | 3 | CC BY 4.0 |
| Wine Quality | UCI | 6,497 | 10 | CC BY 4.0 |
| Titanic | Kaggle | 891 | 2 | Public |
| Heart Disease | UCI | 303 | 2 | CC BY 4.0 |

**Intermediate Level:**

| Dataset | Source | Samples | Classes | License |
|---------|--------|---------|---------|---------|
| Adult Income | UCI | 48,842 | 2 | CC BY 4.0 |
| MNIST Digits | Multiple | 70,000 | 10 | Public |
| Fashion MNIST | Kaggle | 70,000 | 10 | MIT |
| Credit Card Fraud | Kaggle | 284,807 | 2 | DbCL |

**Advanced Level:**

| Dataset | Source | Samples | Classes | License |
|---------|--------|---------|---------|---------|
| ImageNet | ImageNet.org | 14M+ | 21,841 | Research only |
| CIFAR-100 | Toronto CS | 60,000 | 100 | Research |

### Regression Datasets

**Beginner Level:**

| Dataset | Source | Samples | Features | License |
|---------|--------|---------|----------|---------|
| Boston Housing | UCI | 506 | 13 | Public |
| Auto MPG | UCI | 398 | 7 | CC BY 4.0 |
| Diabetes | sklearn | 442 | 10 | BSD |

**Intermediate Level:**

| Dataset | Source | Samples | Features | License |
|---------|--------|---------|----------|---------|
| California Housing | sklearn | 20,640 | 8 | Public |
| House Prices | Kaggle | 1,460 | 79 | Competition |
| Bike Sharing | UCI | 17,389 | 16 | CC BY 4.0 |
| Energy Efficiency | UCI | 768 | 8 | CC BY 4.0 |

**Advanced Level:**

| Dataset | Source | Samples | Features | License |
|---------|--------|---------|----------|---------|
| NYC Taxi Fares | Kaggle | 55M+ | 8 | Competition |
| Ames Housing | Kaggle | 2,930 | 80 | Public |

### Natural Language Processing (NLP) Datasets

**Beginner Level:**

| Dataset | Task | Size | License |
|---------|------|------|---------|
| IMDB Reviews | Sentiment | 50,000 | Public |
| SMS Spam | Classification | 5,574 | CC BY 4.0 |
| 20 Newsgroups | Classification | 20,000 | Public |

**Intermediate Level:**

| Dataset | Task | Size | License |
|---------|------|------|---------|
| Amazon Reviews | Sentiment | 3.6M | Research |
| Yelp Reviews | Sentiment | 6.9M | Research |
| SQuAD | Q&A | 100,000+ | CC BY-SA |
| CoNLL-2003 | NER | 22,137 | Research |

**Loading NLP Datasets in C#:**

```csharp
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

// Load IMDB-style sentiment data
public class SentimentData
{
    [LoadColumn(0)]
    public string? ReviewText { get; set; }
    
    [LoadColumn(1)]
    public bool Sentiment { get; set; }
}

var mlContext = new MLContext();
var data = mlContext.Data.LoadFromTextFile<SentimentData>(
    "imdb_reviews.csv",
    separatorChar: ',',
    hasHeader: true
);

// Text featurization pipeline
var pipeline = mlContext.Transforms.Text.FeaturizeText(
    outputColumnName: "Features",
    inputColumnName: nameof(SentimentData.ReviewText)
);
```

### Time Series Datasets

| Dataset | Source | Records | Frequency | License |
|---------|--------|---------|-----------|---------|
| Air Passengers | R datasets | 144 | Monthly | Public |
| Sunspots | R datasets | 2,820 | Monthly | Public |
| Stock Prices | Yahoo Finance | Varies | Daily | Terms apply |
| Energy Consumption | UCI | 2M+ | Minute | CC BY 4.0 |
| Web Traffic | Kaggle | 145,000 | Daily | Competition |
| M4 Competition | Makridakis | 100,000 | Various | Research |

**Loading Time Series in C#:**

```csharp
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

public class EnergyData
{
    [LoadColumn(0)]
    public DateTime Timestamp { get; set; }
    
    [LoadColumn(1)]
    public float Consumption { get; set; }
}

var mlContext = new MLContext();
var data = mlContext.Data.LoadFromTextFile<EnergyData>(
    "energy.csv",
    separatorChar: ',',
    hasHeader: true
);

// Detect anomalies in time series
var pipeline = mlContext.Transforms.DetectSpikeBySsa(
    outputColumnName: "Prediction",
    inputColumnName: nameof(EnergyData.Consumption),
    confidence: 95,
    pvalueHistoryLength: 30,
    trainingWindowSize: 90,
    seasonalityWindowSize: 30
);
```

### Image Datasets

**Beginner Level:**

| Dataset | Images | Classes | Size | License |
|---------|--------|---------|------|---------|
| MNIST | 70,000 | 10 | 50 MB | Public |
| Fashion MNIST | 70,000 | 10 | 30 MB | MIT |
| CIFAR-10 | 60,000 | 10 | 163 MB | Research |

**Intermediate Level:**

| Dataset | Images | Classes | Size | License |
|---------|--------|---------|------|---------|
| Cats vs Dogs | 25,000 | 2 | 800 MB | Public |
| Stanford Dogs | 20,580 | 120 | 750 MB | Research |
| Food-101 | 101,000 | 101 | 5 GB | Research |

**Advanced Level:**

| Dataset | Images | Classes | Size | License |
|---------|--------|---------|------|---------|
| COCO | 330,000 | 80 | 25 GB | CC BY 4.0 |
| ImageNet | 14M+ | 21,841 | 150 GB | Research |
| Open Images | 9M | 600 | 500 GB | CC BY 4.0 |

**Loading Images in C#:**

```csharp
using Microsoft.ML;
using Microsoft.ML.Vision;

public class ImageData
{
    public string ImagePath { get; set; } = "";
    public string Label { get; set; } = "";
}

var mlContext = new MLContext();

// Load images from folder structure (folder name = label)
var images = Directory.GetDirectories("./images")
    .SelectMany(folder => Directory.GetFiles(folder, "*.jpg")
        .Select(file => new ImageData
        {
            ImagePath = file,
            Label = Path.GetFileName(folder)
        }));

var data = mlContext.Data.LoadFromEnumerable(images);

// Image classification pipeline
var pipeline = mlContext.Transforms.Conversion.MapValueToKey("LabelKey", "Label")
    .Append(mlContext.Transforms.LoadImages("Image", "", "ImagePath"))
    .Append(mlContext.Transforms.ResizeImages("Image", 224, 224))
    .Append(mlContext.Transforms.ExtractPixels("Pixels", "Image"))
    .Append(mlContext.MulticlassClassification.Trainers
        .ImageClassification("LabelKey", "Image"));
```

---

## Best Practices for Dataset Management

### 1. Always Check Licenses

Before using any dataset, verify:
- **Commercial use** – Many research datasets prohibit commercial applications
- **Attribution requirements** – CC BY licenses require crediting the source
- **Redistribution rights** – Can you include the data in your application?

### 2. Create a Data Loading Utility

```csharp
public static class DatasetLoader
{
    private static readonly HttpClient _client = new();
    
    public static async Task<string> DownloadDatasetAsync(
        string url, 
        string localPath,
        bool overwrite = false)
    {
        if (File.Exists(localPath) && !overwrite)
            return localPath;
            
        var content = await _client.GetStringAsync(url);
        await File.WriteAllTextAsync(localPath, content);
        return localPath;
    }
    
    public static async Task<IDataView> LoadUciDatasetAsync(
        MLContext mlContext,
        string datasetName,
        char separator = ',',
        bool hasHeader = false)
    {
        string url = $"https://archive.ics.uci.edu/ml/machine-learning-databases/{datasetName}";
        string localPath = await DownloadDatasetAsync(url, $"./data/{datasetName}");
        
        return mlContext.Data.LoadFromTextFile(
            localPath,
            separatorChar: separator,
            hasHeader: hasHeader
        );
    }
}
```

### 3. Version Your Datasets

Track dataset versions alongside your code:

```csharp
public record DatasetVersion(
    string Name,
    string Version,
    string Checksum,
    DateTime Downloaded
);

// Store in datasets.json for reproducibility
```

### 4. Handle Large Datasets

For datasets exceeding memory:

```csharp
// Stream large CSV files
var mlContext = new MLContext();

// Use lazy loading with caching
var data = mlContext.Data.LoadFromTextFile<LargeData>("big_dataset.csv");
var cachedData = mlContext.Data.Cache(data); // Cache on first access

// Or use database streaming
var dbData = mlContext.Data.CreateDatabaseLoader<SalesData>()
    .Load(
        new DatabaseSource(SqlClientFactory.Instance, connectionString, query)
    );
```

---

## Quick Reference Links

| Source | URL | Best For |
|--------|-----|----------|
| UCI Repository | archive.ics.uci.edu | Classic ML datasets |
| Kaggle | kaggle.com/datasets | Variety & competitions |
| Azure Open Datasets | azure.microsoft.com/services/open-datasets | Azure integration |
| Hugging Face | huggingface.co/datasets | NLP & transformers |
| Google Dataset Search | datasetsearch.research.google.com | Discovery |
| AWS Open Data | registry.opendata.aws | Large-scale data |
| Papers With Code | paperswithcode.com/datasets | Research benchmarks |

This curated list should provide ample resources for practicing every technique covered in this book. Start with the beginner datasets to validate your implementations, then progress to intermediate and advanced datasets as your skills develop.
