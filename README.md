[![Buy on Amazon](https://img.shields.io/badge/Buy%20on-Amazon-orange?style=for-the-badge&logo=amazon)](https://www.amazon.com/dp/B0GNJBBNJG)
[![.NET](https://img.shields.io/badge/.NET-10.0-purple?style=flat-square&logo=dotnet)](https://dotnet.microsoft.com/)
[![ML.NET](https://img.shields.io/badge/ML.NET-5.0.0-blue?style=flat-square)](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

# 📚 About This Repository

This is the **official companion code repository** for the book *The C# Developer's Guide to Data Science: From App Dev to ML Engineer with .NET and Azure*.

**The book teaches experienced C# developers how to break into data science and machine learning using the tools they already know** — no Python required. If you've ever felt left out of the ML revolution because everything seems to require Python, this book is for you.

## What You'll Learn

- ✅ Data wrangling and exploratory analysis with Microsoft.Data.Analysis
- ✅ Machine learning with ML.NET 5.0 (regression, classification, clustering, recommendations)
- ✅ Deep learning with TorchSharp
- ✅ Computer vision using ONNX models
- ✅ Production deployment patterns (ASP.NET Core, Azure Functions, Docker)
- ✅ MLOps with GitHub Actions
- ✅ AI engineering with Semantic Kernel

## Get the Book

📕 **Available on Amazon Kindle and Paperback** — [Buy on Amazon](https://www.amazon.com/dp/B0GNJBBNJG)

---

## 🚀 Getting Started

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) or later
- Visual Studio 2022 / VS Code / Rider

### Clone & Build

```bash
git clone https://github.com/bspann/csharp-data-science-book.git
cd csharp-data-science-book
dotnet build
```

## 📁 Project Structure

```
code/
├── src/
│   └── DataScience.Core/       # Shared utilities and helpers
└── samples/
    ├── Chapter04.DataWrangling/
    ├── Chapter05.TitanicEDA/
    ├── Chapter06.HousingFeatures/
    ├── Chapter07.IrisClassification/
    ├── Chapter08.TaxiFarePrediction/
    ├── Chapter09.CustomerChurn/
    ├── Chapter10.CustomerSegmentation/
    ├── Chapter11.FraudDetection/
    ├── Chapter12.MovieRecommendations/
    ├── Chapter13.SentimentAnalysis/
    ├── Chapter14.SalesForecasting/
    ├── Chapter15.MnistCNN/
    ├── Chapter16.ImageClassifier/
    └── Chapter17.DeploymentAPI/
```

## 🧪 Running Samples

Each sample is a standalone console application or web API:

```bash
# Run any sample
cd code/samples/Chapter07.IrisClassification
dotnet run

# Or run the deployment API
cd code/samples/Chapter17.DeploymentAPI
dotnet run
```

## 📖 Chapter Overview

| Chapter | Topic | Sample Project |
|---------|-------|----------------|
| 4 | Data Wrangling | DataWrangling |
| 5 | Exploratory Data Analysis | TitanicEDA |
| 6 | Feature Engineering | HousingFeatures |
| 7 | Classification | IrisClassification |
| 8 | Regression | TaxiFarePrediction |
| 9 | Binary Classification | CustomerChurn |
| 10 | Clustering | CustomerSegmentation |
| 11 | Anomaly Detection | FraudDetection |
| 12 | Recommendation Systems | MovieRecommendations |
| 13 | NLP & Text Analysis | SentimentAnalysis |
| 14 | Time Series Forecasting | SalesForecasting |
| 15 | Deep Learning Basics | MnistCNN |
| 16 | Computer Vision | ImageClassifier |
| 17 | Model Deployment | DeploymentAPI |

## 📝 License

This code is released under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for personal and commercial projects.

## 🐛 Issues

Found a bug in the sample code? [Open an issue](https://github.com/bspann/csharp-data-science-book/issues).
