# The C# Developer's Guide to Data Science

**From App Dev to ML Engineer with .NET and Azure**

[![.NET](https://img.shields.io/badge/.NET-8.0+-512BD4?logo=dotnet)](https://dotnet.microsoft.com/)
[![ML.NET](https://img.shields.io/badge/ML.NET-5.0.0-blue)](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Book](https://img.shields.io/badge/📚_Book-Available_on_Amazon-orange)](https://amazon.com)

---

## 📖 About This Repository

This is the **official companion code repository** for the book *The C# Developer's Guide to Data Science: From App Dev to ML Engineer with .NET and Azure*.

**The book teaches experienced C# developers how to break into data science and machine learning using the tools they already know** — no Python required. If you've ever felt left out of the ML revolution because everything seems to require Python, this book is for you.

### What You'll Learn

- ✅ Data wrangling and exploratory analysis with Microsoft.Data.Analysis
- ✅ Machine learning with ML.NET 5.0 (regression, classification, clustering, recommendations)
- ✅ Deep learning with TorchSharp
- ✅ Computer vision using ONNX models
- ✅ Production deployment patterns (ASP.NET Core, Azure Functions, Docker)
- ✅ MLOps with GitHub Actions
- ✅ AI engineering with Semantic Kernel

### Get the Book

📕 **Available on Amazon Kindle and Paperback** — [Coming Soon]

---

## 📚 Book Overview

| Part | Chapters | Topics |
|------|----------|--------|
| **I: The Transition** | 1-3 | Why .NET for ML, environment setup, thinking like a data scientist |
| **II: Data Fundamentals** | 4-6 | Data wrangling, EDA, feature engineering |
| **III: Classic ML** | 7-12 | ML.NET, regression, classification, clustering, anomaly detection, recommendations |
| **IV: Advanced Topics** | 13-16 | NLP, time series, deep learning (TorchSharp), computer vision |
| **V: Production & MLOps** | 17-20 | Deployment, Azure ML, MLOps, AI engineering |

## 🚀 Quick Start

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later
- Visual Studio 2022 / VS Code / Rider
- (Optional) NVIDIA GPU with CUDA for deep learning chapters

### Clone and Build

```bash
git clone https://github.com/YOUR_USERNAME/csharp-data-science-book.git
cd csharp-data-science-book
dotnet restore
dotnet build
```

### Run a Sample

```bash
cd code/samples/Chapter07.IrisClassification
dotnet run
```

## 📁 Repository Structure

```
├── chapters/           # Book manuscript (Markdown)
├── code/
│   ├── src/           # Shared libraries
│   │   ├── DataScience.Core/        # Common utilities
│   │   └── DataScience.MLExtensions/ # ML.NET extensions
│   └── samples/       # Chapter code samples
│       ├── Chapter04.DataWrangling/
│       ├── Chapter07.IrisClassification/
│       ├── Chapter08.TaxiFarePrediction/
│       ├── Chapter09.CustomerChurn/
│       ├── Chapter10.CustomerSegmentation/
│       ├── Chapter11.FraudDetection/
│       ├── Chapter12.MovieRecommendations/
│       ├── Chapter13.SentimentAnalysis/
│       ├── Chapter14.SalesForecasting/
│       ├── Chapter15.MnistCNN/
│       ├── Chapter16.ImageClassifier/
│       └── Chapter17.DeploymentAPI/
├── data/              # Sample datasets
├── notebooks/         # Polyglot Notebooks (.dib)
└── .github/workflows/ # CI/CD pipelines
```

## 📦 NuGet Packages Used

| Package | Version | Purpose |
|---------|---------|---------|
| Microsoft.ML | 5.0.0 | Core ML.NET framework |
| Microsoft.Data.Analysis | 0.22.0 | DataFrame operations |
| Microsoft.ML.AutoML | 0.22.0 | Automated ML |
| Microsoft.ML.Recommender | 0.22.0 | Recommendation systems |
| Microsoft.ML.TimeSeries | 5.0.0 | Time series forecasting |
| Microsoft.ML.OnnxTransformer | 5.0.0 | ONNX model integration |
| TorchSharp | 0.102.* | Deep learning |
| ScottPlot | 5.* | Data visualization |

## 🧪 Running Tests

```bash
dotnet test
```

## 📖 Chapter Projects

Each chapter includes hands-on projects:

| Chapter | Project | Description |
|---------|---------|-------------|
| 4 | Melbourne Housing | Data cleaning pipeline |
| 5 | Titanic EDA | Exploratory data analysis |
| 6 | Housing Features | Feature engineering |
| 7 | Iris Classification | First ML.NET model |
| 8 | NYC Taxi Fares | Regression prediction |
| 9 | Customer Churn | Binary classification |
| 10 | E-commerce Segmentation | K-Means clustering |
| 11 | Fraud Detection | Anomaly detection |
| 12 | Movie Recommendations | Matrix factorization |
| 13 | Product Reviews | Sentiment analysis |
| 14 | Retail Sales | Time series forecasting |
| 15 | MNIST Digits | CNN with TorchSharp |
| 16 | Photo Classifier | Transfer learning with ONNX |
| 17 | Sentiment API | Production deployment |

## 🤝 Contributing

Found a bug or want to improve the code? Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b fix/chapter-8-typo`)
3. Commit your changes (`git commit -am 'Fix typo in Chapter 8'`)
4. Push to the branch (`git push origin fix/chapter-8-typo`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Microsoft ML.NET team for the excellent framework
- The [ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners) curriculum for inspiration
- The .NET community

---

**Happy learning!** 🎓

If you find this book helpful, please ⭐ the repository!
