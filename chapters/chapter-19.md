# Chapter 19: MLOps for .NET Developers

You've trained a model. It works beautifully on your laptop. The accuracy metrics look great, your team is excited, and the business stakeholders are ready to see it in action.

Now what?

This is where most data science projects go to die. The gap between "working model" and "model that reliably delivers business value in production" is enormous. It's where the careful work of software engineering—versioning, testing, deployment, monitoring—meets the unique challenges of machine learning systems.

Welcome to MLOps.

If you've done DevOps, you'll recognize the patterns. But ML systems have properties that make them fundamentally harder to manage than traditional software. The code might be identical between two deployments, yet the behavior is completely different because the model weights changed. A system that worked perfectly last month might silently degrade because the input data distribution shifted. Your "same" model might produce different results on different hardware due to floating-point variations.

The good news? Your C# and .NET background gives you a head start. You already think about CI/CD, versioning, and monitoring. You understand dependency injection and testable code. You know how to build systems that are observable and maintainable.

This chapter will teach you to apply those instincts to ML systems—and avoid the pitfalls that catch teams who treat ML as "just deploy the pickle file."

## Why ML Systems Are Different

Before we dive into practices, let's understand what makes ML operations uniquely challenging.

### The Three Axes of Change

Traditional software changes along one axis: code. When you deploy a new version, the code is different, and that's the source of any behavioral changes.

ML systems change along three axes simultaneously:

1. **Code**: Your training pipeline, feature engineering, serving logic
2. **Model**: The trained weights and parameters
3. **Data**: The inputs your model receives and was trained on

[FIGURE: Diagram showing three axes of change - Code, Model, and Data - converging on "ML System Behavior" in the center]

Any of these can cause your system to behave differently. A bug might be in the code, the model, or caused by unexpected data. Debugging requires tracking all three.

### Silent Failures

Traditional software fails loudly. An exception throws, a service returns 500, a test fails. You know something is wrong.

ML systems fail silently. Your model is still returning predictions—they're just *wrong*. The fraud detection model is approving fraudulent transactions. The recommendation engine is showing irrelevant products. The forecast is consistently off by 30%.

Nothing in your monitoring breaks. The API responds quickly. The health checks pass. But the business impact is devastating.

### Data Dependencies

Your model is a function of its training data. If that data was biased, stale, or corrupted, your model inherits those problems. Unlike code dependencies, which you can pin to specific versions and audit, data dependencies are often invisible and constantly shifting.

Consider this scenario: your model was trained on customer data from 2024. It's now 2026, and customer behavior has changed. The model has never seen a post-pandemic shopping pattern, a new product category, or the current economic conditions. It's making predictions based on a world that no longer exists.

This is called **concept drift**, and it happens to every ML model in production. The question isn't whether your model will drift—it's whether you'll detect it before it causes damage.

### Reproducibility Challenges

"It worked on my machine" is a joke in software engineering. In ML, it's a daily reality.

Reproducing an ML experiment requires:
- Exact same code version
- Exact same data (including preprocessing)
- Exact same random seeds
- Same library versions (down to patch level)
- Often, same hardware (GPU results can differ from CPU)

Teams routinely find they cannot reproduce their own results from three months ago. This makes debugging, comparison, and regulatory compliance nightmarish.

## CI/CD for ML Models

Let's start with continuous integration and deployment. The principles are familiar, but the implementation differs.

### What to Test in ML Pipelines

Your ML CI pipeline should test at multiple levels:

**Unit Tests**: Test individual functions in your feature engineering and data processing code.

```csharp
[Fact]
public void NormalizeFeatures_WithValidInput_ReturnsNormalizedValues()
{
    var input = new FeatureVector { Values = new[] { 10f, 20f, 30f } };
    var result = FeatureProcessor.Normalize(input, mean: 20f, stdDev: 10f);
    
    Assert.Equal(-1f, result.Values[0], precision: 5);
    Assert.Equal(0f, result.Values[1], precision: 5);
    Assert.Equal(1f, result.Values[2], precision: 5);
}
```

**Data Validation Tests**: Verify that input data meets expected schema and distribution.

```csharp
[Fact]
public void TrainingData_MeetsQualityThresholds()
{
    var data = DataLoader.LoadTrainingData();
    
    Assert.True(data.Count >= 10000, "Insufficient training examples");
    Assert.True(data.All(r => r.Label != null), "Missing labels detected");
    Assert.True(data.NullPercentage("price") < 0.01, "Too many null prices");
    
    var labelDistribution = data.GroupBy(r => r.Label).ToDictionary(g => g.Key, g => g.Count());
    Assert.True(labelDistribution.Values.Min() > 100, "Class imbalance too severe");
}
```

**Model Quality Tests**: Verify that a trained model meets minimum performance thresholds.

```csharp
[Fact]
public void Model_MeetsMinimumAccuracy()
{
    var model = ModelLoader.LoadLatest();
    var testData = DataLoader.LoadTestSet();
    
    var metrics = ModelEvaluator.Evaluate(model, testData);
    
    Assert.True(metrics.Accuracy > 0.85, $"Accuracy {metrics.Accuracy} below threshold");
    Assert.True(metrics.F1Score > 0.80, $"F1 {metrics.F1Score} below threshold");
    Assert.True(metrics.AUC > 0.90, $"AUC {metrics.AUC} below threshold");
}
```

**Integration Tests**: Test the full prediction pipeline end-to-end.

```csharp
[Fact]
public async Task PredictionEndpoint_ReturnsValidPrediction()
{
    var client = _factory.CreateClient();
    var request = new PredictionRequest 
    { 
        Features = new Dictionary<string, object>
        {
            ["age"] = 35,
            ["income"] = 75000,
            ["tenure_months"] = 24
        }
    };
    
    var response = await client.PostAsJsonAsync("/api/predict", request);
    
    response.EnsureSuccessStatusCode();
    var result = await response.Content.ReadFromJsonAsync<PredictionResponse>();
    Assert.InRange(result.Probability, 0, 1);
    Assert.NotNull(result.ModelVersion);
}
```

**Performance Tests**: Ensure inference meets latency requirements.

```csharp
[Fact]
public void Inference_MeetsLatencyRequirements()
{
    var model = ModelLoader.LoadLatest();
    var warmupInput = GenerateRandomInput();
    
    // Warm up the model
    for (int i = 0; i < 100; i++)
        model.Predict(warmupInput);
    
    var inputs = Enumerable.Range(0, 1000).Select(_ => GenerateRandomInput()).ToList();
    var sw = Stopwatch.StartNew();
    
    foreach (var input in inputs)
        model.Predict(input);
    
    sw.Stop();
    var avgLatencyMs = sw.ElapsedMilliseconds / 1000.0;
    
    Assert.True(avgLatencyMs < 10, $"Average latency {avgLatencyMs}ms exceeds 10ms threshold");
}
```

### The Training Pipeline

Your training pipeline should be automated and reproducible. Here's a structure that works well for .NET projects:

```
/src
  /Training
    /DataPreparation
    /FeatureEngineering
    /Training
    /Evaluation
  /Inference
    /Api
    /ModelLoader
/models
  /v1.0.0
    model.zip
    metrics.json
    config.json
/pipelines
  train.yaml
  deploy.yaml
```

The training code itself should be a console application that can run in CI:

```csharp
public class TrainingPipeline
{
    private readonly MLContext _mlContext;
    private readonly IConfiguration _config;
    private readonly ILogger<TrainingPipeline> _logger;

    public async Task<TrainingResult> RunAsync(TrainingOptions options)
    {
        _logger.LogInformation("Starting training run {RunId}", options.RunId);
        
        // Load data
        var rawData = await LoadDataAsync(options.DataPath);
        _logger.LogInformation("Loaded {Count} training examples", rawData.Count());
        
        // Split data
        var split = _mlContext.Data.TrainTestSplit(rawData, testFraction: 0.2);
        
        // Build pipeline
        var pipeline = BuildPipeline(options);
        
        // Train
        var stopwatch = Stopwatch.StartNew();
        var model = pipeline.Fit(split.TrainSet);
        stopwatch.Stop();
        
        _logger.LogInformation("Training completed in {Duration}", stopwatch.Elapsed);
        
        // Evaluate
        var predictions = model.Transform(split.TestSet);
        var metrics = _mlContext.BinaryClassification.Evaluate(predictions);
        
        // Save model
        var modelPath = Path.Combine(options.OutputPath, "model.zip");
        _mlContext.Model.Save(model, rawData.Schema, modelPath);
        
        // Save metrics
        var result = new TrainingResult
        {
            RunId = options.RunId,
            ModelPath = modelPath,
            Accuracy = metrics.Accuracy,
            AUC = metrics.AreaUnderRocCurve,
            F1Score = metrics.F1Score,
            TrainingDuration = stopwatch.Elapsed,
            Timestamp = DateTime.UtcNow
        };
        
        await SaveMetricsAsync(result, options.OutputPath);
        
        return result;
    }
}
```

### GitHub Actions for ML Training

Here's a complete GitHub Actions workflow for training and evaluating an ML model:

```yaml
name: ML Training Pipeline

on:
  push:
    paths:
      - 'src/Training/**'
      - 'data/training/**'
  workflow_dispatch:
    inputs:
      experiment_name:
        description: 'Experiment name for tracking'
        required: true
        default: 'manual-run'

env:
  DOTNET_VERSION: '8.0.x'
  MODEL_REGISTRY: 'models'

jobs:
  validate-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Run Data Validation
        run: dotnet test src/Training.Tests --filter Category=DataValidation
      
      - name: Upload Data Quality Report
        uses: actions/upload-artifact@v4
        with:
          name: data-quality-report
          path: reports/data-quality.json

  train:
    needs: validate-data
    runs-on: ubuntu-latest
    outputs:
      model_version: ${{ steps.version.outputs.version }}
      metrics_passed: ${{ steps.evaluate.outputs.passed }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Generate Version
        id: version
        run: |
          VERSION="v$(date +%Y%m%d)-${{ github.run_number }}"
          echo "version=$VERSION" >> $GITHUB_OUTPUT
          echo "Generated version: $VERSION"
      
      - name: Download Training Data
        run: |
          dotnet run --project src/DataDownloader -- \
            --output data/training \
            --date-range "last-30-days"
      
      - name: Train Model
        run: |
          dotnet run --project src/Training -- \
            --run-id ${{ steps.version.outputs.version }} \
            --data-path data/training \
            --output-path ${{ env.MODEL_REGISTRY }}/${{ steps.version.outputs.version }} \
            --experiment "${{ github.event.inputs.experiment_name || 'ci-run' }}"
      
      - name: Evaluate Model
        id: evaluate
        run: |
          RESULT=$(dotnet run --project src/Evaluation -- \
            --model-path ${{ env.MODEL_REGISTRY }}/${{ steps.version.outputs.version }}/model.zip \
            --test-data data/test \
            --min-accuracy 0.85 \
            --min-f1 0.80)
          echo "passed=$RESULT" >> $GITHUB_OUTPUT
      
      - name: Upload Model Artifact
        uses: actions/upload-artifact@v4
        with:
          name: model-${{ steps.version.outputs.version }}
          path: |
            ${{ env.MODEL_REGISTRY }}/${{ steps.version.outputs.version }}/
      
      - name: Upload Training Metrics
        uses: actions/upload-artifact@v4
        with:
          name: metrics-${{ steps.version.outputs.version }}
          path: ${{ env.MODEL_REGISTRY }}/${{ steps.version.outputs.version }}/metrics.json

  register-model:
    needs: train
    if: needs.train.outputs.metrics_passed == 'true'
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Download Model
        uses: actions/download-artifact@v4
        with:
          name: model-${{ needs.train.outputs.model_version }}
          path: model
      
      - name: Register Model in Registry
        run: |
          dotnet run --project src/ModelRegistry -- register \
            --model-path model/model.zip \
            --version ${{ needs.train.outputs.model_version }} \
            --metrics-path model/metrics.json \
            --stage "staging"
      
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: model-${{ needs.train.outputs.model_version }}
          files: |
            model/model.zip
            model/metrics.json
          body: |
            ## Model Release ${{ needs.train.outputs.model_version }}
            
            Automated training run from commit ${{ github.sha }}
            
            See metrics.json for evaluation results.
```

This workflow validates data quality, trains the model, evaluates performance against thresholds, and registers successful models automatically.

## Model Versioning and Tracking

Every deployed model needs an identity. You need to know:

- Which version is running in production
- What code and data produced it
- How it performed during training
- When it was deployed
- Who approved the deployment

### Version Scheme for Models

I recommend a composite version scheme:

```
v{YYYYMMDD}-{build_number}-{git_short_sha}
```

For example: `v20260214-42-a3b2c1d`

This gives you:
- Date context (when was this trained?)
- Build ordering (which came first?)
- Code traceability (what code produced this?)

Store this version with the model:

```csharp
public class ModelMetadata
{
    public string Version { get; set; }
    public string GitCommit { get; set; }
    public DateTime TrainedAt { get; set; }
    public string TrainingDataPath { get; set; }
    public string TrainingDataHash { get; set; }
    public Dictionary<string, double> Metrics { get; set; }
    public Dictionary<string, object> Hyperparameters { get; set; }
    public string Framework { get; set; }
    public string[] Dependencies { get; set; }
}
```

### Model Registry

A model registry is the central repository for your trained models. Think of it like NuGet, but for ML models.

Here's a simple file-based registry for smaller teams:

```csharp
public class FileModelRegistry : IModelRegistry
{
    private readonly string _basePath;
    
    public async Task RegisterAsync(ModelArtifact artifact)
    {
        var versionPath = Path.Combine(_basePath, artifact.Metadata.Version);
        Directory.CreateDirectory(versionPath);
        
        // Copy model
        File.Copy(artifact.ModelPath, Path.Combine(versionPath, "model.zip"));
        
        // Save metadata
        var metadataJson = JsonSerializer.Serialize(artifact.Metadata, _jsonOptions);
        await File.WriteAllTextAsync(
            Path.Combine(versionPath, "metadata.json"), 
            metadataJson);
        
        // Update index
        await UpdateIndexAsync(artifact.Metadata);
    }
    
    public async Task<ModelArtifact> GetAsync(string version)
    {
        var versionPath = Path.Combine(_basePath, version);
        
        var metadataJson = await File.ReadAllTextAsync(
            Path.Combine(versionPath, "metadata.json"));
        var metadata = JsonSerializer.Deserialize<ModelMetadata>(metadataJson, _jsonOptions);
        
        return new ModelArtifact
        {
            ModelPath = Path.Combine(versionPath, "model.zip"),
            Metadata = metadata
        };
    }
    
    public async Task<ModelArtifact> GetLatestAsync(string stage = "production")
    {
        var index = await LoadIndexAsync();
        var latest = index.Models
            .Where(m => m.Stage == stage)
            .OrderByDescending(m => m.TrainedAt)
            .FirstOrDefault();
            
        return latest != null ? await GetAsync(latest.Version) : null;
    }
    
    public async Task PromoteAsync(string version, string fromStage, string toStage)
    {
        var index = await LoadIndexAsync();
        var entry = index.Models.First(m => m.Version == version);
        
        if (entry.Stage != fromStage)
            throw new InvalidOperationException(
                $"Model {version} is in stage {entry.Stage}, not {fromStage}");
        
        entry.Stage = toStage;
        entry.PromotedAt = DateTime.UtcNow;
        
        await SaveIndexAsync(index);
    }
}
```

For larger teams, use Azure ML's model registry, MLflow, or a similar managed service.

### Tracking Experiments

Every training run should be logged as an experiment. Here's a lightweight tracking approach:

```csharp
public class ExperimentTracker
{
    private readonly string _experimentName;
    private readonly string _runId;
    private readonly Dictionary<string, object> _params = new();
    private readonly Dictionary<string, double> _metrics = new();
    private readonly List<string> _artifacts = new();

    public void LogParameter(string name, object value)
    {
        _params[name] = value;
    }

    public void LogMetric(string name, double value)
    {
        _metrics[name] = value;
    }

    public void LogArtifact(string path)
    {
        _artifacts.Add(path);
    }

    public async Task SaveAsync(string outputPath)
    {
        var run = new ExperimentRun
        {
            ExperimentName = _experimentName,
            RunId = _runId,
            StartTime = _startTime,
            EndTime = DateTime.UtcNow,
            Parameters = _params,
            Metrics = _metrics,
            Artifacts = _artifacts,
            GitCommit = GetGitCommit(),
            GitBranch = GetGitBranch()
        };

        var json = JsonSerializer.Serialize(run, _jsonOptions);
        await File.WriteAllTextAsync(
            Path.Combine(outputPath, "experiment.json"), 
            json);
    }
}
```

Usage in training:

```csharp
var tracker = new ExperimentTracker("fraud-detection", runId);

tracker.LogParameter("algorithm", "FastTree");
tracker.LogParameter("learning_rate", 0.1);
tracker.LogParameter("num_trees", 100);
tracker.LogParameter("training_data_rows", trainingData.Count());

// ... training code ...

tracker.LogMetric("accuracy", metrics.Accuracy);
tracker.LogMetric("auc", metrics.AreaUnderRocCurve);
tracker.LogMetric("f1", metrics.F1Score);
tracker.LogMetric("training_duration_seconds", stopwatch.Elapsed.TotalSeconds);

tracker.LogArtifact(modelPath);
await tracker.SaveAsync(outputPath);
```

[FIGURE: Screenshot of an experiment tracking dashboard showing multiple runs with their parameters and metrics in a comparison table]

## A/B Testing Models in Production

Deploying a new model shouldn't be all-or-nothing. A/B testing lets you gradually roll out changes and measure real-world impact before committing.

### Traffic Splitting Architecture

```csharp
public class ModelRouter
{
    private readonly IModelRegistry _registry;
    private readonly ITrafficSplitter _splitter;
    private readonly Dictionary<string, PredictionEngine> _engines = new();

    public async Task<PredictionResult> PredictAsync(PredictionRequest request)
    {
        // Determine which model to use based on traffic rules
        var assignment = _splitter.GetAssignment(request.UserId);
        
        // Get or create prediction engine for this model version
        var engine = await GetOrCreateEngineAsync(assignment.ModelVersion);
        
        // Make prediction
        var prediction = engine.Predict(request.ToInput());
        
        // Log the assignment for analysis
        await LogPredictionAsync(new PredictionLog
        {
            RequestId = request.RequestId,
            UserId = request.UserId,
            ModelVersion = assignment.ModelVersion,
            ExperimentId = assignment.ExperimentId,
            Prediction = prediction,
            Timestamp = DateTime.UtcNow
        });
        
        return new PredictionResult
        {
            Prediction = prediction,
            ModelVersion = assignment.ModelVersion
        };
    }
}

public class TrafficSplitter : ITrafficSplitter
{
    private readonly List<TrafficRule> _rules;

    public TrafficAssignment GetAssignment(string userId)
    {
        // Consistent hashing ensures same user always gets same model
        var hash = ComputeHash(userId);
        var bucket = hash % 100;
        
        foreach (var rule in _rules.OrderBy(r => r.Priority))
        {
            if (bucket < rule.CumulativePercentage)
            {
                return new TrafficAssignment
                {
                    ModelVersion = rule.ModelVersion,
                    ExperimentId = rule.ExperimentId
                };
            }
        }
        
        return new TrafficAssignment { ModelVersion = "default" };
    }
}
```

### Configuration-Driven Experiments

Store experiment configurations separately from code:

```json
{
  "experiments": [
    {
      "id": "exp-20260214-new-features",
      "name": "Test new temporal features",
      "status": "running",
      "startDate": "2026-02-14T00:00:00Z",
      "variants": [
        {
          "name": "control",
          "modelVersion": "v20260201-38-b4c5d6e",
          "trafficPercentage": 80
        },
        {
          "name": "treatment",
          "modelVersion": "v20260214-42-a3b2c1d",
          "trafficPercentage": 20
        }
      ],
      "successMetrics": ["conversion_rate", "revenue_per_user"],
      "guardrailMetrics": ["error_rate", "latency_p99"]
    }
  ]
}
```

### Analyzing Experiment Results

```csharp
public class ExperimentAnalyzer
{
    public async Task<ExperimentResults> AnalyzeAsync(string experimentId)
    {
        var logs = await _logStore.GetPredictionLogsAsync(experimentId);
        var outcomes = await _outcomeStore.GetOutcomesAsync(experimentId);
        
        // Join predictions with outcomes
        var joined = from log in logs
                     join outcome in outcomes on log.RequestId equals outcome.RequestId
                     select new { log, outcome };
        
        var byVariant = joined.GroupBy(x => x.log.ModelVersion);
        
        var results = new ExperimentResults { ExperimentId = experimentId };
        
        foreach (var variant in byVariant)
        {
            var variantData = variant.ToList();
            
            results.Variants.Add(new VariantResult
            {
                ModelVersion = variant.Key,
                SampleSize = variantData.Count,
                ConversionRate = variantData.Average(x => x.outcome.Converted ? 1.0 : 0.0),
                RevenuePerUser = variantData.Average(x => x.outcome.Revenue),
                AverageLatencyMs = variantData.Average(x => x.log.LatencyMs)
            });
        }
        
        // Statistical significance testing
        results.StatisticalAnalysis = ComputeSignificance(results.Variants);
        
        return results;
    }
    
    private StatisticalAnalysis ComputeSignificance(List<VariantResult> variants)
    {
        var control = variants.First(v => v.ModelVersion.Contains("control") || v.IsBaseline);
        var treatment = variants.First(v => v != control);
        
        // Chi-squared test for conversion rate
        var chiSquared = ComputeChiSquared(
            control.ConversionRate, control.SampleSize,
            treatment.ConversionRate, treatment.SampleSize);
        
        return new StatisticalAnalysis
        {
            PValue = chiSquared.PValue,
            IsSignificant = chiSquared.PValue < 0.05,
            Lift = (treatment.ConversionRate - control.ConversionRate) / control.ConversionRate,
            ConfidenceInterval = chiSquared.ConfidenceInterval
        };
    }
}
```

### GitHub Actions for A/B Test Management

```yaml
name: A/B Test Deployment

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to test'
        required: true
      traffic_percentage:
        description: 'Percentage of traffic (1-50)'
        required: true
        default: '10'
      experiment_name:
        description: 'Experiment name'
        required: true

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Validate Model Exists
        run: |
          dotnet run --project src/ModelRegistry -- exists \
            --version ${{ github.event.inputs.model_version }}
      
      - name: Validate Traffic Percentage
        run: |
          TRAFFIC=${{ github.event.inputs.traffic_percentage }}
          if [ "$TRAFFIC" -gt 50 ]; then
            echo "Error: Initial traffic percentage cannot exceed 50%"
            exit 1
          fi

  deploy-experiment:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Create Experiment Config
        run: |
          cat > experiment.json << EOF
          {
            "id": "exp-$(date +%Y%m%d)-${{ github.run_number }}",
            "name": "${{ github.event.inputs.experiment_name }}",
            "treatment_model": "${{ github.event.inputs.model_version }}",
            "traffic_percentage": ${{ github.event.inputs.traffic_percentage }},
            "created_by": "${{ github.actor }}",
            "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
          }
          EOF
      
      - name: Deploy Experiment
        run: |
          dotnet run --project src/ExperimentManager -- deploy \
            --config experiment.json \
            --environment production
      
      - name: Update Traffic Rules
        run: |
          dotnet run --project src/TrafficManager -- update \
            --experiment-config experiment.json
```

## Monitoring Model Performance

A model that works today might fail tomorrow. Continuous monitoring is essential.

### The Four Pillars of ML Monitoring

**1. Operational Metrics**: Is the system running?
- Request latency (p50, p95, p99)
- Throughput (requests/second)
- Error rates
- Resource utilization

**2. Data Quality Metrics**: Is the input data valid?
- Schema violations
- Missing values
- Out-of-range values
- Unexpected categories

**3. Model Performance Metrics**: Is the model accurate?
- Prediction distribution shifts
- Accuracy on labeled samples
- Confidence score distributions

**4. Business Metrics**: Is the model delivering value?
- Conversion rates
- Revenue impact
- User engagement
- Error costs

### Detecting Data Drift

Data drift occurs when the distribution of input features changes over time. Here's how to detect it:

```csharp
public class DriftDetector
{
    private readonly FeatureStatistics _baseline;
    
    public DriftReport Detect(IEnumerable<FeatureVector> currentData)
    {
        var report = new DriftReport { Timestamp = DateTime.UtcNow };
        var current = ComputeStatistics(currentData);
        
        foreach (var feature in _baseline.Features)
        {
            var baselineStats = _baseline.GetStats(feature);
            var currentStats = current.GetStats(feature);
            
            // Population Stability Index (PSI)
            var psi = ComputePSI(baselineStats.Distribution, currentStats.Distribution);
            
            // Kolmogorov-Smirnov test for continuous features
            var ksResult = KolmogorovSmirnovTest(
                baselineStats.Values, 
                currentStats.Values);
            
            report.Features.Add(new FeatureDriftResult
            {
                FeatureName = feature,
                PSI = psi,
                KSStatistic = ksResult.Statistic,
                KSPValue = ksResult.PValue,
                DriftDetected = psi > 0.2 || ksResult.PValue < 0.01,
                BaselineMean = baselineStats.Mean,
                CurrentMean = currentStats.Mean,
                MeanShift = (currentStats.Mean - baselineStats.Mean) / baselineStats.StdDev
            });
        }
        
        report.OverallDriftScore = report.Features.Average(f => f.PSI);
        report.RequiresAttention = report.Features.Any(f => f.DriftDetected);
        
        return report;
    }
    
    private double ComputePSI(double[] baseline, double[] current)
    {
        // PSI = Σ (current% - baseline%) * ln(current% / baseline%)
        double psi = 0;
        for (int i = 0; i < baseline.Length; i++)
        {
            var b = Math.Max(baseline[i], 0.0001);
            var c = Math.Max(current[i], 0.0001);
            psi += (c - b) * Math.Log(c / b);
        }
        return psi;
    }
}
```

### Model Decay Detection

Even without data drift, models can decay. Here's how to track prediction quality over time:

```csharp
public class ModelDecayMonitor
{
    private readonly ILabelCollector _labelCollector;
    
    public async Task<DecayReport> CheckDecayAsync(string modelVersion, TimeSpan window)
    {
        // Get predictions from the time window
        var predictions = await _predictionStore.GetPredictionsAsync(
            modelVersion, 
            DateTime.UtcNow - window, 
            DateTime.UtcNow);
        
        // Get actual outcomes (labels) for these predictions
        var withLabels = await _labelCollector.JoinLabelsAsync(predictions);
        
        // Compute current metrics
        var currentMetrics = ComputeMetrics(withLabels);
        
        // Compare with baseline (training metrics)
        var baseline = await _registry.GetMetricsAsync(modelVersion);
        
        return new DecayReport
        {
            ModelVersion = modelVersion,
            WindowStart = DateTime.UtcNow - window,
            WindowEnd = DateTime.UtcNow,
            SampleSize = withLabels.Count,
            CurrentAccuracy = currentMetrics.Accuracy,
            BaselineAccuracy = baseline.Accuracy,
            AccuracyDrop = baseline.Accuracy - currentMetrics.Accuracy,
            CurrentF1 = currentMetrics.F1Score,
            BaselineF1 = baseline.F1Score,
            F1Drop = baseline.F1Score - currentMetrics.F1Score,
            DecayDetected = currentMetrics.Accuracy < baseline.Accuracy * 0.95
        };
    }
}
```

### Alerting Pipeline

```csharp
public class MLAlertingService : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await CheckDriftAsync();
                await CheckDecayAsync();
                await CheckOperationalMetricsAsync();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in monitoring loop");
            }
            
            await Task.Delay(TimeSpan.FromMinutes(15), stoppingToken);
        }
    }
    
    private async Task CheckDriftAsync()
    {
        var report = await _driftDetector.DetectAsync();
        
        if (report.OverallDriftScore > 0.25)
        {
            await _alertService.SendAlertAsync(new Alert
            {
                Severity = AlertSeverity.Critical,
                Title = "Significant Data Drift Detected",
                Message = $"PSI score {report.OverallDriftScore:F3} exceeds threshold. " +
                         $"Features with drift: {string.Join(", ", report.DriftedFeatures)}",
                Metadata = new Dictionary<string, object>
                {
                    ["drift_report"] = report
                }
            });
        }
        else if (report.RequiresAttention)
        {
            await _alertService.SendAlertAsync(new Alert
            {
                Severity = AlertSeverity.Warning,
                Title = "Minor Data Drift Detected",
                Message = $"Some features showing drift: {string.Join(", ", report.DriftedFeatures)}"
            });
        }
    }
}
```

[FIGURE: Dashboard showing drift detection metrics with time series charts for PSI scores across features, with a red threshold line and annotations where retraining was triggered]

## Retraining Pipelines

When drift or decay is detected, you need to retrain. This should be automated.

### Trigger-Based Retraining

```csharp
public class RetrainingOrchestrator
{
    public async Task EvaluateRetrainingAsync()
    {
        var triggers = new List<RetrainingTrigger>();
        
        // Check scheduled retraining
        var lastTraining = await _registry.GetLastTrainingDateAsync();
        if (DateTime.UtcNow - lastTraining > TimeSpan.FromDays(30))
        {
            triggers.Add(new RetrainingTrigger 
            { 
                Type = TriggerType.Scheduled,
                Reason = "30 days since last training"
            });
        }
        
        // Check drift-based retraining
        var driftReport = await _driftDetector.DetectAsync();
        if (driftReport.OverallDriftScore > 0.2)
        {
            triggers.Add(new RetrainingTrigger
            {
                Type = TriggerType.DataDrift,
                Reason = $"PSI score {driftReport.OverallDriftScore:F3}",
                Details = driftReport
            });
        }
        
        // Check performance-based retraining
        var decayReport = await _decayMonitor.CheckDecayAsync(
            await _registry.GetProductionVersionAsync(),
            TimeSpan.FromDays(7));
            
        if (decayReport.AccuracyDrop > 0.05)
        {
            triggers.Add(new RetrainingTrigger
            {
                Type = TriggerType.PerformanceDecay,
                Reason = $"Accuracy dropped {decayReport.AccuracyDrop:P1}",
                Details = decayReport
            });
        }
        
        // Check data volume trigger
        var newDataCount = await _dataStore.CountNewRecordsAsync(lastTraining);
        if (newDataCount > 100000)
        {
            triggers.Add(new RetrainingTrigger
            {
                Type = TriggerType.DataVolume,
                Reason = $"{newDataCount:N0} new training examples available"
            });
        }
        
        if (triggers.Any())
        {
            await InitiateRetrainingAsync(triggers);
        }
    }
    
    private async Task InitiateRetrainingAsync(List<RetrainingTrigger> triggers)
    {
        _logger.LogInformation("Initiating retraining due to: {Triggers}", 
            string.Join(", ", triggers.Select(t => t.Type)));
        
        // Start retraining workflow
        await _workflowClient.StartWorkflowAsync("retraining-pipeline", new
        {
            Triggers = triggers,
            InitiatedAt = DateTime.UtcNow,
            AutoDeploy = triggers.All(t => t.Type != TriggerType.PerformanceDecay)
        });
    }
}
```

### GitHub Actions Retraining Workflow

```yaml
name: Automated Retraining Pipeline

on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  repository_dispatch:
    types: [trigger-retraining]
  workflow_dispatch:
    inputs:
      reason:
        description: 'Reason for manual retraining'
        required: true

env:
  DOTNET_VERSION: '8.0.x'
  SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}

jobs:
  prepare-data:
    runs-on: ubuntu-latest
    outputs:
      data_hash: ${{ steps.hash.outputs.hash }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Fetch Latest Data
        run: |
          dotnet run --project src/DataPipeline -- \
            --mode incremental \
            --output data/training \
            --include-feedback
      
      - name: Compute Data Hash
        id: hash
        run: |
          HASH=$(find data/training -type f -exec sha256sum {} \; | sort | sha256sum | cut -d' ' -f1)
          echo "hash=$HASH" >> $GITHUB_OUTPUT
      
      - name: Upload Training Data
        uses: actions/upload-artifact@v4
        with:
          name: training-data
          path: data/training

  train-candidate:
    needs: prepare-data
    runs-on: ubuntu-latest
    outputs:
      model_version: ${{ steps.train.outputs.version }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Download Training Data
        uses: actions/download-artifact@v4
        with:
          name: training-data
          path: data/training
      
      - name: Train Model
        id: train
        run: |
          VERSION="v$(date +%Y%m%d)-${{ github.run_number }}"
          
          dotnet run --project src/Training -- \
            --run-id $VERSION \
            --data-path data/training \
            --output-path models/$VERSION \
            --data-hash ${{ needs.prepare-data.outputs.data_hash }}
          
          echo "version=$VERSION" >> $GITHUB_OUTPUT
      
      - name: Upload Model
        uses: actions/upload-artifact@v4
        with:
          name: candidate-model
          path: models/${{ steps.train.outputs.version }}

  evaluate-candidate:
    needs: train-candidate
    runs-on: ubuntu-latest
    outputs:
      approved: ${{ steps.compare.outputs.approved }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Download Candidate Model
        uses: actions/download-artifact@v4
        with:
          name: candidate-model
          path: candidate
      
      - name: Get Production Model
        run: |
          dotnet run --project src/ModelRegistry -- download \
            --stage production \
            --output production
      
      - name: Compare Models
        id: compare
        run: |
          RESULT=$(dotnet run --project src/ModelComparison -- \
            --candidate candidate/model.zip \
            --production production/model.zip \
            --test-data data/test \
            --min-improvement -0.01)  # Allow up to 1% regression
          
          echo "approved=$RESULT" >> $GITHUB_OUTPUT
      
      - name: Generate Comparison Report
        run: |
          dotnet run --project src/ModelComparison -- report \
            --output comparison-report.md
      
      - name: Upload Report
        uses: actions/upload-artifact@v4
        with:
          name: comparison-report
          path: comparison-report.md

  deploy-candidate:
    needs: [train-candidate, evaluate-candidate]
    if: needs.evaluate-candidate.outputs.approved == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Download Candidate Model
        uses: actions/download-artifact@v4
        with:
          name: candidate-model
          path: candidate
      
      - name: Register Model
        run: |
          dotnet run --project src/ModelRegistry -- register \
            --model-path candidate/model.zip \
            --version ${{ needs.train-candidate.outputs.model_version }} \
            --stage staging
      
      - name: Deploy to Staging
        run: |
          dotnet run --project src/Deploy -- \
            --model-version ${{ needs.train-candidate.outputs.model_version }} \
            --environment staging
      
      - name: Run Smoke Tests
        run: |
          dotnet test src/SmokeTests --filter Environment=Staging
      
      - name: Promote to Production (10% traffic)
        run: |
          dotnet run --project src/TrafficManager -- \
            --model-version ${{ needs.train-candidate.outputs.model_version }} \
            --traffic-percentage 10 \
            --environment production
      
      - name: Notify Team
        run: |
          curl -X POST $SLACK_WEBHOOK \
            -H 'Content-type: application/json' \
            -d '{
              "text": "🚀 New model deployed to production",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Model ${{ needs.train-candidate.outputs.model_version }}*\nDeployed with 10% traffic. Monitor dashboards for performance."
                  }
                }
              ]
            }'

  notify-failure:
    needs: [evaluate-candidate]
    if: needs.evaluate-candidate.outputs.approved == 'false'
    runs-on: ubuntu-latest
    steps:
      - name: Notify Team of Regression
        run: |
          curl -X POST ${{ env.SLACK_WEBHOOK }} \
            -H 'Content-type: application/json' \
            -d '{
              "text": "⚠️ Retraining candidate rejected - performance regression detected",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "Review the comparison report for details. Manual intervention may be required."
                  }
                }
              ]
            }'
```

## Project: Build an MLOps Pipeline with GitHub Actions

Let's bring everything together. We'll build a complete MLOps pipeline for a fraud detection model.

### Project Structure

```
fraud-detection-mlops/
├── .github/
│   └── workflows/
│       ├── train.yaml
│       ├── deploy.yaml
│       ├── monitor.yaml
│       └── retrain.yaml
├── src/
│   ├── FraudDetection.Training/
│   ├── FraudDetection.Inference/
│   ├── FraudDetection.Monitoring/
│   ├── FraudDetection.Registry/
│   └── FraudDetection.Tests/
├── models/
│   └── .gitkeep
├── config/
│   ├── training-config.json
│   └── monitoring-thresholds.json
└── README.md
```

### Step 1: The Training Configuration

```json
{
  "training": {
    "algorithm": "FastTree",
    "hyperparameters": {
      "numberOfTrees": 100,
      "numberOfLeaves": 20,
      "minimumExampleCountPerLeaf": 10,
      "learningRate": 0.2
    },
    "features": [
      "transaction_amount",
      "hour_of_day",
      "day_of_week",
      "merchant_category",
      "distance_from_home",
      "time_since_last_transaction",
      "transaction_frequency_24h"
    ],
    "label": "is_fraud"
  },
  "evaluation": {
    "minimumAccuracy": 0.95,
    "minimumPrecision": 0.85,
    "minimumRecall": 0.70,
    "minimumAUC": 0.92
  },
  "deployment": {
    "initialTrafficPercentage": 10,
    "fullRolloutAfterHours": 48,
    "rollbackThreshold": 0.05
  }
}
```

### Step 2: The Main Training Workflow

```yaml
name: Fraud Detection - Training Pipeline

on:
  push:
    branches: [main]
    paths:
      - 'src/FraudDetection.Training/**'
      - 'config/training-config.json'
  schedule:
    - cron: '0 3 * * 1'  # Every Monday at 3 AM
  workflow_dispatch:

env:
  DOTNET_VERSION: '8.0.x'
  AZURE_STORAGE_CONNECTION: ${{ secrets.AZURE_STORAGE_CONNECTION }}

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Build
        run: dotnet build src/FraudDetection.sln
      
      - name: Unit Tests
        run: dotnet test src/FraudDetection.Tests --filter Category!=Integration

  validate-data:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Download Training Data
        run: |
          dotnet run --project src/FraudDetection.DataPipeline -- download \
            --connection "$AZURE_STORAGE_CONNECTION" \
            --output data/raw
      
      - name: Validate Data Quality
        run: |
          dotnet run --project src/FraudDetection.DataPipeline -- validate \
            --input data/raw \
            --output reports/data-quality.json \
            --min-rows 50000 \
            --max-null-percent 0.05

  train:
    needs: [build-and-test, validate-data]
    runs-on: ubuntu-latest
    outputs:
      model_version: ${{ steps.train.outputs.version }}
      metrics_json: ${{ steps.train.outputs.metrics }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Download Data
        run: |
          dotnet run --project src/FraudDetection.DataPipeline -- download \
            --connection "$AZURE_STORAGE_CONNECTION" \
            --output data/raw
      
      - name: Prepare Features
        run: |
          dotnet run --project src/FraudDetection.Training -- prepare \
            --input data/raw \
            --output data/features \
            --config config/training-config.json
      
      - name: Train Model
        id: train
        run: |
          VERSION="v$(date +%Y%m%d)-${{ github.run_number }}-$(git rev-parse --short HEAD)"
          
          dotnet run --project src/FraudDetection.Training -- train \
            --data data/features \
            --output models/$VERSION \
            --config config/training-config.json \
            --run-id $VERSION
          
          # Read metrics
          METRICS=$(cat models/$VERSION/metrics.json)
          
          echo "version=$VERSION" >> $GITHUB_OUTPUT
          echo "metrics=$METRICS" >> $GITHUB_OUTPUT
      
      - name: Upload Model Artifact
        uses: actions/upload-artifact@v4
        with:
          name: model-${{ steps.train.outputs.version }}
          path: models/${{ steps.train.outputs.version }}

  evaluate:
    needs: train
    runs-on: ubuntu-latest
    outputs:
      passed: ${{ steps.evaluate.outputs.passed }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Download Model
        uses: actions/download-artifact@v4
        with:
          name: model-${{ needs.train.outputs.model_version }}
          path: model
      
      - name: Evaluate Against Thresholds
        id: evaluate
        run: |
          RESULT=$(dotnet run --project src/FraudDetection.Evaluation -- \
            --model model/model.zip \
            --config config/training-config.json)
          
          echo "passed=$RESULT" >> $GITHUB_OUTPUT
      
      - name: Compare with Production
        run: |
          dotnet run --project src/FraudDetection.Registry -- download-production \
            --output production-model
          
          dotnet run --project src/FraudDetection.Evaluation -- compare \
            --candidate model/model.zip \
            --baseline production-model/model.zip \
            --output reports/comparison.md
      
      - name: Upload Evaluation Report
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-report
          path: reports/

  register:
    needs: [train, evaluate]
    if: needs.evaluate.outputs.passed == 'true'
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Download Model
        uses: actions/download-artifact@v4
        with:
          name: model-${{ needs.train.outputs.model_version }}
          path: model
      
      - name: Register Model
        run: |
          dotnet run --project src/FraudDetection.Registry -- register \
            --model model/model.zip \
            --version ${{ needs.train.outputs.model_version }} \
            --metrics model/metrics.json \
            --stage staging \
            --connection "$AZURE_STORAGE_CONNECTION"
      
      - name: Create Release
        uses: softprops/action-gh-release@v1
        with:
          tag_name: ${{ needs.train.outputs.model_version }}
          name: "Model ${{ needs.train.outputs.model_version }}"
          files: |
            model/model.zip
            model/metrics.json
            model/metadata.json
```

### Step 3: The Monitoring Workflow

```yaml
name: Fraud Detection - Model Monitoring

on:
  schedule:
    - cron: '0 * * * *'  # Every hour
  workflow_dispatch:

env:
  DOTNET_VERSION: '8.0.x'

jobs:
  monitor:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: ${{ env.DOTNET_VERSION }}
      
      - name: Check Data Drift
        id: drift
        run: |
          RESULT=$(dotnet run --project src/FraudDetection.Monitoring -- drift \
            --window-hours 24 \
            --threshold 0.2)
          echo "drift_detected=$RESULT" >> $GITHUB_OUTPUT
      
      - name: Check Model Performance
        id: performance
        run: |
          RESULT=$(dotnet run --project src/FraudDetection.Monitoring -- performance \
            --window-hours 24 \
            --accuracy-threshold 0.93)
          echo "performance_degraded=$RESULT" >> $GITHUB_OUTPUT
      
      - name: Check Operational Metrics
        id: operational
        run: |
          RESULT=$(dotnet run --project src/FraudDetection.Monitoring -- operational \
            --latency-p99-threshold 50 \
            --error-rate-threshold 0.01)
          echo "operational_issues=$RESULT" >> $GITHUB_OUTPUT
      
      - name: Trigger Retraining
        if: steps.drift.outputs.drift_detected == 'true' || steps.performance.outputs.performance_degraded == 'true'
        uses: peter-evans/repository-dispatch@v2
        with:
          event-type: trigger-retraining
          client-payload: |
            {
              "drift_detected": "${{ steps.drift.outputs.drift_detected }}",
              "performance_degraded": "${{ steps.performance.outputs.performance_degraded }}",
              "triggered_at": "${{ github.event.repository.updated_at }}"
            }
      
      - name: Alert on Issues
        if: failure() || steps.operational.outputs.operational_issues == 'true'
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "⚠️ Fraud Detection Model Alert",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "Issues detected:\n• Drift: ${{ steps.drift.outputs.drift_detected }}\n• Performance: ${{ steps.performance.outputs.performance_degraded }}\n• Operational: ${{ steps.operational.outputs.operational_issues }}"
                  }
                }
              ]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Step 4: The Deployment Workflow

```yaml
name: Fraud Detection - Model Deployment

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - staging
          - production
      traffic_percentage:
        description: 'Traffic percentage (production only)'
        required: false
        default: '10'

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Verify Model Exists
        run: |
          dotnet run --project src/FraudDetection.Registry -- verify \
            --version ${{ inputs.model_version }}

  deploy-staging:
    needs: validate
    if: inputs.environment == 'staging'
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Staging
        run: |
          dotnet run --project src/FraudDetection.Deploy -- \
            --version ${{ inputs.model_version }} \
            --environment staging \
            --traffic-percentage 100
      
      - name: Run Integration Tests
        run: dotnet test src/FraudDetection.Tests --filter Category=Integration

  deploy-production:
    needs: validate
    if: inputs.environment == 'production'
    runs-on: ubuntu-latest
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Deploy to Production
        run: |
          dotnet run --project src/FraudDetection.Deploy -- \
            --version ${{ inputs.model_version }} \
            --environment production \
            --traffic-percentage ${{ inputs.traffic_percentage }}
      
      - name: Update Traffic Rules
        run: |
          dotnet run --project src/FraudDetection.TrafficManager -- update \
            --model-version ${{ inputs.model_version }} \
            --percentage ${{ inputs.traffic_percentage }}
      
      - name: Create Deployment Record
        run: |
          echo '{
            "version": "${{ inputs.model_version }}",
            "environment": "production",
            "traffic": ${{ inputs.traffic_percentage }},
            "deployed_by": "${{ github.actor }}",
            "deployed_at": "'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"
          }' > deployment-record.json
          
          dotnet run --project src/FraudDetection.Registry -- record-deployment \
            --input deployment-record.json
```

This project gives you a complete, production-ready MLOps pipeline. The key patterns:

1. **Automated testing** at multiple levels (unit, data, model quality)
2. **Versioned artifacts** with full traceability
3. **Staged deployments** with gradual traffic shifting
4. **Continuous monitoring** with automated alerts
5. **Automatic retraining** triggered by drift or decay

## Summary

MLOps is where software engineering discipline meets machine learning's unique challenges. The key principles:

- **Version everything**: Code, data, models, and configurations
- **Test at every level**: Units, data quality, model performance, integration
- **Monitor continuously**: Operational metrics, data drift, model decay
- **Automate retraining**: Trigger on schedule, drift, or performance degradation
- **Deploy gradually**: A/B testing and traffic splitting reduce risk

Your .NET background is an asset here. The practices—CI/CD, testing, monitoring, configuration management—are familiar. The challenge is applying them to the unique properties of ML systems: the three axes of change, silent failures, and inevitable drift.

The GitHub Actions workflows in this chapter are production-ready starting points. Adapt them to your specific needs, but keep the core patterns: automated validation, staged deployment, continuous monitoring, and feedback loops that trigger retraining.

In the next chapter, we'll explore advanced deployment strategies, including edge deployment, model optimization, and serving at scale.

## Exercises

**Exercise 1: Implement a Data Validation Suite**

Create a comprehensive data validation pipeline that checks:
- Schema compliance (correct columns and types)
- Statistical properties (distributions within expected ranges)
- Referential integrity (valid foreign keys)
- Temporal consistency (no future dates, logical time ordering)

Output a detailed report that can be used as a CI gate.

**Exercise 2: Build a Model Comparison Tool**

Create a tool that compares two models (candidate vs. baseline) on the same test set and generates a markdown report including:
- Side-by-side metrics comparison
- Performance on subgroups (sliced by categorical features)
- Prediction distribution comparison
- Statistical significance of differences
- Go/no-go recommendation

**Exercise 3: Implement Gradual Rollout with Auto-Rollback**

Extend the A/B testing infrastructure to support gradual rollout:
- Start at 5% traffic
- Increase by 10% every hour if guardrail metrics are healthy
- Auto-rollback if error rate exceeds threshold
- Full rollout after 24 hours of healthy metrics

**Exercise 4: Create a Drift Detection Dashboard**

Build an ASP.NET Core dashboard that displays:
- Real-time PSI scores for each feature
- Time series of prediction distributions
- Alerts when drift exceeds thresholds
- Historical view of drift events correlated with retraining

**Exercise 5: Design a Feature Store Integration**

Implement a feature store pattern that:
- Stores computed features with versioning
- Ensures consistency between training and inference
- Tracks feature lineage (which raw data produced which features)
- Supports point-in-time lookups for training data preparation

Document your design and implement the core interfaces with at least one concrete implementation (can be file-based for simplicity).
