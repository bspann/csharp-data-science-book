# Chapter 17: Model Deployment Patterns

You've trained your model. The accuracy metrics look solid, the validation curves are smooth, and you're feeling good about what you've built. Now comes the part that separates data science experiments from production systems: getting that model into the hands of users.

Deployment is where many ML projects go to die. The gap between "it works on my machine" and "it reliably serves predictions in production" is wider than most developers expect. But here's the good news: as a C# developer, you already know how to build production systems. You understand web APIs, dependency injection, containerization, and cloud deployment. The ML-specific patterns are just a new layer on skills you've already mastered.

In this chapter, we'll take your trained models and deploy them properly. We'll cover the full spectrum: from serializing models to disk, through ASP.NET Core APIs with proper pooling, to Azure Functions for serverless inference, to Docker containers for portable deployment. By the end, you'll have a complete mental model of how ML models go from training notebook to production endpoint.

## Saving and Loading ML.NET Models

Before you can deploy a model, you need to save it. ML.NET models are serialized to a binary format that captures both the trained parameters and the transformation pipeline. This is important: you're not just saving weights, you're saving the entire data processing chain that your model expects.

### The Basic Pattern

Saving a model is straightforward:

```csharp
public class ModelPersistence
{
    private readonly MLContext _mlContext;
    
    public ModelPersistence()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public void SaveModel(
        ITransformer model, 
        DataViewSchema inputSchema, 
        string modelPath)
    {
        // Save model to file
        _mlContext.Model.Save(model, inputSchema, modelPath);
        
        Console.WriteLine($"Model saved to: {modelPath}");
        Console.WriteLine($"File size: {new FileInfo(modelPath).Length:N0} bytes");
    }
    
    public ITransformer LoadModel(string modelPath, out DataViewSchema inputSchema)
    {
        // Load model from file
        var model = _mlContext.Model.Load(modelPath, out inputSchema);
        
        Console.WriteLine($"Model loaded from: {modelPath}");
        return model;
    }
}
```

The `inputSchema` parameter is crucial. It captures the expected structure of your input data—column names, types, and any metadata. When you load the model later, you'll get this schema back, ensuring you know exactly what input format the model expects.

### Stream-Based Loading

In production scenarios, you often don't want to load from a file path. Maybe your model lives in Azure Blob Storage, or it's embedded as a resource, or it comes from a database. ML.NET supports stream-based operations:

```csharp
public class StreamBasedModelLoader
{
    private readonly MLContext _mlContext;
    
    public StreamBasedModelLoader()
    {
        _mlContext = new MLContext();
    }
    
    public async Task<ITransformer> LoadFromBlobStorageAsync(
        BlobClient blobClient)
    {
        using var memoryStream = new MemoryStream();
        await blobClient.DownloadToAsync(memoryStream);
        memoryStream.Position = 0;
        
        return _mlContext.Model.Load(memoryStream, out _);
    }
    
    public ITransformer LoadFromEmbeddedResource(string resourceName)
    {
        var assembly = Assembly.GetExecutingAssembly();
        using var stream = assembly.GetManifestResourceStream(resourceName);
        
        if (stream == null)
            throw new FileNotFoundException($"Resource not found: {resourceName}");
        
        return _mlContext.Model.Load(stream, out _);
    }
    
    public async Task SaveToBlobStorageAsync(
        ITransformer model,
        DataViewSchema schema,
        BlobClient blobClient)
    {
        using var memoryStream = new MemoryStream();
        _mlContext.Model.Save(model, schema, memoryStream);
        memoryStream.Position = 0;
        
        await blobClient.UploadAsync(memoryStream, overwrite: true);
    }
}
```

### Model Versioning

Production systems need model versioning. You'll want to track which model version is running, roll back if something goes wrong, and A/B test different versions. Here's a simple versioning scheme:

```csharp
public record ModelMetadata(
    string Version,
    DateTime TrainedAt,
    string DatasetHash,
    Dictionary<string, double> Metrics,
    string Description);

public class VersionedModelManager
{
    private readonly string _modelDirectory;
    
    public VersionedModelManager(string modelDirectory)
    {
        _modelDirectory = modelDirectory;
        Directory.CreateDirectory(modelDirectory);
    }
    
    public void SaveVersionedModel(
        MLContext mlContext,
        ITransformer model,
        DataViewSchema schema,
        ModelMetadata metadata)
    {
        var versionDir = Path.Combine(_modelDirectory, metadata.Version);
        Directory.CreateDirectory(versionDir);
        
        // Save the model
        var modelPath = Path.Combine(versionDir, "model.zip");
        mlContext.Model.Save(model, schema, modelPath);
        
        // Save metadata
        var metadataPath = Path.Combine(versionDir, "metadata.json");
        var json = JsonSerializer.Serialize(metadata, new JsonSerializerOptions 
        { 
            WriteIndented = true 
        });
        File.WriteAllText(metadataPath, json);
        
        // Update latest symlink (or marker file on Windows)
        UpdateLatestMarker(metadata.Version);
    }
    
    public (ITransformer Model, ModelMetadata Metadata) LoadVersion(
        MLContext mlContext, 
        string version)
    {
        var versionDir = Path.Combine(_modelDirectory, version);
        
        var modelPath = Path.Combine(versionDir, "model.zip");
        var model = mlContext.Model.Load(modelPath, out _);
        
        var metadataPath = Path.Combine(versionDir, "metadata.json");
        var metadata = JsonSerializer.Deserialize<ModelMetadata>(
            File.ReadAllText(metadataPath))!;
        
        return (model, metadata);
    }
    
    public (ITransformer Model, ModelMetadata Metadata) LoadLatest(MLContext mlContext)
    {
        var latestVersion = GetLatestVersion();
        return LoadVersion(mlContext, latestVersion);
    }
    
    private string GetLatestVersion()
    {
        var markerPath = Path.Combine(_modelDirectory, "latest.txt");
        return File.ReadAllText(markerPath).Trim();
    }
    
    private void UpdateLatestMarker(string version)
    {
        var markerPath = Path.Combine(_modelDirectory, "latest.txt");
        File.WriteAllText(markerPath, version);
    }
}
```

[FIGURE: Diagram showing model versioning structure with directories for v1.0.0, v1.0.1, v1.1.0, each containing model.zip and metadata.json, with a latest.txt pointer]

### ONNX Export for Cross-Platform Deployment

While ML.NET's native format is great for .NET deployments, ONNX gives you portability. You can deploy the same model to different runtimes, languages, and platforms:

```csharp
public void ExportToOnnx(
    MLContext mlContext,
    ITransformer model,
    IDataView sampleData,
    string onnxPath)
{
    using var stream = File.Create(onnxPath);
    
    mlContext.Model.ConvertToOnnx(
        model,
        sampleData,
        stream);
    
    Console.WriteLine($"ONNX model exported to: {onnxPath}");
}
```

The `sampleData` parameter is required because ONNX needs to know the exact input shapes and types. Pass a small representative sample of your training data.

## Web API Deployment with ASP.NET Core

The most common deployment pattern is wrapping your model in a REST API. Clients send prediction requests, your API runs inference, and returns results. Simple in concept—but the details matter for performance and reliability.

### The PredictionEngine Problem

Your first instinct might be to create a `PredictionEngine` and use it directly:

```csharp
// DON'T DO THIS IN PRODUCTION
public class NaiveController : ControllerBase
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    
    public NaiveController()
    {
        _mlContext = new MLContext();
        _model = _mlContext.Model.Load("model.zip", out _);
    }
    
    [HttpPost("predict")]
    public PredictionResult Predict(InputData input)
    {
        // Creating PredictionEngine on every request - expensive!
        var engine = _mlContext.Model.CreatePredictionEngine<InputData, PredictionResult>(_model);
        return engine.Predict(input);
    }
}
```

This code has a serious problem: `CreatePredictionEngine` is expensive. It allocates memory, builds internal structures, and does significant setup work. Creating one for every request kills your throughput.

But you can't just create one `PredictionEngine` and share it—it's not thread-safe. Using a single instance across concurrent requests will cause race conditions and corrupt predictions.

### PredictionEnginePool: The Solution

ML.NET provides `PredictionEnginePool<TInput, TOutput>` to solve this exact problem. It maintains a pool of `PredictionEngine` instances that are reused across requests, with proper thread safety:

```csharp
// Install: Microsoft.Extensions.ML (version 5.0.0)

public class SentimentInput
{
    [LoadColumn(0)]
    public string? Text { get; set; }
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsPositive { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
}
```

Configure the pool in `Program.cs`:

```csharp
using Microsoft.Extensions.ML;

var builder = WebApplication.CreateBuilder(args);

// Add PredictionEnginePool to DI
builder.Services.AddPredictionEnginePool<SentimentInput, SentimentPrediction>()
    .FromFile(modelName: "SentimentModel", filePath: "models/sentiment.zip");

builder.Services.AddControllers();

var app = builder.Build();

app.MapControllers();

app.Run();
```

Now inject and use the pool in your controller:

```csharp
[ApiController]
[Route("api/[controller]")]
public class SentimentController : ControllerBase
{
    private readonly PredictionEnginePool<SentimentInput, SentimentPrediction> _predictionPool;
    private readonly ILogger<SentimentController> _logger;
    
    public SentimentController(
        PredictionEnginePool<SentimentInput, SentimentPrediction> predictionPool,
        ILogger<SentimentController> logger)
    {
        _predictionPool = predictionPool;
        _logger = logger;
    }
    
    [HttpPost("analyze")]
    public ActionResult<SentimentResponse> Analyze([FromBody] SentimentRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return BadRequest("Text is required");
        }
        
        var input = new SentimentInput { Text = request.Text };
        
        var stopwatch = Stopwatch.StartNew();
        var prediction = _predictionPool.Predict(modelName: "SentimentModel", example: input);
        stopwatch.Stop();
        
        _logger.LogInformation(
            "Prediction completed in {ElapsedMs}ms. Sentiment: {Sentiment}, Probability: {Probability:P2}",
            stopwatch.ElapsedMilliseconds,
            prediction.IsPositive ? "Positive" : "Negative",
            prediction.Probability);
        
        return new SentimentResponse
        {
            Text = request.Text,
            Sentiment = prediction.IsPositive ? "positive" : "negative",
            Confidence = prediction.Probability,
            InferenceTimeMs = stopwatch.ElapsedMilliseconds
        };
    }
    
    [HttpPost("analyze/batch")]
    public ActionResult<BatchSentimentResponse> AnalyzeBatch([FromBody] BatchSentimentRequest request)
    {
        if (request.Texts == null || request.Texts.Count == 0)
        {
            return BadRequest("At least one text is required");
        }
        
        var stopwatch = Stopwatch.StartNew();
        
        var results = request.Texts
            .Select(text => 
            {
                var input = new SentimentInput { Text = text };
                var prediction = _predictionPool.Predict("SentimentModel", input);
                return new SentimentResult
                {
                    Text = text,
                    Sentiment = prediction.IsPositive ? "positive" : "negative",
                    Confidence = prediction.Probability
                };
            })
            .ToList();
        
        stopwatch.Stop();
        
        return new BatchSentimentResponse
        {
            Results = results,
            TotalInferenceTimeMs = stopwatch.ElapsedMilliseconds,
            AverageInferenceTimeMs = stopwatch.ElapsedMilliseconds / (double)results.Count
        };
    }
}

public record SentimentRequest(string Text);

public record SentimentResponse
{
    public string Text { get; init; } = "";
    public string Sentiment { get; init; } = "";
    public float Confidence { get; init; }
    public long InferenceTimeMs { get; init; }
}

public record BatchSentimentRequest(List<string> Texts);

public record BatchSentimentResponse
{
    public List<SentimentResult> Results { get; init; } = new();
    public long TotalInferenceTimeMs { get; init; }
    public double AverageInferenceTimeMs { get; init; }
}

public record SentimentResult
{
    public string Text { get; init; } = "";
    public string Sentiment { get; init; } = "";
    public float Confidence { get; init; }
}
```

### Loading Models from Multiple Sources

The `PredictionEnginePool` supports loading from files, URIs, and streams. For cloud deployments, loading from blob storage is common:

```csharp
builder.Services.AddPredictionEnginePool<SentimentInput, SentimentPrediction>()
    .FromUri(
        modelName: "SentimentModel",
        uri: "https://mystorageaccount.blob.core.windows.net/models/sentiment.zip",
        period: TimeSpan.FromHours(1)); // Reload every hour
```

The `period` parameter enables automatic model reloading. Your API will pick up new model versions without restarting.

### Multiple Models in One API

Real applications often serve multiple models. Register each one:

```csharp
builder.Services.AddPredictionEnginePool<SentimentInput, SentimentPrediction>()
    .FromFile("SentimentModel", "models/sentiment.zip");

builder.Services.AddPredictionEnginePool<SpamInput, SpamPrediction>()
    .FromFile("SpamModel", "models/spam.zip");

builder.Services.AddPredictionEnginePool<PriceInput, PricePrediction>()
    .FromFile("PriceModel", "models/pricing.zip");
```

### Health Checks and Monitoring

Production APIs need health checks. Add model-specific health verification:

```csharp
public class ModelHealthCheck : IHealthCheck
{
    private readonly PredictionEnginePool<SentimentInput, SentimentPrediction> _pool;
    
    public ModelHealthCheck(PredictionEnginePool<SentimentInput, SentimentPrediction> pool)
    {
        _pool = pool;
    }
    
    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // Run a test prediction
            var testInput = new SentimentInput { Text = "health check test" };
            var prediction = _pool.Predict("SentimentModel", testInput);
            
            // Verify we got a reasonable result
            if (prediction.Probability >= 0 && prediction.Probability <= 1)
            {
                return Task.FromResult(HealthCheckResult.Healthy("Model is responding correctly"));
            }
            
            return Task.FromResult(HealthCheckResult.Degraded("Model returned unexpected values"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(HealthCheckResult.Unhealthy("Model prediction failed", ex));
        }
    }
}

// In Program.cs
builder.Services.AddHealthChecks()
    .AddCheck<ModelHealthCheck>("sentiment_model");
```

[FIGURE: Architecture diagram showing ASP.NET Core API with PredictionEnginePool, showing request flow from client through controller to pooled prediction engines]

## Azure Functions for ML Inference

Not every workload needs an always-running API. For sporadic prediction requests, event-driven inference, or cost-sensitive applications, Azure Functions offers serverless deployment that scales to zero when idle.

### HTTP-Triggered Inference Function

Here's a complete Azure Functions project for ML inference:

```csharp
// Function project file (.csproj)
// <PackageReference Include="Microsoft.Azure.Functions.Worker" Version="1.21.0" />
// <PackageReference Include="Microsoft.Azure.Functions.Worker.Sdk" Version="1.17.0" />
// <PackageReference Include="Microsoft.Azure.Functions.Worker.Extensions.Http" Version="3.1.0" />
// <PackageReference Include="Microsoft.ML" Version="5.0.0" />
// <PackageReference Include="Microsoft.Extensions.ML" Version="5.0.0" />

using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.ML;
using System.Net;

public class PredictionFunction
{
    private readonly PredictionEnginePool<HousePriceInput, HousePricePrediction> _predictionPool;
    private readonly ILogger<PredictionFunction> _logger;
    
    public PredictionFunction(
        PredictionEnginePool<HousePriceInput, HousePricePrediction> predictionPool,
        ILogger<PredictionFunction> logger)
    {
        _predictionPool = predictionPool;
        _logger = logger;
    }
    
    [Function("PredictHousePrice")]
    public async Task<HttpResponseData> PredictHousePrice(
        [HttpTrigger(AuthorizationLevel.Function, "post")] HttpRequestData req)
    {
        _logger.LogInformation("House price prediction request received");
        
        var requestBody = await req.ReadFromJsonAsync<HousePriceRequest>();
        
        if (requestBody == null)
        {
            var badRequest = req.CreateResponse(HttpStatusCode.BadRequest);
            await badRequest.WriteStringAsync("Invalid request body");
            return badRequest;
        }
        
        var input = new HousePriceInput
        {
            SquareFootage = requestBody.SquareFootage,
            Bedrooms = requestBody.Bedrooms,
            Bathrooms = requestBody.Bathrooms,
            YearBuilt = requestBody.YearBuilt,
            LotSize = requestBody.LotSize,
            Neighborhood = requestBody.Neighborhood
        };
        
        var prediction = _predictionPool.Predict("HousePriceModel", input);
        
        var response = req.CreateResponse(HttpStatusCode.OK);
        await response.WriteAsJsonAsync(new HousePriceResponse
        {
            PredictedPrice = prediction.Price,
            ConfidenceInterval = new ConfidenceInterval
            {
                Lower = prediction.Price * 0.9f,  // Simplified; real implementation would use proper intervals
                Upper = prediction.Price * 1.1f
            }
        });
        
        _logger.LogInformation("Predicted price: {Price:C}", prediction.Price);
        
        return response;
    }
}

public class HousePriceInput
{
    public float SquareFootage { get; set; }
    public float Bedrooms { get; set; }
    public float Bathrooms { get; set; }
    public float YearBuilt { get; set; }
    public float LotSize { get; set; }
    public string Neighborhood { get; set; } = "";
}

public class HousePricePrediction
{
    [ColumnName("Score")]
    public float Price { get; set; }
}

public record HousePriceRequest(
    float SquareFootage,
    float Bedrooms,
    float Bathrooms,
    float YearBuilt,
    float LotSize,
    string Neighborhood);

public record HousePriceResponse
{
    public float PredictedPrice { get; init; }
    public ConfidenceInterval ConfidenceInterval { get; init; } = new();
}

public record ConfidenceInterval
{
    public float Lower { get; init; }
    public float Upper { get; init; }
}
```

Configure the function app:

```csharp
// Program.cs for Azure Functions
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.ML;

var host = new HostBuilder()
    .ConfigureFunctionsWorkerDefaults()
    .ConfigureServices(services =>
    {
        // Load model from embedded resource or local file
        services.AddPredictionEnginePool<HousePriceInput, HousePricePrediction>()
            .FromFile("HousePriceModel", "models/house_price.zip");
    })
    .Build();

host.Run();
```

### Queue-Triggered Batch Processing

For batch inference jobs, use queue triggers:

```csharp
public class BatchPredictionFunction
{
    private readonly PredictionEnginePool<HousePriceInput, HousePricePrediction> _predictionPool;
    private readonly BlobServiceClient _blobClient;
    private readonly ILogger<BatchPredictionFunction> _logger;
    
    public BatchPredictionFunction(
        PredictionEnginePool<HousePriceInput, HousePricePrediction> predictionPool,
        BlobServiceClient blobClient,
        ILogger<BatchPredictionFunction> logger)
    {
        _predictionPool = predictionPool;
        _blobClient = blobClient;
        _logger = logger;
    }
    
    [Function("ProcessBatchPrediction")]
    public async Task ProcessBatch(
        [QueueTrigger("prediction-jobs")] BatchPredictionJob job)
    {
        _logger.LogInformation("Processing batch job: {JobId}", job.JobId);
        
        // Download input data from blob
        var containerClient = _blobClient.GetBlobContainerClient("prediction-data");
        var inputBlob = containerClient.GetBlobClient(job.InputBlobName);
        
        var inputContent = await inputBlob.DownloadContentAsync();
        var inputs = JsonSerializer.Deserialize<List<HousePriceInput>>(
            inputContent.Value.Content.ToString())!;
        
        // Process all predictions
        var results = inputs.Select(input => new
        {
            Input = input,
            Prediction = _predictionPool.Predict("HousePriceModel", input)
        }).ToList();
        
        // Upload results
        var outputBlob = containerClient.GetBlobClient($"results/{job.JobId}.json");
        var outputJson = JsonSerializer.Serialize(results);
        await outputBlob.UploadAsync(
            BinaryData.FromString(outputJson), 
            overwrite: true);
        
        _logger.LogInformation(
            "Completed batch job: {JobId}, processed {Count} records",
            job.JobId,
            results.Count);
    }
}

public record BatchPredictionJob(string JobId, string InputBlobName);
```

### Cold Start Considerations

Azure Functions have cold start latency—the first invocation after a period of inactivity takes longer as the runtime initializes. For ML workloads, this includes loading your model into memory.

Mitigation strategies:

1. **Premium plan**: Keeps instances warm with pre-warmed workers
2. **Model size optimization**: Smaller models load faster
3. **Warm-up triggers**: Use timer triggers to keep functions warm during business hours

```csharp
// Keep the function warm during business hours
[Function("KeepWarm")]
public void KeepWarm([TimerTrigger("0 */5 9-17 * * 1-5")] TimerInfo timer)
{
    // Run a dummy prediction to keep the model loaded
    var dummyInput = new HousePriceInput
    {
        SquareFootage = 2000,
        Bedrooms = 3,
        Bathrooms = 2,
        YearBuilt = 2000,
        LotSize = 5000,
        Neighborhood = "Test"
    };
    
    _predictionPool.Predict("HousePriceModel", dummyInput);
    _logger.LogDebug("Warm-up prediction completed");
}
```

## Containerizing ML Models with Docker

Containers give you consistent, portable deployments. Your model runs the same way in development, staging, and production—no more "works on my machine" surprises.

### Basic Dockerfile for ML.NET API

```dockerfile
# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy project files
COPY ["PredictionApi/PredictionApi.csproj", "PredictionApi/"]
RUN dotnet restore "PredictionApi/PredictionApi.csproj"

# Copy source code
COPY . .
WORKDIR "/src/PredictionApi"
RUN dotnet build -c Release -o /app/build

# Publish stage
FROM build AS publish
RUN dotnet publish -c Release -o /app/publish /p:UseAppHost=false

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS final
WORKDIR /app

# Copy published app
COPY --from=publish /app/publish .

# Copy models directory
COPY models/ ./models/

# Configure the app
ENV ASPNETCORE_URLS=http://+:8080
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

ENTRYPOINT ["dotnet", "PredictionApi.dll"]
```

### Optimizing Container Size

ML models can be large. Optimize your container:

```dockerfile
# Use Alpine for smaller base image
FROM mcr.microsoft.com/dotnet/aspnet:8.0-alpine AS final

# Install only required native dependencies
RUN apk add --no-cache \
    libstdc++ \
    libgcc \
    icu-libs

WORKDIR /app
COPY --from=publish /app/publish .
COPY models/ ./models/

# Use trimming for smaller deployments (test thoroughly!)
# In .csproj: <PublishTrimmed>true</PublishTrimmed>

ENV ASPNETCORE_URLS=http://+:8080
ENV DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=false

ENTRYPOINT ["dotnet", "PredictionApi.dll"]
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  prediction-api:
    build:
      context: .
      dockerfile: PredictionApi/Dockerfile
    ports:
      - "8080:8080"
    environment:
      - ASPNETCORE_ENVIRONMENT=Development
      - MODEL_PATH=/models/sentiment.zip
    volumes:
      - ./models:/models:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '2'
        reservations:
          memory: 512M
          cpus: '1'
  
  # Optional: Model serving with monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Kubernetes Deployment

For production Kubernetes deployments:

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prediction-api
  labels:
    app: prediction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prediction-api
  template:
    metadata:
      labels:
        app: prediction-api
    spec:
      containers:
      - name: prediction-api
        image: myregistry.azurecr.io/prediction-api:v1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        env:
        - name: ASPNETCORE_ENVIRONMENT
          value: "Production"
        - name: MODEL_BLOB_URL
          valueFrom:
            secretKeyRef:
              name: model-secrets
              key: blob-url
---
apiVersion: v1
kind: Service
metadata:
  name: prediction-api
spec:
  selector:
    app: prediction-api
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prediction-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

[FIGURE: Kubernetes architecture showing load balancer, multiple prediction-api pods, HPA controller, and connection to Azure Blob Storage for model files]

## Project: Production-Ready Sentiment Analysis API

Let's bring everything together with a complete, production-ready sentiment analysis API. This project demonstrates the patterns you'll use in real deployments.

### Project Structure

```
SentimentApi/
├── SentimentApi.csproj
├── Program.cs
├── appsettings.json
├── appsettings.Production.json
├── Models/
│   ├── SentimentInput.cs
│   ├── SentimentPrediction.cs
│   └── ApiModels.cs
├── Services/
│   ├── ISentimentService.cs
│   └── SentimentService.cs
├── Controllers/
│   └── SentimentController.cs
├── Middleware/
│   └── RequestTimingMiddleware.cs
├── HealthChecks/
│   └── ModelHealthCheck.cs
├── Dockerfile
└── models/
    └── sentiment_model.zip
```

### Core Models

```csharp
// Models/SentimentInput.cs
using Microsoft.ML.Data;

namespace SentimentApi.Models;

public class SentimentInput
{
    [LoadColumn(0)]
    [ColumnName("Text")]
    public string Text { get; set; } = "";
}

// Models/SentimentPrediction.cs
using Microsoft.ML.Data;

namespace SentimentApi.Models;

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsPositive { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
}

// Models/ApiModels.cs
namespace SentimentApi.Models;

public record AnalyzeRequest(string Text);

public record AnalyzeResponse(
    string Text,
    string Sentiment,
    float Confidence,
    long InferenceTimeMs,
    string ModelVersion);

public record BatchAnalyzeRequest(List<string> Texts);

public record BatchAnalyzeResponse(
    List<SentimentResult> Results,
    int TotalCount,
    long TotalInferenceTimeMs,
    double AverageInferenceTimeMs,
    string ModelVersion);

public record SentimentResult(
    string Text,
    string Sentiment,
    float Confidence);

public record ModelInfo(
    string Version,
    DateTime LoadedAt,
    long PredictionCount,
    double AverageInferenceTimeMs);
```

### Sentiment Service

```csharp
// Services/ISentimentService.cs
using SentimentApi.Models;

namespace SentimentApi.Services;

public interface ISentimentService
{
    AnalyzeResponse Analyze(string text);
    BatchAnalyzeResponse AnalyzeBatch(IEnumerable<string> texts);
    ModelInfo GetModelInfo();
}

// Services/SentimentService.cs
using System.Diagnostics;
using Microsoft.Extensions.ML;
using SentimentApi.Models;

namespace SentimentApi.Services;

public class SentimentService : ISentimentService
{
    private readonly PredictionEnginePool<SentimentInput, SentimentPrediction> _predictionPool;
    private readonly ILogger<SentimentService> _logger;
    private readonly IConfiguration _configuration;
    
    private readonly DateTime _loadedAt = DateTime.UtcNow;
    private long _predictionCount;
    private double _totalInferenceTime;
    private readonly object _statsLock = new();
    
    private const string ModelName = "SentimentModel";
    
    public SentimentService(
        PredictionEnginePool<SentimentInput, SentimentPrediction> predictionPool,
        ILogger<SentimentService> logger,
        IConfiguration configuration)
    {
        _predictionPool = predictionPool;
        _logger = logger;
        _configuration = configuration;
    }
    
    public AnalyzeResponse Analyze(string text)
    {
        var input = new SentimentInput { Text = text };
        
        var stopwatch = Stopwatch.StartNew();
        var prediction = _predictionPool.Predict(ModelName, input);
        stopwatch.Stop();
        
        UpdateStats(stopwatch.ElapsedMilliseconds);
        
        var sentiment = prediction.IsPositive ? "positive" : "negative";
        
        _logger.LogDebug(
            "Analyzed text ({Length} chars): {Sentiment} ({Confidence:P2}) in {Ms}ms",
            text.Length,
            sentiment,
            prediction.Probability,
            stopwatch.ElapsedMilliseconds);
        
        return new AnalyzeResponse(
            Text: text,
            Sentiment: sentiment,
            Confidence: prediction.Probability,
            InferenceTimeMs: stopwatch.ElapsedMilliseconds,
            ModelVersion: GetModelVersion());
    }
    
    public BatchAnalyzeResponse AnalyzeBatch(IEnumerable<string> texts)
    {
        var textList = texts.ToList();
        var stopwatch = Stopwatch.StartNew();
        
        var results = textList.Select(text =>
        {
            var input = new SentimentInput { Text = text };
            var prediction = _predictionPool.Predict(ModelName, input);
            
            return new SentimentResult(
                Text: text,
                Sentiment: prediction.IsPositive ? "positive" : "negative",
                Confidence: prediction.Probability);
        }).ToList();
        
        stopwatch.Stop();
        
        UpdateStats(stopwatch.ElapsedMilliseconds, textList.Count);
        
        _logger.LogInformation(
            "Batch analyzed {Count} texts in {Ms}ms (avg: {AvgMs:F2}ms)",
            textList.Count,
            stopwatch.ElapsedMilliseconds,
            stopwatch.ElapsedMilliseconds / (double)textList.Count);
        
        return new BatchAnalyzeResponse(
            Results: results,
            TotalCount: results.Count,
            TotalInferenceTimeMs: stopwatch.ElapsedMilliseconds,
            AverageInferenceTimeMs: stopwatch.ElapsedMilliseconds / (double)results.Count,
            ModelVersion: GetModelVersion());
    }
    
    public ModelInfo GetModelInfo()
    {
        lock (_statsLock)
        {
            var avgTime = _predictionCount > 0 
                ? _totalInferenceTime / _predictionCount 
                : 0;
            
            return new ModelInfo(
                Version: GetModelVersion(),
                LoadedAt: _loadedAt,
                PredictionCount: _predictionCount,
                AverageInferenceTimeMs: avgTime);
        }
    }
    
    private void UpdateStats(long inferenceTimeMs, int count = 1)
    {
        lock (_statsLock)
        {
            _predictionCount += count;
            _totalInferenceTime += inferenceTimeMs;
        }
    }
    
    private string GetModelVersion() => 
        _configuration["ModelVersion"] ?? "1.0.0";
}
```

### Controller

```csharp
// Controllers/SentimentController.cs
using Microsoft.AspNetCore.Mvc;
using SentimentApi.Models;
using SentimentApi.Services;

namespace SentimentApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public class SentimentController : ControllerBase
{
    private readonly ISentimentService _sentimentService;
    private readonly ILogger<SentimentController> _logger;
    
    public SentimentController(
        ISentimentService sentimentService,
        ILogger<SentimentController> logger)
    {
        _sentimentService = sentimentService;
        _logger = logger;
    }
    
    /// <summary>
    /// Analyze sentiment of a single text
    /// </summary>
    [HttpPost("analyze")]
    [ProducesResponseType(typeof(AnalyzeResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public ActionResult<AnalyzeResponse> Analyze([FromBody] AnalyzeRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Text))
        {
            return BadRequest(new { error = "Text is required" });
        }
        
        if (request.Text.Length > 10000)
        {
            return BadRequest(new { error = "Text exceeds maximum length of 10000 characters" });
        }
        
        var result = _sentimentService.Analyze(request.Text);
        return Ok(result);
    }
    
    /// <summary>
    /// Analyze sentiment of multiple texts in a single request
    /// </summary>
    [HttpPost("analyze/batch")]
    [ProducesResponseType(typeof(BatchAnalyzeResponse), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public ActionResult<BatchAnalyzeResponse> AnalyzeBatch([FromBody] BatchAnalyzeRequest request)
    {
        if (request.Texts == null || request.Texts.Count == 0)
        {
            return BadRequest(new { error = "At least one text is required" });
        }
        
        if (request.Texts.Count > 100)
        {
            return BadRequest(new { error = "Maximum batch size is 100 texts" });
        }
        
        var invalidTexts = request.Texts
            .Select((text, index) => new { text, index })
            .Where(x => string.IsNullOrWhiteSpace(x.text) || x.text.Length > 10000)
            .ToList();
        
        if (invalidTexts.Any())
        {
            return BadRequest(new 
            { 
                error = "Invalid texts found",
                invalidIndexes = invalidTexts.Select(x => x.index)
            });
        }
        
        var result = _sentimentService.AnalyzeBatch(request.Texts);
        return Ok(result);
    }
    
    /// <summary>
    /// Get model information and statistics
    /// </summary>
    [HttpGet("model/info")]
    [ProducesResponseType(typeof(ModelInfo), StatusCodes.Status200OK)]
    public ActionResult<ModelInfo> GetModelInfo()
    {
        return Ok(_sentimentService.GetModelInfo());
    }
}
```

### Middleware and Health Checks

```csharp
// Middleware/RequestTimingMiddleware.cs
using System.Diagnostics;

namespace SentimentApi.Middleware;

public class RequestTimingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger<RequestTimingMiddleware> _logger;
    
    public RequestTimingMiddleware(RequestDelegate next, ILogger<RequestTimingMiddleware> logger)
    {
        _next = next;
        _logger = logger;
    }
    
    public async Task InvokeAsync(HttpContext context)
    {
        var stopwatch = Stopwatch.StartNew();
        
        context.Response.OnStarting(() =>
        {
            stopwatch.Stop();
            context.Response.Headers["X-Response-Time-Ms"] = 
                stopwatch.ElapsedMilliseconds.ToString();
            return Task.CompletedTask;
        });
        
        await _next(context);
        
        _logger.LogInformation(
            "{Method} {Path} completed in {Ms}ms with status {Status}",
            context.Request.Method,
            context.Request.Path,
            stopwatch.ElapsedMilliseconds,
            context.Response.StatusCode);
    }
}

// HealthChecks/ModelHealthCheck.cs
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.ML;
using SentimentApi.Models;

namespace SentimentApi.HealthChecks;

public class ModelHealthCheck : IHealthCheck
{
    private readonly PredictionEnginePool<SentimentInput, SentimentPrediction> _pool;
    private readonly ILogger<ModelHealthCheck> _logger;
    
    public ModelHealthCheck(
        PredictionEnginePool<SentimentInput, SentimentPrediction> pool,
        ILogger<ModelHealthCheck> logger)
    {
        _pool = pool;
        _logger = logger;
    }
    
    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var testInput = new SentimentInput { Text = "This is a health check test" };
            var stopwatch = Stopwatch.StartNew();
            var prediction = _pool.Predict("SentimentModel", testInput);
            stopwatch.Stop();
            
            if (prediction.Probability < 0 || prediction.Probability > 1)
            {
                return Task.FromResult(HealthCheckResult.Degraded(
                    "Model returned invalid probability"));
            }
            
            if (stopwatch.ElapsedMilliseconds > 1000)
            {
                return Task.FromResult(HealthCheckResult.Degraded(
                    $"Model inference is slow: {stopwatch.ElapsedMilliseconds}ms"));
            }
            
            return Task.FromResult(HealthCheckResult.Healthy(
                $"Model is healthy. Inference time: {stopwatch.ElapsedMilliseconds}ms"));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Model health check failed");
            return Task.FromResult(HealthCheckResult.Unhealthy(
                "Model prediction failed", ex));
        }
    }
}
```

### Application Entry Point

```csharp
// Program.cs
using Microsoft.Extensions.ML;
using SentimentApi.HealthChecks;
using SentimentApi.Middleware;
using SentimentApi.Models;
using SentimentApi.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() 
    { 
        Title = "Sentiment Analysis API", 
        Version = "v1",
        Description = "Production-ready sentiment analysis using ML.NET"
    });
});

// Add PredictionEnginePool
var modelPath = builder.Configuration["ModelPath"] ?? "models/sentiment_model.zip";
builder.Services.AddPredictionEnginePool<SentimentInput, SentimentPrediction>()
    .FromFile(modelName: "SentimentModel", filePath: modelPath);

// Add services
builder.Services.AddScoped<ISentimentService, SentimentService>();

// Add health checks
builder.Services.AddHealthChecks()
    .AddCheck<ModelHealthCheck>("model", tags: new[] { "ready" });

var app = builder.Build();

// Configure middleware pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseMiddleware<RequestTimingMiddleware>();

app.UseHttpsRedirection();
app.UseAuthorization();

app.MapControllers();

// Health check endpoints
app.MapHealthChecks("/health/live", new()
{
    Predicate = _ => false // Liveness: just check app is running
});

app.MapHealthChecks("/health/ready", new()
{
    Predicate = check => check.Tags.Contains("ready")
});

app.Run();
```

### Configuration

```json
// appsettings.json
{
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  },
  "ModelPath": "models/sentiment_model.zip",
  "ModelVersion": "1.0.0"
}

// appsettings.Production.json
{
  "Logging": {
    "LogLevel": {
      "Default": "Warning",
      "SentimentApi": "Information"
    }
  },
  "ModelPath": "/app/models/sentiment_model.zip"
}
```

### Production Dockerfile

```dockerfile
# Build stage
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY ["SentimentApi.csproj", "./"]
RUN dotnet restore

COPY . .
RUN dotnet publish -c Release -o /app/publish

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0-alpine AS final

# Install curl for health checks
RUN apk add --no-cache curl

WORKDIR /app

# Create non-root user
RUN addgroup -g 1000 appgroup && \
    adduser -u 1000 -G appgroup -D appuser

COPY --from=build /app/publish .
COPY models/ ./models/

# Set ownership
RUN chown -R appuser:appgroup /app

USER appuser

ENV ASPNETCORE_URLS=http://+:8080
ENV ASPNETCORE_ENVIRONMENT=Production
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8080/health/live || exit 1

ENTRYPOINT ["dotnet", "SentimentApi.dll"]
```

### Testing the API

```bash
# Build and run
docker build -t sentiment-api:v1 .
docker run -p 8080:8080 sentiment-api:v1

# Test single prediction
curl -X POST http://localhost:8080/api/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! Best purchase ever."}'

# Test batch prediction
curl -X POST http://localhost:8080/api/sentiment/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great service!", "Terrible experience.", "It was okay."]}'

# Check model info
curl http://localhost:8080/api/sentiment/model/info

# Health checks
curl http://localhost:8080/health/live
curl http://localhost:8080/health/ready
```

[FIGURE: Complete system diagram showing Docker container with ASP.NET Core API, PredictionEnginePool, model file, and external connections for monitoring, logging, and client requests]

## Common Deployment Patterns and When to Use Them

Let's step back and consider the broader picture. You now have several deployment options, each suited to different scenarios.

### Pattern 1: Direct Integration

**When to use**: The model is consumed by a single application, latency is critical, and you want the simplest architecture.

```csharp
// Model loaded directly into the application
public class OrderFraudDetector
{
    private readonly PredictionEnginePool<OrderInput, FraudPrediction> _pool;
    
    public async Task<bool> IsFraudulentAsync(Order order)
    {
        var input = MapToInput(order);
        var prediction = _pool.Predict("FraudModel", input);
        return prediction.IsFraud && prediction.Probability > 0.8f;
    }
}
```

**Pros**: Lowest latency (no network hop), simplest deployment, no additional services to manage.

**Cons**: Model updates require application redeployment, model must fit in application memory, harder to share across services.

### Pattern 2: Sidecar Service

**When to use**: Multiple services need the same model, you want independent model updates, but you want to avoid network latency.

Deploy the model as a sidecar container alongside your main application in the same pod (Kubernetes) or task (ECS). Communication happens over localhost.

```yaml
# Kubernetes pod with sidecar
spec:
  containers:
  - name: main-app
    image: myapp:v1
  - name: ml-sidecar
    image: ml-service:v1
    ports:
    - containerPort: 8081
```

**Pros**: Low latency (localhost), independent model updates, model memory isolated.

**Cons**: More complex deployment, sidecar per instance increases resource usage.

### Pattern 3: Centralized ML Service

**When to use**: Multiple applications need the model, you want centralized monitoring and management, slight latency is acceptable.

This is the REST API pattern we built in this chapter. All prediction requests go to a dedicated service.

**Pros**: Centralized management, single place to monitor, easy A/B testing, clear separation of concerns.

**Cons**: Network latency, additional service to maintain, potential bottleneck.

### Pattern 4: Event-Driven Inference

**When to use**: Predictions don't need to be synchronous, batch processing is acceptable, cost optimization is important.

```csharp
// Queue-triggered processing
[Function("ProcessPredictions")]
public async Task Process([QueueTrigger("predictions")] PredictionRequest request)
{
    var result = _pool.Predict("Model", request.Input);
    await SaveResultAsync(request.CorrelationId, result);
    await NotifyCallerAsync(request.CallbackUrl, result);
}
```

**Pros**: Decoupled, scalable, handles bursts gracefully, can scale to zero.

**Cons**: Added latency, eventual consistency, more complex error handling.

### Pattern 5: Edge Deployment

**When to use**: Low-latency requirements, offline capability needed, data privacy concerns prevent cloud inference.

Deploy models directly to edge devices—IoT gateways, mobile apps, or user browsers. ML.NET models can run anywhere .NET runs.

**Pros**: Lowest possible latency, works offline, data stays local.

**Cons**: Limited compute resources, harder to update models, device heterogeneity.

### Choosing the Right Pattern

| Factor | Direct | Sidecar | Centralized | Event-Driven | Edge |
|--------|--------|---------|-------------|--------------|------|
| Latency | Best | Good | Moderate | Variable | Best |
| Scalability | Limited | Good | Best | Best | N/A |
| Model Updates | Hard | Moderate | Easy | Easy | Hard |
| Complexity | Low | Moderate | Moderate | High | Moderate |
| Cost Efficiency | Good | Moderate | Good | Best | Varies |

Most production systems use a combination. You might have a centralized service for standard workloads, event-driven processing for batch jobs, and edge deployment for latency-critical paths.

## Troubleshooting Common Deployment Issues

Even with solid patterns, things go wrong. Here are the issues I see most often:

### Memory Issues

**Symptom**: OutOfMemoryException, container OOM kills, degraded performance over time.

**Causes and fixes**:

1. **Model too large for container limits**: Increase memory limits or optimize model size
2. **PredictionEngine not being pooled**: Switch to PredictionEnginePool
3. **Memory leaks from improper disposal**: Ensure IDisposable patterns are followed

```csharp
// Check model size before deploying
var modelInfo = new FileInfo("model.zip");
Console.WriteLine($"Model size: {modelInfo.Length / 1024.0 / 1024.0:F2} MB");

// For production, profile with:
// dotnet-counters monitor --process-id <PID> --counters System.Runtime
```

### Cold Start Latency

**Symptom**: First requests after idle period are very slow.

**Fixes**:

1. Use warm-up requests in readiness checks
2. Configure minimum instances (Kubernetes, Azure Functions Premium)
3. Reduce model size with pruning or quantization
4. Pre-load models at application startup

```csharp
// Warm-up during startup
public class ModelWarmupService : IHostedService
{
    private readonly PredictionEnginePool<Input, Output> _pool;
    
    public Task StartAsync(CancellationToken cancellationToken)
    {
        // Force model load and JIT compilation
        var warmupInput = new Input { /* test data */ };
        for (int i = 0; i < 10; i++)
        {
            _pool.Predict("Model", warmupInput);
        }
        return Task.CompletedTask;
    }
    
    public Task StopAsync(CancellationToken ct) => Task.CompletedTask;
}
```

### Inconsistent Predictions

**Symptom**: Same input produces different outputs across deployments.

**Causes and fixes**:

1. **Non-deterministic model training**: Set random seeds
2. **Different ML.NET versions**: Pin exact package versions
3. **Feature engineering drift**: Ensure preprocessing is identical
4. **Schema mismatches**: Validate input schema matches model expectations

```csharp
// Validate schema at startup
var expectedColumns = new[] { "Feature1", "Feature2", "Feature3" };
var modelSchema = _mlContext.Model.Load("model.zip", out var schema);

foreach (var col in expectedColumns)
{
    if (schema.GetColumnOrNull(col) == null)
    {
        throw new InvalidOperationException($"Missing expected column: {col}");
    }
}
```

### Throughput Issues

**Symptom**: API can't handle expected load.

**Fixes**:

1. Scale horizontally with multiple instances
2. Implement request batching on the client side
3. Use async patterns for I/O operations
4. Profile to find the actual bottleneck (is it CPU? Memory? Network?)

```csharp
// Monitor pool statistics
public class PoolMetricsMiddleware
{
    public async Task InvokeAsync(HttpContext context)
    {
        var beforeGC = GC.GetTotalMemory(false);
        var beforeThread = ThreadPool.ThreadCount;
        
        await _next(context);
        
        _logger.LogDebug(
            "Request stats - Memory delta: {MemDelta:N0}, Threads: {Threads}",
            GC.GetTotalMemory(false) - beforeGC,
            ThreadPool.ThreadCount);
    }
}
```

## Deployment Checklist

Before deploying your ML model to production, verify:

**Model Quality**
- [ ] Validation metrics meet business requirements
- [ ] Model tested on holdout data from realistic distribution
- [ ] Edge cases and failure modes documented
- [ ] Model versioned and reproducible

**API Design**
- [ ] Input validation with clear error messages
- [ ] Request/response schemas documented
- [ ] Batch endpoint for efficiency
- [ ] Model info endpoint for observability

**Performance**
- [ ] Using PredictionEnginePool (not raw PredictionEngine)
- [ ] Response times measured and acceptable
- [ ] Load tested at expected throughput
- [ ] Memory usage profiled

**Reliability**
- [ ] Health checks implemented (liveness and readiness)
- [ ] Graceful degradation on model failures
- [ ] Timeouts configured
- [ ] Error handling comprehensive

**Operations**
- [ ] Logging structured and meaningful
- [ ] Metrics exposed for monitoring
- [ ] Container builds reproducibly
- [ ] Deployment automated

## Exercises

**Exercise 1: Model Hot Reloading**

Implement automatic model reloading when a new version is uploaded to blob storage. Your solution should:
- Poll for new model versions every 5 minutes
- Load the new model without dropping requests
- Log model version transitions
- Fall back to the previous version if the new model fails to load

*Hint: Use `PredictionEnginePool` with a URI source and configure the reload period. Add try-catch around prediction to handle transient loading issues.*

**Exercise 2: A/B Testing Infrastructure**

Extend the sentiment API to support A/B testing between two model versions. Requirements:
- Accept a `model_version` parameter (optional) to force a specific version
- Without the parameter, randomly assign requests 50/50 between versions
- Track metrics (accuracy, latency) separately for each version
- Add an endpoint to view A/B test results

**Exercise 3: Batch Processing Optimization**

The current batch endpoint processes predictions sequentially. Optimize it:
- Implement parallel prediction using `Parallel.ForEach` or `Task.WhenAll`
- Respect a configurable concurrency limit
- Benchmark the improvement with 1000 texts
- Consider: Why can't we use a single PredictionEngine across parallel threads?

**Exercise 4: Rate Limiting and Quotas**

Add rate limiting to protect your API:
- Implement per-client rate limits (e.g., 100 requests/minute)
- Add daily quotas with tracking
- Return appropriate 429 responses with retry-after headers
- Create an admin endpoint to view usage statistics

**Exercise 5: Model Monitoring Dashboard**

Create a simple monitoring solution:
- Track prediction distribution over time (positive vs. negative ratio)
- Alert if distribution shifts dramatically (potential data drift)
- Log low-confidence predictions for review
- Build a simple HTML dashboard endpoint showing these metrics

## Summary

Deploying ML models is software engineering. The patterns you've learned—dependency injection, pooling, health checks, containerization—are standard practices applied to a new context.

Key takeaways:

- **PredictionEnginePool is essential**: Never create PredictionEngine instances per-request. The pool handles thread safety and reuse.

- **Models are versioned artifacts**: Treat them like code. Version, store, and deploy systematically.

- **Health checks matter**: Know when your model is healthy and ready to serve traffic.

- **Containers provide portability**: Build once, run anywhere. Your dev environment matches production.

- **Azure Functions offer flexibility**: Not everything needs an always-on API. Serverless works well for sporadic workloads.

The production sentiment API we built is a template. Adapt it to your specific models—the patterns remain the same. You now have the foundation to take any ML.NET model from trained artifact to production service.

In the next chapter, we'll tackle model monitoring and observability—how to know your model is performing correctly in production, detect drift, and maintain quality over time.
