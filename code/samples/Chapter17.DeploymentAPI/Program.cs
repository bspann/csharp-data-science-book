// Chapter 17: ML Model Deployment API
// Demonstrates PredictionEnginePool for thread-safe ML.NET model serving

using System.Diagnostics;
using Microsoft.AspNetCore.Diagnostics.HealthChecks;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.ML;
using Microsoft.ML.Data;

var builder = WebApplication.CreateBuilder(args);

// -----------------------------------------------------------------------------
// ML.NET Model Types
// -----------------------------------------------------------------------------

// Input schema matching the trained sentiment model
public class SentimentInput
{
    [LoadColumn(0)]
    [ColumnName("SentimentText")]
    public string Text { get; set; } = "";
}

// Output schema with prediction results
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool IsPositive { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
}

// -----------------------------------------------------------------------------
// API Request/Response DTOs
// -----------------------------------------------------------------------------

public record PredictRequest(string Text);

public record PredictResponse(
    string Text,
    string Sentiment,
    float Confidence,
    long InferenceMs);

public record BatchPredictRequest(List<string> Texts);

public record BatchPredictResponse(
    List<PredictResponse> Results,
    int Count,
    long TotalMs);

// -----------------------------------------------------------------------------
// Health Check for Model
// -----------------------------------------------------------------------------

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
            // Run a test prediction to verify model is loaded and working
            var testInput = new SentimentInput { Text = "health check" };
            var stopwatch = Stopwatch.StartNew();
            var prediction = _pool.Predict("SentimentModel", testInput);
            stopwatch.Stop();
            
            // Validate the prediction is reasonable
            if (prediction.Probability is < 0 or > 1)
            {
                return Task.FromResult(HealthCheckResult.Degraded(
                    "Model returned invalid probability"));
            }
            
            return Task.FromResult(HealthCheckResult.Healthy(
                $"Model OK - inference: {stopwatch.ElapsedMilliseconds}ms"));
        }
        catch (Exception ex)
        {
            return Task.FromResult(HealthCheckResult.Unhealthy(
                "Model prediction failed", ex));
        }
    }
}

// -----------------------------------------------------------------------------
// Service Configuration
// -----------------------------------------------------------------------------

// Configure PredictionEnginePool for thread-safe model serving
// This pools PredictionEngine instances to avoid expensive per-request creation
var modelPath = builder.Configuration["ModelPath"] ?? "models/sentiment.zip";

builder.Services.AddPredictionEnginePool<SentimentInput, SentimentPrediction>()
    .FromFile(modelName: "SentimentModel", filePath: modelPath);

// Add health checks including model health
builder.Services.AddHealthChecks()
    .AddCheck<ModelHealthCheck>("sentiment-model", tags: ["ready"]);

// Add OpenAPI support
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// Enable Swagger in development
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// -----------------------------------------------------------------------------
// API Endpoints
// -----------------------------------------------------------------------------

// Single prediction endpoint
app.MapPost("/api/predict", (
    PredictRequest request,
    PredictionEnginePool<SentimentInput, SentimentPrediction> pool,
    ILogger<Program> logger) =>
{
    if (string.IsNullOrWhiteSpace(request.Text))
    {
        return Results.BadRequest(new { error = "Text is required" });
    }
    
    var input = new SentimentInput { Text = request.Text };
    
    var stopwatch = Stopwatch.StartNew();
    var prediction = pool.Predict("SentimentModel", input);
    stopwatch.Stop();
    
    var sentiment = prediction.IsPositive ? "positive" : "negative";
    
    logger.LogInformation(
        "Predicted {Sentiment} ({Confidence:P2}) for {Length} chars in {Ms}ms",
        sentiment, prediction.Probability, request.Text.Length, stopwatch.ElapsedMilliseconds);
    
    return Results.Ok(new PredictResponse(
        Text: request.Text,
        Sentiment: sentiment,
        Confidence: prediction.Probability,
        InferenceMs: stopwatch.ElapsedMilliseconds));
})
.WithName("Predict")
.WithOpenApi();

// Batch prediction endpoint
app.MapPost("/api/predict/batch", (
    BatchPredictRequest request,
    PredictionEnginePool<SentimentInput, SentimentPrediction> pool,
    ILogger<Program> logger) =>
{
    if (request.Texts is null || request.Texts.Count == 0)
    {
        return Results.BadRequest(new { error = "At least one text is required" });
    }
    
    if (request.Texts.Count > 100)
    {
        return Results.BadRequest(new { error = "Maximum batch size is 100" });
    }
    
    var stopwatch = Stopwatch.StartNew();
    
    var results = request.Texts.Select(text =>
    {
        var input = new SentimentInput { Text = text };
        var prediction = pool.Predict("SentimentModel", input);
        
        return new PredictResponse(
            Text: text,
            Sentiment: prediction.IsPositive ? "positive" : "negative",
            Confidence: prediction.Probability,
            InferenceMs: 0); // Individual timing not tracked in batch
    }).ToList();
    
    stopwatch.Stop();
    
    logger.LogInformation(
        "Batch predicted {Count} texts in {Ms}ms",
        results.Count, stopwatch.ElapsedMilliseconds);
    
    return Results.Ok(new BatchPredictResponse(
        Results: results,
        Count: results.Count,
        TotalMs: stopwatch.ElapsedMilliseconds));
})
.WithName("PredictBatch")
.WithOpenApi();

// -----------------------------------------------------------------------------
// Health Check Endpoints
// -----------------------------------------------------------------------------

// Liveness probe - just confirms app is running
app.MapHealthChecks("/health/live", new HealthCheckOptions
{
    Predicate = _ => false // Skip all checks, just confirm app responds
});

// Readiness probe - confirms model is loaded and working
app.MapHealthChecks("/health/ready", new HealthCheckOptions
{
    Predicate = check => check.Tags.Contains("ready"),
    ResponseWriter = async (context, report) =>
    {
        context.Response.ContentType = "application/json";
        var result = new
        {
            status = report.Status.ToString(),
            checks = report.Entries.Select(e => new
            {
                name = e.Key,
                status = e.Value.Status.ToString(),
                description = e.Value.Description
            })
        };
        await context.Response.WriteAsJsonAsync(result);
    }
});

app.Run();
