# Chapter 18: Azure Machine Learning

You've built ML.NET models that run beautifully on your laptop. Your prediction engines return results in microseconds. Everything works.

Then your manager asks: "Can we train this on 50 million records?"

Your laptop fans spin up just thinking about it. Training that took minutes with sample data will take days—or crash entirely when you run out of memory. You need compute that scales, storage that can handle enterprise data volumes, and infrastructure that doesn't depend on your machine being awake.

This is where Azure Machine Learning enters the picture. It's not a replacement for ML.NET or your local development workflow. It's the infrastructure layer that lets you scale beyond what any single machine can handle, while keeping everything in the .NET ecosystem you already know.

In this chapter, we'll go from zero to a fully deployed, production-ready model on Azure ML. You'll learn to provision workspaces, train on cloud compute, version your models properly, deploy to managed endpoints, and—critically—keep your Azure bill from spiraling out of control.

## Why Azure ML for .NET Developers?

Before we dive into setup, let's be clear about when Azure ML makes sense and when it doesn't.

**Azure ML solves real problems:**

- Training on datasets too large for local memory
- Parallel experimentation across hyperparameter combinations
- Reproducible training runs with tracked lineage
- Model versioning and governance for compliance
- Managed deployment without Kubernetes expertise
- Collaboration across data science and engineering teams

**Azure ML might be overkill when:**

- Your data fits comfortably in RAM
- You're deploying to existing .NET applications (just embed the model)
- You're prototyping and iteration speed matters most
- Your organization hasn't adopted Azure

The honest assessment: Azure ML adds complexity. You're trading simplicity for scale, local control for cloud infrastructure. That trade-off makes sense for production enterprise systems; it rarely makes sense for weekend projects or quick experiments.

With that framing, let's build.

## Setting Up Your Azure ML Workspace

The workspace is the foundation of everything in Azure ML. It's the top-level container that holds your data, compute, experiments, models, and endpoints. Think of it as your project's home directory in the cloud.

### Creating a Workspace via Azure Portal

The quickest path for a first workspace:

1. Navigate to portal.azure.com
2. Search for "Machine Learning" in the top search bar
3. Click "Create" and select "Azure Machine Learning"
4. Fill in the required fields:
   - **Subscription**: Your Azure subscription
   - **Resource group**: Create new or use existing
   - **Workspace name**: Something memorable (e.g., `mlworkspace-prod`)
   - **Region**: Choose based on data residency and cost

[FIGURE: Azure portal workspace creation screen showing the required fields: subscription, resource group, workspace name, and region selectors]

Behind the scenes, Azure creates several dependent resources:

- **Storage Account**: For datasets, model artifacts, logs
- **Key Vault**: For secrets and connection strings
- **Application Insights**: For monitoring and telemetry
- **Container Registry**: For Docker images (created on first use)

These resources are created automatically with reasonable defaults. You can customize them, but the defaults work fine for most scenarios.

### Creating a Workspace via C# SDK

For infrastructure-as-code approaches, use the Azure.ResourceManager.MachineLearning package:

```csharp
using Azure;
using Azure.Identity;
using Azure.ResourceManager;
using Azure.ResourceManager.MachineLearning;
using Azure.ResourceManager.MachineLearning.Models;
using Azure.ResourceManager.Resources;

public class WorkspaceManager
{
    private readonly ArmClient _armClient;
    
    public WorkspaceManager()
    {
        // Uses Azure CLI credentials, Managed Identity, or environment variables
        _armClient = new ArmClient(new DefaultAzureCredential());
    }
    
    public async Task<MachineLearningWorkspaceResource> CreateWorkspaceAsync(
        string subscriptionId,
        string resourceGroupName,
        string workspaceName,
        string location)
    {
        var subscription = await _armClient.GetSubscriptionResource(
            new ResourceIdentifier($"/subscriptions/{subscriptionId}"))
            .GetAsync();
            
        var resourceGroup = await subscription.Value
            .GetResourceGroupAsync(resourceGroupName);
        
        var workspaceData = new MachineLearningWorkspaceData(location)
        {
            Description = "ML workspace for C# data science projects",
            FriendlyName = workspaceName,
            // Storage, KeyVault, AppInsights are auto-created if not specified
        };
        
        var operation = await resourceGroup.Value
            .GetMachineLearningWorkspaces()
            .CreateOrUpdateAsync(WaitUntil.Completed, workspaceName, workspaceData);
            
        Console.WriteLine($"Workspace created: {operation.Value.Data.Name}");
        return operation.Value;
    }
}
```

### The Azure ML Client for Day-to-Day Operations

For most operations—submitting jobs, registering models, managing endpoints—you'll use the `Azure.AI.MachineLearning` client library:

```csharp
using Azure.AI.MachineLearning;
using Azure.Identity;

public class AzureMLClient
{
    private readonly MLClient _client;
    
    public AzureMLClient(string subscriptionId, string resourceGroup, string workspaceName)
    {
        _client = new MLClient(
            new DefaultAzureCredential(),
            subscriptionId,
            resourceGroup,
            workspaceName);
    }
    
    public MLClient Client => _client;
}
```

The `DefaultAzureCredential` class is your friend. It tries multiple authentication methods in order—environment variables, managed identity, Azure CLI, Visual Studio credentials—so your code works both locally and in production without changes.

### Workspace Configuration Best Practices

A few things I've learned the hard way:

**Region selection matters more than you think.** GPU compute isn't available in all regions, and pricing varies significantly. East US and West Europe tend to have the best availability. Check https://azure.microsoft.com/pricing/details/machine-learning/ for current pricing by region.

**Use resource groups deliberately.** Put your ML workspace in its own resource group, or at least separate from unrelated resources. When you inevitably need to clean up or move things, this saves hours.

**Enable soft delete on Key Vault.** Accidentally deleting secrets during workspace cleanup is more common than you'd expect. Soft delete gives you a recovery window.

**Tag everything.** Apply cost allocation tags from day one:

```csharp
var workspaceData = new MachineLearningWorkspaceData(location)
{
    Tags =
    {
        ["project"] = "customer-churn-prediction",
        ["environment"] = "production",
        ["owner"] = "ml-team",
        ["cost-center"] = "engineering"
    }
};
```

You'll thank yourself when finance asks which project is burning through GPU credits.

## Training in the Cloud

Local training has a ceiling: your machine's RAM and CPU. Azure ML removes that ceiling by letting you run training jobs on cloud compute clusters that scale to your needs.

### Understanding Compute Options

Azure ML offers several compute types:

| Compute Type | Best For | Billing |
|--------------|----------|---------|
| Compute Instance | Development, notebooks | Per-hour while running |
| Compute Cluster | Training jobs | Per-hour, auto-scales to zero |
| Serverless Compute | Occasional jobs | Per-job, no management |
| Attached Compute | Existing VMs, Databricks | Your existing resources |

For training, compute clusters are usually the right choice. They scale up when you submit jobs and scale down to zero when idle—you only pay for actual training time.

### Creating a Compute Cluster

```csharp
using Azure.AI.MachineLearning.Models;

public async Task<ComputeResource> CreateComputeClusterAsync(
    MLClient client,
    string clusterName,
    string vmSize = "Standard_DS3_v2",
    int minNodes = 0,
    int maxNodes = 4)
{
    var clusterConfig = new AmlCompute(
        vmSize: vmSize,
        minInstances: minNodes,
        maxInstances: maxNodes,
        idleTimeBeforeScaleDown: TimeSpan.FromMinutes(10))
    {
        Tier = VmTier.Dedicated, // or LowPriority for 60-80% discount
        RemoteLoginPortPublicAccess = RemoteLoginPortPublicAccess.Disabled
    };
    
    var compute = new ComputeResource(clusterConfig)
    {
        Name = clusterName,
        Description = "Training cluster for ML workloads"
    };
    
    var operation = await client.Compute.CreateOrUpdateAsync(
        WaitUntil.Completed, 
        clusterName, 
        compute);
    
    Console.WriteLine($"Cluster created: {operation.Value.Name}");
    Console.WriteLine($"VM Size: {vmSize}, Max Nodes: {maxNodes}");
    
    return operation.Value;
}
```

**VM size selection guide:**

- **Standard_DS3_v2**: Good default for tabular data, 4 cores, 14GB RAM
- **Standard_E4s_v3**: Memory-optimized for larger datasets, 4 cores, 32GB RAM
- **Standard_NC6**: GPU compute for deep learning, 1 K80 GPU
- **Standard_NC24ads_A100_v4**: High-end training, A100 GPU

Start small. You can always scale up if training is too slow, but you can't get money back from overprovisioned compute.

[FIGURE: Diagram showing compute cluster auto-scaling: idle at 0 nodes, scaling up to process a job queue, then scaling back down after job completion]

### Submitting a Training Job

Here's where Azure ML shines: you can submit a training script and let the cloud handle execution, logging, and artifact storage.

First, let's create a training script. This is C# code that will run on the cloud compute:

```csharp
// TrainingJob/Program.cs
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Text.Json;

// Azure ML sets environment variables for input/output paths
var inputPath = Environment.GetEnvironmentVariable("AZUREML_INPUT_DATA") 
    ?? "./data/train.csv";
var outputPath = Environment.GetEnvironmentVariable("AZUREML_OUTPUT_MODEL") 
    ?? "./outputs/model.zip";

Console.WriteLine($"Loading data from: {inputPath}");

var mlContext = new MLContext(seed: 42);

// Load training data
var dataView = mlContext.Data.LoadFromTextFile<CustomerData>(
    inputPath,
    separatorChar: ',',
    hasHeader: true);

// Split for validation
var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

// Build pipeline
var pipeline = mlContext.Transforms.Categorical.OneHotEncoding("ContractEncoded", "Contract")
    .Append(mlContext.Transforms.Concatenate("Features", 
        "Tenure", "MonthlyCharges", "TotalCharges", "ContractEncoded"))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
        labelColumnName: "Churned",
        featureColumnName: "Features"));

Console.WriteLine("Training model...");
var model = pipeline.Fit(split.TrainSet);

// Evaluate
var predictions = model.Transform(split.TestSet);
var metrics = mlContext.BinaryClassification.Evaluate(predictions, "Churned");

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

// Log metrics to Azure ML (picked up automatically from stdout)
var metricsJson = JsonSerializer.Serialize(new
{
    accuracy = metrics.Accuracy,
    auc = metrics.AreaUnderRocCurve,
    f1 = metrics.F1Score
});
Console.WriteLine($"AZUREML_METRICS:{metricsJson}");

// Save model
Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
mlContext.Model.Save(model, dataView.Schema, outputPath);
Console.WriteLine($"Model saved to: {outputPath}");

public class CustomerData
{
    [LoadColumn(0)] public float Tenure { get; set; }
    [LoadColumn(1)] public float MonthlyCharges { get; set; }
    [LoadColumn(2)] public float TotalCharges { get; set; }
    [LoadColumn(3)] public string Contract { get; set; } = "";
    [LoadColumn(4), ColumnName("Churned")] public bool Churned { get; set; }
}
```

Now, submit this job from your local machine:

```csharp
public async Task<Job> SubmitTrainingJobAsync(
    MLClient client,
    string experimentName,
    string computeTarget,
    string dataAssetName)
{
    // Reference the registered data asset
    var dataInput = new UriFileJobInput(
        new Uri($"azureml://datastores/workspaceblobstore/paths/data/{dataAssetName}"));
    
    // Define the job
    var jobProperties = new CommandJob(
        command: "dotnet run --project TrainingJob",
        environmentId: "azureml:dotnet-8-ml:1", // Custom or curated environment
        computeId: $"azureml:{computeTarget}")
    {
        DisplayName = $"churn-training-{DateTime.UtcNow:yyyyMMdd-HHmmss}",
        ExperimentName = experimentName,
        Inputs =
        {
            ["training_data"] = dataInput
        },
        Outputs =
        {
            ["model"] = new UriFileJobOutput()
        },
        EnvironmentVariables =
        {
            ["AZUREML_INPUT_DATA"] = "${{inputs.training_data}}",
            ["AZUREML_OUTPUT_MODEL"] = "${{outputs.model}}/model.zip"
        }
    };
    
    var job = new Job(jobProperties);
    
    var operation = await client.Jobs.CreateOrUpdateAsync(
        WaitUntil.Completed,
        job.Name!,
        job);
    
    Console.WriteLine($"Job submitted: {operation.Value.Name}");
    Console.WriteLine($"Status: {operation.Value.Properties.Status}");
    
    return operation.Value;
}
```

### Creating a Custom Environment

Azure ML environments specify the runtime for your training jobs. For .NET, you'll need a custom environment:

```csharp
public async Task<EnvironmentVersion> CreateDotNetEnvironmentAsync(MLClient client)
{
    var dockerfile = @"
FROM mcr.microsoft.com/dotnet/sdk:8.0

# Install ML.NET dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY TrainingJob/ ./TrainingJob/

# Restore dependencies
RUN dotnet restore TrainingJob/TrainingJob.csproj

# Build release
RUN dotnet build TrainingJob/TrainingJob.csproj -c Release

ENTRYPOINT [""dotnet"", ""run"", ""--project"", ""TrainingJob"", ""-c"", ""Release""]
";

    var envProperties = new EnvironmentVersionProperties
    {
        Image = null, // Using Dockerfile instead
        BuildContext = new BuildContext
        {
            DockerfilePath = "Dockerfile",
            ContextUri = new Uri("https://your-storage/build-context.zip")
        },
        Description = ".NET 8 environment with ML.NET for training jobs"
    };
    
    var environment = new EnvironmentVersion(envProperties)
    {
        Name = "dotnet-8-ml",
        Version = "1"
    };
    
    var operation = await client.Environments.CreateOrUpdateAsync(
        WaitUntil.Completed,
        "dotnet-8-ml",
        "1",
        environment);
    
    return operation.Value;
}
```

### Monitoring Training Jobs

Once submitted, you can monitor job progress:

```csharp
public async Task MonitorJobAsync(MLClient client, string jobName)
{
    while (true)
    {
        var job = await client.Jobs.GetAsync(jobName);
        var status = job.Value.Properties.Status;
        
        Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] Status: {status}");
        
        if (status == JobStatus.Completed || 
            status == JobStatus.Failed || 
            status == JobStatus.Canceled)
        {
            if (status == JobStatus.Completed)
            {
                // Fetch metrics from job outputs
                Console.WriteLine("Training completed successfully!");
            }
            else
            {
                Console.WriteLine($"Training ended with status: {status}");
            }
            break;
        }
        
        await Task.Delay(TimeSpan.FromSeconds(30));
    }
}
```

[FIGURE: Azure ML Studio job monitoring view showing the run metrics chart, resource utilization, and logs panel]

## Model Registry and Versioning

Training a good model is only half the battle. You need to track what you trained, when, with what data, and which version is currently in production. Azure ML's model registry handles this.

### Registering a Model

After training completes, register the model artifact:

```csharp
public async Task<ModelVersion> RegisterModelAsync(
    MLClient client,
    string modelName,
    string modelPath,
    string description,
    Dictionary<string, string>? tags = null,
    Dictionary<string, string>? properties = null)
{
    var modelProperties = new ModelVersionProperties
    {
        Description = description,
        ModelType = "mlnet",
        ModelUri = new Uri(modelPath),
        Tags = tags ?? new Dictionary<string, string>(),
        Properties = properties ?? new Dictionary<string, string>()
    };
    
    // Add standard metadata
    modelProperties.Properties["framework"] = "ML.NET";
    modelProperties.Properties["runtime"] = ".NET 8";
    modelProperties.Properties["registered_at"] = DateTime.UtcNow.ToString("O");
    
    var modelVersion = new ModelVersion(modelProperties);
    
    // Versioning is automatic - each registration gets an incremented version
    var operation = await client.Models.CreateOrUpdateAsync(
        WaitUntil.Completed,
        modelName,
        modelVersion);
    
    Console.WriteLine($"Registered: {modelName} v{operation.Value.Version}");
    return operation.Value;
}

// Usage after training job completes
await RegisterModelAsync(
    client,
    modelName: "customer-churn-classifier",
    modelPath: "azureml://jobs/{job-id}/outputs/model/model.zip",
    description: "Binary classifier for customer churn prediction",
    tags: new Dictionary<string, string>
    {
        ["dataset"] = "customer_data_v3",
        ["algorithm"] = "SDCA Logistic Regression",
        ["accuracy"] = "0.87"
    });
```

### Model Versioning Strategy

Every registration creates a new version. This gives you full lineage:

```csharp
public async Task ListModelVersionsAsync(MLClient client, string modelName)
{
    Console.WriteLine($"Versions for {modelName}:");
    Console.WriteLine(new string('-', 60));
    
    await foreach (var version in client.Models.GetVersionsAsync(modelName))
    {
        var props = version.Properties;
        Console.WriteLine($"v{version.Version}");
        Console.WriteLine($"  Created: {props.Properties.GetValueOrDefault("registered_at", "unknown")}");
        Console.WriteLine($"  Description: {props.Description}");
        
        if (props.Tags.TryGetValue("accuracy", out var accuracy))
        {
            Console.WriteLine($"  Accuracy: {accuracy}");
        }
        
        Console.WriteLine();
    }
}
```

Output:
```
Versions for customer-churn-classifier:
------------------------------------------------------------
v3
  Created: 2026-02-14T10:30:00Z
  Description: Binary classifier for customer churn prediction
  Accuracy: 0.87

v2
  Created: 2026-02-10T14:15:00Z
  Description: Updated with new features
  Accuracy: 0.84

v1
  Created: 2026-02-01T09:00:00Z
  Description: Initial model
  Accuracy: 0.79
```

### Model Comparison and Promotion

Build workflows that compare model versions before promotion:

```csharp
public class ModelPromoter
{
    private readonly MLClient _client;
    
    public async Task<bool> ShouldPromoteAsync(
        string modelName,
        string candidateVersion,
        string productionVersion,
        double minimumImprovementThreshold = 0.01)
    {
        var candidate = await _client.Models.GetVersionAsync(modelName, candidateVersion);
        var production = await _client.Models.GetVersionAsync(modelName, productionVersion);
        
        var candidateAccuracy = double.Parse(
            candidate.Value.Properties.Tags.GetValueOrDefault("accuracy", "0"));
        var productionAccuracy = double.Parse(
            production.Value.Properties.Tags.GetValueOrDefault("accuracy", "0"));
        
        var improvement = candidateAccuracy - productionAccuracy;
        
        Console.WriteLine($"Candidate (v{candidateVersion}): {candidateAccuracy:P2}");
        Console.WriteLine($"Production (v{productionVersion}): {productionAccuracy:P2}");
        Console.WriteLine($"Improvement: {improvement:P2}");
        
        if (improvement >= minimumImprovementThreshold)
        {
            Console.WriteLine($"✓ Candidate exceeds threshold ({minimumImprovementThreshold:P2})");
            return true;
        }
        
        Console.WriteLine($"✗ Improvement below threshold");
        return false;
    }
    
    public async Task PromoteToProductionAsync(string modelName, string version)
    {
        // Update model tags to mark as production
        var model = await _client.Models.GetVersionAsync(modelName, version);
        model.Value.Properties.Tags["stage"] = "production";
        model.Value.Properties.Tags["promoted_at"] = DateTime.UtcNow.ToString("O");
        
        await _client.Models.CreateOrUpdateAsync(
            WaitUntil.Completed,
            modelName,
            model.Value);
        
        Console.WriteLine($"Promoted {modelName} v{version} to production");
    }
}
```

## Managed Endpoints

A trained model sitting in a registry doesn't provide value. It needs to serve predictions. Azure ML managed endpoints handle deployment, scaling, and routing without requiring you to manage Kubernetes.

### Creating a Managed Online Endpoint

```csharp
public async Task<OnlineEndpoint> CreateEndpointAsync(
    MLClient client,
    string endpointName,
    string description)
{
    var endpointProperties = new OnlineEndpointProperties(AuthMode.Key)
    {
        Description = description,
        // Properties for traffic management
        Properties =
        {
            ["enable-access-logs"] = "true"
        }
    };
    
    var endpoint = new OnlineEndpoint(endpointProperties)
    {
        Name = endpointName,
        Location = "eastus", // Should match workspace location
        Tags =
        {
            ["project"] = "customer-churn",
            ["environment"] = "production"
        }
    };
    
    var operation = await client.OnlineEndpoints.CreateOrUpdateAsync(
        WaitUntil.Completed,
        endpointName,
        endpoint);
    
    Console.WriteLine($"Endpoint created: {operation.Value.Name}");
    Console.WriteLine($"Scoring URI: {operation.Value.Properties.ScoringUri}");
    
    return operation.Value;
}
```

### Deploying a Model to the Endpoint

```csharp
public async Task<OnlineDeployment> DeployModelAsync(
    MLClient client,
    string endpointName,
    string deploymentName,
    string modelName,
    string modelVersion)
{
    var deploymentProperties = new ManagedOnlineDeploymentProperties
    {
        Model = $"azureml:{modelName}:{modelVersion}",
        InstanceType = "Standard_DS2_v2",
        InstanceCount = 1,
        // Custom scoring script for ML.NET model
        CodeConfiguration = new CodeConfiguration("./scoring", "score.cs"),
        EnvironmentId = "azureml:dotnet-8-inference:1",
        RequestSettings = new OnlineRequestSettings
        {
            MaxQueueWait = TimeSpan.FromSeconds(30),
            RequestTimeout = TimeSpan.FromSeconds(10),
            MaxConcurrentRequestsPerInstance = 10
        },
        ScaleSettings = new DefaultScaleSettings() // Manual scaling
    };
    
    var deployment = new OnlineDeployment(deploymentProperties)
    {
        Name = deploymentName,
        EndpointName = endpointName
    };
    
    var operation = await client.OnlineDeployments.CreateOrUpdateAsync(
        WaitUntil.Completed,
        endpointName,
        deploymentName,
        deployment);
    
    Console.WriteLine($"Deployment created: {deploymentName}");
    return operation.Value;
}
```

### The Scoring Script

For ML.NET models, create a scoring entry point:

```csharp
// scoring/score.cs
using Microsoft.ML;
using System.Text.Json;

public static class Scorer
{
    private static PredictionEngine<CustomerData, ChurnPrediction>? _engine;
    private static readonly object _lock = new();
    
    public static void Init()
    {
        var modelPath = Environment.GetEnvironmentVariable("AZUREML_MODEL_DIR") 
            + "/model.zip";
        
        var mlContext = new MLContext();
        var model = mlContext.Model.Load(modelPath, out _);
        _engine = mlContext.Model.CreatePredictionEngine<CustomerData, ChurnPrediction>(model);
        
        Console.WriteLine("Model loaded successfully");
    }
    
    public static string Run(string input)
    {
        if (_engine == null)
        {
            lock (_lock)
            {
                if (_engine == null) Init();
            }
        }
        
        var request = JsonSerializer.Deserialize<PredictionRequest>(input);
        
        var predictions = request!.Data
            .Select(d => _engine!.Predict(d))
            .Select(p => new
            {
                willChurn = p.PredictedLabel,
                probability = p.Probability,
                score = p.Score
            })
            .ToList();
        
        return JsonSerializer.Serialize(new { predictions });
    }
}

public class PredictionRequest
{
    public List<CustomerData> Data { get; set; } = new();
}

public class ChurnPrediction
{
    public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
```

### Calling the Endpoint

```csharp
public class ChurnPredictionClient
{
    private readonly HttpClient _httpClient;
    private readonly string _endpointUri;
    
    public ChurnPredictionClient(string endpointUri, string apiKey)
    {
        _httpClient = new HttpClient();
        _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        _httpClient.DefaultRequestHeaders.Add("Content-Type", "application/json");
        _endpointUri = endpointUri;
    }
    
    public async Task<List<ChurnPrediction>> PredictAsync(List<CustomerData> customers)
    {
        var request = new { data = customers };
        var json = JsonSerializer.Serialize(request);
        var content = new StringContent(json, Encoding.UTF8, "application/json");
        
        var response = await _httpClient.PostAsync(_endpointUri, content);
        response.EnsureSuccessStatusCode();
        
        var responseJson = await response.Content.ReadAsStringAsync();
        var result = JsonSerializer.Deserialize<PredictionResponse>(responseJson);
        
        return result!.Predictions;
    }
}

// Usage
var client = new ChurnPredictionClient(
    "https://churn-endpoint.eastus.inference.ml.azure.com/score",
    apiKey: Environment.GetEnvironmentVariable("AZUREML_ENDPOINT_KEY")!);

var customers = new List<CustomerData>
{
    new() { Tenure = 12, MonthlyCharges = 79.99f, TotalCharges = 959.88f, Contract = "Month-to-month" },
    new() { Tenure = 48, MonthlyCharges = 89.99f, TotalCharges = 4319.52f, Contract = "Two year" }
};

var predictions = await client.PredictAsync(customers);

foreach (var (customer, prediction) in customers.Zip(predictions))
{
    Console.WriteLine($"Tenure: {customer.Tenure} months, Contract: {customer.Contract}");
    Console.WriteLine($"  Churn Risk: {(prediction.PredictedLabel ? "HIGH" : "LOW")}");
    Console.WriteLine($"  Probability: {prediction.Probability:P1}");
}
```

### Blue-Green Deployments

Managed endpoints support traffic splitting for safe rollouts:

```csharp
public async Task UpdateTrafficAsync(
    MLClient client,
    string endpointName,
    Dictionary<string, int> trafficAllocation)
{
    var endpoint = await client.OnlineEndpoints.GetAsync(endpointName);
    
    // Clear existing traffic
    endpoint.Value.Properties.Traffic.Clear();
    
    // Set new allocation
    foreach (var (deployment, percentage) in trafficAllocation)
    {
        endpoint.Value.Properties.Traffic[deployment] = percentage;
    }
    
    await client.OnlineEndpoints.CreateOrUpdateAsync(
        WaitUntil.Completed,
        endpointName,
        endpoint.Value);
    
    Console.WriteLine("Traffic updated:");
    foreach (var (deployment, percentage) in trafficAllocation)
    {
        Console.WriteLine($"  {deployment}: {percentage}%");
    }
}

// Gradual rollout example
await UpdateTrafficAsync(client, "churn-endpoint", new()
{
    ["blue-v2"] = 90,   // Current production
    ["green-v3"] = 10   // New version getting 10% of traffic
});

// After validation, complete the rollout
await UpdateTrafficAsync(client, "churn-endpoint", new()
{
    ["green-v3"] = 100
});
```

[FIGURE: Blue-green deployment diagram showing traffic split between two deployment versions, with monitoring dashboard comparing latency and error rates]

## Cost Optimization Strategies

Azure ML can get expensive fast. Here's what actually works to control costs.

### Use Low-Priority Compute for Training

Low-priority (spot) VMs cost 60-80% less but can be preempted. For training jobs that can resume from checkpoints, this is almost always the right choice:

```csharp
var clusterConfig = new AmlCompute(
    vmSize: "Standard_NC6",
    minInstances: 0,
    maxInstances: 4)
{
    Tier = VmTier.LowPriority, // 60-80% savings
    IdleTimeBeforeScaleDown = TimeSpan.FromMinutes(5)
};
```

For jobs that can't tolerate preemption, use dedicated VMs. But most training jobs should use spot instances with checkpointing.

### Aggressive Idle Timeout

Set compute clusters to scale down quickly:

```csharp
// Scale to zero after 5 minutes of idleness
IdleTimeBeforeScaleDown = TimeSpan.FromMinutes(5)
```

The default is often 30+ minutes. You're paying for that idle time.

### Right-Size Your Instances

For inference endpoints, match instance size to actual load:

```csharp
public async Task AnalyzeEndpointMetricsAsync(string endpointName)
{
    // Fetch metrics from Application Insights
    var metrics = await GetEndpointMetricsAsync(endpointName, TimeSpan.FromDays(7));
    
    Console.WriteLine($"Endpoint: {endpointName}");
    Console.WriteLine($"  Avg Requests/min: {metrics.AvgRequestsPerMinute:F1}");
    Console.WriteLine($"  P95 Latency: {metrics.P95LatencyMs:F0}ms");
    Console.WriteLine($"  CPU Utilization: {metrics.AvgCpuPercent:F1}%");
    Console.WriteLine($"  Memory Utilization: {metrics.AvgMemoryPercent:F1}%");
    
    // Recommendations
    if (metrics.AvgCpuPercent < 20 && metrics.AvgMemoryPercent < 30)
    {
        Console.WriteLine("\n⚠️ Instance appears oversized. Consider downsizing.");
    }
    
    if (metrics.AvgCpuPercent > 80)
    {
        Console.WriteLine("\n⚠️ CPU constrained. Consider upsizing or adding instances.");
    }
}
```

### Autoscaling for Variable Load

Configure autoscaling to match capacity with demand:

```csharp
var scaleSettings = new TargetUtilizationScaleSettings
{
    MinInstances = 1,
    MaxInstances = 10,
    TargetUtilizationPercentage = 70,
    PollingInterval = TimeSpan.FromSeconds(30),
    CooldownPeriod = TimeSpan.FromMinutes(5)
};
```

### Cost Monitoring and Alerts

Set up budget alerts before you get a surprise bill:

```csharp
// Using Azure Cost Management API
public async Task SetupBudgetAlertAsync(
    string resourceGroupName,
    decimal monthlyBudget,
    string alertEmail)
{
    var budget = new Budget
    {
        Category = "Cost",
        Amount = monthlyBudget,
        TimeGrain = TimeGrain.Monthly,
        TimePeriod = new BudgetTimePeriod
        {
            StartDate = new DateTime(DateTime.Now.Year, DateTime.Now.Month, 1),
            EndDate = new DateTime(DateTime.Now.Year + 1, DateTime.Now.Month, 1)
        },
        Notifications = new Dictionary<string, Notification>
        {
            ["actual_80_percent"] = new Notification
            {
                Enabled = true,
                Operator = NotificationOperator.GreaterThan,
                Threshold = 80,
                ContactEmails = new[] { alertEmail },
                NotificationThresholdType = ThresholdType.Actual
            },
            ["forecasted_100_percent"] = new Notification
            {
                Enabled = true,
                Operator = NotificationOperator.GreaterThan,
                Threshold = 100,
                ContactEmails = new[] { alertEmail },
                NotificationThresholdType = ThresholdType.Forecasted
            }
        }
    };
    
    Console.WriteLine($"Budget alert set: ${monthlyBudget}/month for {resourceGroupName}");
}
```

### Real Cost Numbers

Here's what typical workloads actually cost (US East, as of 2026):

| Workload | Configuration | Monthly Cost |
|----------|--------------|--------------|
| Training (occasional) | 10 hours NC6 spot | ~$20 |
| Training (regular) | 40 hours NC6 spot | ~$80 |
| Inference (low traffic) | 1x DS2_v2 | ~$100 |
| Inference (medium) | 2x DS3_v2 | ~$300 |
| Inference (high) | 5x DS3_v2 + autoscale | ~$700 |

The real killer is forgetting to turn things off. A single NC6 VM left running costs $250/month. Set those idle timeouts.

## Project: Train and Deploy a Customer Churn Model on Azure ML

Let's put everything together in a complete, production-ready project. We'll build an end-to-end pipeline that:

1. Uploads training data to Azure ML
2. Trains a model on cloud compute
3. Registers the model with versioning
4. Deploys to a managed endpoint
5. Serves predictions via REST API

### Project Structure

```
ChurnPrediction/
├── src/
│   ├── ChurnPrediction.Training/      # Training job code
│   │   ├── Program.cs
│   │   └── ChurnPrediction.Training.csproj
│   ├── ChurnPrediction.Scoring/       # Inference code
│   │   ├── Scorer.cs
│   │   └── ChurnPrediction.Scoring.csproj
│   └── ChurnPrediction.Client/        # Endpoint client
│       ├── ChurnClient.cs
│       └── ChurnPrediction.Client.csproj
├── pipelines/
│   └── AzureMLPipeline.cs             # Orchestration
├── data/
│   └── customer_churn.csv
└── ChurnPrediction.sln
```

### Step 1: Define Data Classes

```csharp
// Shared data contracts
namespace ChurnPrediction.Shared;

public class CustomerData
{
    public string CustomerId { get; set; } = "";
    public int Tenure { get; set; }
    public string Contract { get; set; } = "";
    public string PaymentMethod { get; set; } = "";
    public float MonthlyCharges { get; set; }
    public float TotalCharges { get; set; }
    public bool HasOnlineSecurity { get; set; }
    public bool HasTechSupport { get; set; }
    public bool Churned { get; set; }
}

public class ChurnPrediction
{
    public bool WillChurn { get; set; }
    public float ChurnProbability { get; set; }
    public string RiskLevel => ChurnProbability switch
    {
        > 0.7f => "High",
        > 0.4f => "Medium",
        _ => "Low"
    };
}
```

### Step 2: The Training Job

```csharp
// ChurnPrediction.Training/Program.cs
using Microsoft.ML;
using Microsoft.ML.Data;
using ChurnPrediction.Shared;
using System.Text.Json;

Console.WriteLine("=== Churn Prediction Training Job ===");
Console.WriteLine($"Started at: {DateTime.UtcNow:O}");

// Azure ML provides these via environment variables
var inputPath = Environment.GetEnvironmentVariable("INPUT_DATA_PATH") 
    ?? "data/customer_churn.csv";
var outputDir = Environment.GetEnvironmentVariable("OUTPUT_MODEL_DIR") 
    ?? "outputs";

Console.WriteLine($"Input: {inputPath}");
Console.WriteLine($"Output: {outputDir}");

var mlContext = new MLContext(seed: 42);

// Load data
Console.WriteLine("\n[1/5] Loading data...");
var dataView = mlContext.Data.LoadFromTextFile<CustomerDataML>(
    inputPath,
    separatorChar: ',',
    hasHeader: true);

var rowCount = dataView.GetRowCount() ?? 0;
Console.WriteLine($"Loaded {rowCount:N0} records");

// Split data
Console.WriteLine("\n[2/5] Splitting data...");
var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 42);
Console.WriteLine($"Training set: {split.TrainSet.GetRowCount():N0} records");
Console.WriteLine($"Test set: {split.TestSet.GetRowCount():N0} records");

// Build pipeline
Console.WriteLine("\n[3/5] Building pipeline...");
var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "ContractEncoded", inputColumnName: "Contract")
    .Append(mlContext.Transforms.Categorical.OneHotEncoding(
        outputColumnName: "PaymentEncoded", inputColumnName: "PaymentMethod"))
    .Append(mlContext.Transforms.Concatenate("Features",
        "Tenure", "MonthlyCharges", "TotalCharges",
        "HasOnlineSecurity", "HasTechSupport",
        "ContractEncoded", "PaymentEncoded"))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(
        labelColumnName: "Churned",
        featureColumnName: "Features",
        maximumNumberOfIterations: 100));

// Train
Console.WriteLine("\n[4/5] Training model...");
var stopwatch = System.Diagnostics.Stopwatch.StartNew();
var model = pipeline.Fit(split.TrainSet);
stopwatch.Stop();
Console.WriteLine($"Training completed in {stopwatch.Elapsed.TotalSeconds:F1} seconds");

// Evaluate
Console.WriteLine("\n[5/5] Evaluating model...");
var predictions = model.Transform(split.TestSet);
var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "Churned");

Console.WriteLine("\n=== Model Metrics ===");
Console.WriteLine($"Accuracy:    {metrics.Accuracy:P2}");
Console.WriteLine($"AUC:         {metrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score:    {metrics.F1Score:P2}");
Console.WriteLine($"Precision:   {metrics.PositivePrecision:P2}");
Console.WriteLine($"Recall:      {metrics.PositiveRecall:P2}");

// Log metrics for Azure ML (parsed from stdout)
var metricsOutput = new
{
    accuracy = metrics.Accuracy,
    auc = metrics.AreaUnderRocCurve,
    f1_score = metrics.F1Score,
    precision = metrics.PositivePrecision,
    recall = metrics.PositiveRecall,
    training_time_seconds = stopwatch.Elapsed.TotalSeconds,
    training_samples = split.TrainSet.GetRowCount(),
    test_samples = split.TestSet.GetRowCount()
};
Console.WriteLine($"\nAZUREML_LOG_METRICS:{JsonSerializer.Serialize(metricsOutput)}");

// Save model
Directory.CreateDirectory(outputDir);
var modelPath = Path.Combine(outputDir, "churn_model.zip");
mlContext.Model.Save(model, dataView.Schema, modelPath);
Console.WriteLine($"\nModel saved: {modelPath}");

// Also save the metrics
var metricsPath = Path.Combine(outputDir, "metrics.json");
File.WriteAllText(metricsPath, JsonSerializer.Serialize(metricsOutput, 
    new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Metrics saved: {metricsPath}");

Console.WriteLine($"\nCompleted at: {DateTime.UtcNow:O}");

// ML.NET data class (with LoadColumn attributes)
public class CustomerDataML
{
    [LoadColumn(0)] public string CustomerId { get; set; } = "";
    [LoadColumn(1)] public float Tenure { get; set; }
    [LoadColumn(2)] public string Contract { get; set; } = "";
    [LoadColumn(3)] public string PaymentMethod { get; set; } = "";
    [LoadColumn(4)] public float MonthlyCharges { get; set; }
    [LoadColumn(5)] public float TotalCharges { get; set; }
    [LoadColumn(6)] public bool HasOnlineSecurity { get; set; }
    [LoadColumn(7)] public bool HasTechSupport { get; set; }
    [LoadColumn(8), ColumnName("Churned")] public bool Churned { get; set; }
}
```

### Step 3: The Pipeline Orchestrator

```csharp
// pipelines/AzureMLPipeline.cs
using Azure;
using Azure.AI.MachineLearning;
using Azure.AI.MachineLearning.Models;
using Azure.Identity;

namespace ChurnPrediction.Pipelines;

public class ChurnModelPipeline
{
    private readonly MLClient _client;
    private readonly string _computeTarget;
    private readonly string _environmentName;
    
    public ChurnModelPipeline(
        string subscriptionId,
        string resourceGroup,
        string workspaceName,
        string computeTarget = "cpu-cluster",
        string environmentName = "dotnet-8-ml")
    {
        _client = new MLClient(
            new DefaultAzureCredential(),
            subscriptionId,
            resourceGroup,
            workspaceName);
        _computeTarget = computeTarget;
        _environmentName = environmentName;
    }
    
    public async Task<string> RunFullPipelineAsync(
        string dataPath,
        string experimentName = "churn-prediction")
    {
        Console.WriteLine("=== Churn Model Pipeline ===\n");
        
        // 1. Upload data
        Console.WriteLine("[1/5] Uploading training data...");
        var dataUri = await UploadDataAsync(dataPath);
        Console.WriteLine($"Data uploaded to: {dataUri}\n");
        
        // 2. Submit training job
        Console.WriteLine("[2/5] Submitting training job...");
        var jobName = await SubmitTrainingJobAsync(dataUri, experimentName);
        Console.WriteLine($"Job submitted: {jobName}\n");
        
        // 3. Wait for completion
        Console.WriteLine("[3/5] Waiting for training to complete...");
        var metrics = await WaitForJobCompletionAsync(jobName);
        Console.WriteLine($"Training complete. Accuracy: {metrics.Accuracy:P2}\n");
        
        // 4. Register model
        Console.WriteLine("[4/5] Registering model...");
        var modelVersion = await RegisterModelAsync(jobName, metrics);
        Console.WriteLine($"Model registered: v{modelVersion}\n");
        
        // 5. Deploy to endpoint
        Console.WriteLine("[5/5] Deploying to endpoint...");
        var endpoint = await DeployModelAsync(modelVersion);
        Console.WriteLine($"Deployed to: {endpoint}\n");
        
        return endpoint;
    }
    
    private async Task<string> UploadDataAsync(string localPath)
    {
        var blobClient = _client.Datastores.GetDefault();
        var remotePath = $"data/churn/{DateTime.UtcNow:yyyyMMdd-HHmmss}/customer_churn.csv";
        
        await blobClient.UploadFileAsync(localPath, remotePath);
        
        return $"azureml://datastores/workspaceblobstore/paths/{remotePath}";
    }
    
    private async Task<string> SubmitTrainingJobAsync(string dataUri, string experimentName)
    {
        var jobId = $"churn-train-{DateTime.UtcNow:yyyyMMddHHmmss}";
        
        var jobProperties = new CommandJob(
            command: "dotnet run --project ChurnPrediction.Training -c Release",
            environmentId: $"azureml:{_environmentName}:latest",
            computeId: $"azureml:{_computeTarget}")
        {
            DisplayName = jobId,
            ExperimentName = experimentName,
            Inputs =
            {
                ["training_data"] = new UriFileJobInput(new Uri(dataUri))
            },
            Outputs =
            {
                ["model_output"] = new UriFileJobOutput()
            },
            EnvironmentVariables =
            {
                ["INPUT_DATA_PATH"] = "${{inputs.training_data}}",
                ["OUTPUT_MODEL_DIR"] = "${{outputs.model_output}}"
            }
        };
        
        var job = new Job(jobProperties) { Name = jobId };
        
        await _client.Jobs.CreateOrUpdateAsync(WaitUntil.Started, jobId, job);
        
        return jobId;
    }
    
    private async Task<TrainingMetrics> WaitForJobCompletionAsync(string jobName)
    {
        while (true)
        {
            var job = await _client.Jobs.GetAsync(jobName);
            var status = job.Value.Properties.Status;
            
            Console.Write($"\r  Status: {status}          ");
            
            if (status == JobStatus.Completed)
            {
                Console.WriteLine();
                
                // Parse metrics from job outputs
                return await ExtractMetricsAsync(jobName);
            }
            
            if (status == JobStatus.Failed || status == JobStatus.Canceled)
            {
                throw new Exception($"Job {status}: {job.Value.Properties.StatusMessage}");
            }
            
            await Task.Delay(TimeSpan.FromSeconds(30));
        }
    }
    
    private async Task<TrainingMetrics> ExtractMetricsAsync(string jobName)
    {
        // In practice, read from job outputs or logs
        // Simplified for example
        return new TrainingMetrics
        {
            Accuracy = 0.87,
            Auc = 0.92,
            F1Score = 0.84
        };
    }
    
    private async Task<string> RegisterModelAsync(string jobName, TrainingMetrics metrics)
    {
        var modelProperties = new ModelVersionProperties
        {
            Description = "Customer churn prediction model",
            ModelUri = new Uri($"azureml://jobs/{jobName}/outputs/model_output/churn_model.zip"),
            Tags =
            {
                ["accuracy"] = metrics.Accuracy.ToString("F4"),
                ["auc"] = metrics.Auc.ToString("F4"),
                ["f1_score"] = metrics.F1Score.ToString("F4"),
                ["trained_date"] = DateTime.UtcNow.ToString("O")
            },
            Properties =
            {
                ["framework"] = "ML.NET",
                ["algorithm"] = "LBFGS Logistic Regression"
            }
        };
        
        var model = new ModelVersion(modelProperties);
        
        var result = await _client.Models.CreateOrUpdateAsync(
            WaitUntil.Completed,
            "churn-classifier",
            model);
        
        return result.Value.Version;
    }
    
    private async Task<string> DeployModelAsync(string modelVersion)
    {
        var endpointName = "churn-prediction-endpoint";
        var deploymentName = $"churn-v{modelVersion}";
        
        // Create endpoint if it doesn't exist
        try
        {
            await _client.OnlineEndpoints.GetAsync(endpointName);
        }
        catch (RequestFailedException ex) when (ex.Status == 404)
        {
            var endpoint = new OnlineEndpoint(
                new OnlineEndpointProperties(AuthMode.Key)
                {
                    Description = "Customer churn prediction endpoint"
                })
            {
                Name = endpointName
            };
            
            await _client.OnlineEndpoints.CreateOrUpdateAsync(
                WaitUntil.Completed, endpointName, endpoint);
        }
        
        // Create deployment
        var deploymentProperties = new ManagedOnlineDeploymentProperties
        {
            Model = $"azureml:churn-classifier:{modelVersion}",
            InstanceType = "Standard_DS2_v2",
            InstanceCount = 1,
            EnvironmentId = $"azureml:{_environmentName}:latest"
        };
        
        var deployment = new OnlineDeployment(deploymentProperties)
        {
            Name = deploymentName,
            EndpointName = endpointName
        };
        
        await _client.OnlineDeployments.CreateOrUpdateAsync(
            WaitUntil.Completed, endpointName, deploymentName, deployment);
        
        // Route all traffic to new deployment
        var endpointResource = await _client.OnlineEndpoints.GetAsync(endpointName);
        endpointResource.Value.Properties.Traffic[deploymentName] = 100;
        await _client.OnlineEndpoints.CreateOrUpdateAsync(
            WaitUntil.Completed, endpointName, endpointResource.Value);
        
        return endpointResource.Value.Properties.ScoringUri?.ToString() ?? "";
    }
}

public class TrainingMetrics
{
    public double Accuracy { get; set; }
    public double Auc { get; set; }
    public double F1Score { get; set; }
}
```

### Step 4: The Client Library

```csharp
// ChurnPrediction.Client/ChurnClient.cs
using System.Net.Http.Json;
using ChurnPrediction.Shared;

namespace ChurnPrediction.Client;

public class ChurnPredictionClient : IDisposable
{
    private readonly HttpClient _httpClient;
    
    public ChurnPredictionClient(string endpointUri, string apiKey)
    {
        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(endpointUri),
            Timeout = TimeSpan.FromSeconds(30)
        };
        _httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
    }
    
    public async Task<List<ChurnPrediction>> PredictAsync(
        IEnumerable<CustomerData> customers,
        CancellationToken ct = default)
    {
        var request = new { data = customers.ToList() };
        
        var response = await _httpClient.PostAsJsonAsync("/score", request, ct);
        response.EnsureSuccessStatusCode();
        
        var result = await response.Content.ReadFromJsonAsync<PredictionResponse>(ct);
        return result?.Predictions ?? new List<ChurnPrediction>();
    }
    
    public async Task<ChurnPrediction> PredictSingleAsync(
        CustomerData customer,
        CancellationToken ct = default)
    {
        var predictions = await PredictAsync(new[] { customer }, ct);
        return predictions.FirstOrDefault() ?? throw new Exception("No prediction returned");
    }
    
    public void Dispose() => _httpClient.Dispose();
    
    private class PredictionResponse
    {
        public List<ChurnPrediction> Predictions { get; set; } = new();
    }
}

// Usage example
public static class Program
{
    public static async Task Main()
    {
        using var client = new ChurnPredictionClient(
            Environment.GetEnvironmentVariable("CHURN_ENDPOINT_URI")!,
            Environment.GetEnvironmentVariable("CHURN_ENDPOINT_KEY")!);
        
        var customer = new CustomerData
        {
            CustomerId = "CUST-12345",
            Tenure = 6,
            Contract = "Month-to-month",
            PaymentMethod = "Electronic check",
            MonthlyCharges = 89.99f,
            TotalCharges = 539.94f,
            HasOnlineSecurity = false,
            HasTechSupport = false
        };
        
        var prediction = await client.PredictSingleAsync(customer);
        
        Console.WriteLine($"Customer: {customer.CustomerId}");
        Console.WriteLine($"Churn Risk: {prediction.RiskLevel}");
        Console.WriteLine($"Probability: {prediction.ChurnProbability:P1}");
        
        if (prediction.WillChurn)
        {
            Console.WriteLine("\n⚠️ Recommend proactive retention outreach");
        }
    }
}
```

### Running the Pipeline

```csharp
// Program.cs - Main entry point
using ChurnPrediction.Pipelines;

var pipeline = new ChurnModelPipeline(
    subscriptionId: Environment.GetEnvironmentVariable("AZURE_SUBSCRIPTION_ID")!,
    resourceGroup: "ml-resources",
    workspaceName: "ml-workspace-prod");

var endpoint = await pipeline.RunFullPipelineAsync(
    dataPath: "data/customer_churn.csv",
    experimentName: "churn-prediction-v2");

Console.WriteLine($"✓ Pipeline complete!");
Console.WriteLine($"Endpoint: {endpoint}");
Console.WriteLine("\nTest with:");
Console.WriteLine($"  curl -X POST {endpoint} \\");
Console.WriteLine("    -H 'Authorization: Bearer $API_KEY' \\");
Console.WriteLine("    -H 'Content-Type: application/json' \\");
Console.WriteLine("    -d '{\"data\": [{\"tenure\": 12, \"monthlyCharges\": 79.99, ...}]}'");
```

## Summary

Azure Machine Learning transforms ML.NET from a local tool into an enterprise platform. You've learned to:

- Provision workspaces and compute clusters using C# SDK
- Submit training jobs that scale beyond your laptop
- Version and register models with full lineage tracking
- Deploy to managed endpoints with zero Kubernetes knowledge
- Control costs through right-sizing, spot instances, and autoscaling

The key insight: Azure ML isn't replacing your ML.NET skills—it's amplifying them. You still write the same training code, the same prediction logic, the same pipeline transformations. Azure ML just runs it at scale and manages the infrastructure you don't want to build yourself.

Start small. Train a single model on cloud compute. Register it. Deploy it. Call the endpoint. Once that workflow is comfortable, you'll have the foundation to build sophisticated MLOps pipelines that would take months to implement from scratch.

## Exercises

**Exercise 1: Environment Optimization**

Create a custom Azure ML environment optimized for ML.NET inference. Your environment should:
- Use the smallest possible base image
- Include only necessary dependencies
- Pre-compile the scoring application
- Measure the cold-start time and compare with a naive approach

What's the minimum container size you can achieve while still running ML.NET inference?

**Exercise 2: Hyperparameter Sweep**

Implement a hyperparameter sweep job that trains the churn model with different configurations:
- L1 regularization: [0.001, 0.01, 0.1, 1.0]
- L2 regularization: [0.001, 0.01, 0.1, 1.0]
- Maximum iterations: [50, 100, 200]

Use Azure ML's sweep functionality to run experiments in parallel. Register only the best-performing model.

**Exercise 3: Model Comparison Dashboard**

Build a C# application that:
- Lists all versions of a registered model
- Compares metrics across versions
- Identifies the best model by a specified metric
- Generates a markdown report summarizing model evolution

Use the Azure ML SDK to pull model metadata and metrics.

**Exercise 4: Cost Analysis Tool**

Create a cost monitoring utility that:
- Queries Azure Cost Management for ML workspace spending
- Breaks down costs by resource type (compute, storage, endpoints)
- Identifies idle resources that should be deleted
- Projects monthly costs based on current usage patterns

Alert when projected costs exceed a configurable threshold.

**Exercise 5: Blue-Green Deployment Pipeline**

Implement a complete blue-green deployment workflow:
1. Deploy new model version to "green" deployment (0% traffic)
2. Run automated validation tests against green deployment
3. Gradually shift traffic (10% → 50% → 100%) over configured time periods
4. Monitor error rates during rollout
5. Automatic rollback if error rate exceeds threshold

Include proper logging and notification at each stage.
