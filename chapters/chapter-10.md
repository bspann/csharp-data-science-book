# Chapter 10: Clustering — Finding Groups

*"The goal is to turn data into information, and information into insight."* — Carly Fiorina

Throughout this book, we've explored supervised learning—training models on labeled data where we know the "right answers." Classification taught us to predict categories; regression taught us to predict continuous values. But what happens when you don't have labels? What if you're staring at a massive customer database and wondering, "Are there natural groups hiding in here that I haven't discovered yet?"

Welcome to **unsupervised learning**, where algorithms discover structure in data without being told what to look for. In this chapter, we'll focus on **clustering**—the art of finding natural groupings in your data. By the end, you'll understand how clustering works, implement it in ML.NET, and complete a real-world customer segmentation project that could transform how a business understands its customers.

## The Shift to Unsupervised Learning

In supervised learning, every training example comes with a label: "This email is spam," "This tumor is malignant," "This house sold for $350,000." The algorithm learns the relationship between features and labels, then applies that knowledge to new data.

Unsupervised learning flips the script. You have data—potentially lots of it—but no labels. No one has told you which customers are "valuable" versus "at-risk." No one has categorized your products into meaningful groups. The algorithm's job is to find patterns, structures, and relationships that exist naturally in the data.

### Why No Labels?

Labels are expensive. Someone has to create them:

- **Manual labeling is time-consuming.** Imagine asking a team to manually categorize 100,000 customers into segments. It would take months.
- **Labels may not exist.** What if you don't know what categories should exist? You can't label data with categories you haven't discovered yet.
- **Labels can be biased.** Human-created labels reflect human assumptions. Unsupervised learning can reveal groupings that humans never considered.

### What Can Unsupervised Learning Do?

Unsupervised learning encompasses several techniques:

| Technique | Purpose | Example |
|-----------|---------|---------|
| **Clustering** | Group similar items together | Customer segmentation |
| **Dimensionality Reduction** | Compress many features into fewer | Visualizing high-dimensional data |
| **Anomaly Detection** | Find unusual data points | Fraud detection |
| **Association Rules** | Find items that co-occur | Market basket analysis |

This chapter focuses on clustering, the most widely used unsupervised technique. We'll touch on dimensionality reduction when we visualize our clusters, setting the stage for a deeper dive in Chapter 11.

## Understanding Clustering

Clustering algorithms group data points so that items within a cluster are more similar to each other than to items in other clusters. Think of it as organizing a drawer full of mixed objects—you naturally group socks with socks, shirts with shirts, without anyone telling you the categories.

### When to Use Clustering

Clustering shines in several scenarios:

**Exploration and Discovery**
You have a new dataset and want to understand its structure. What natural groupings exist? Are there distinct populations hiding in the data?

**Segmentation**
Divide customers, products, or users into meaningful groups for targeted strategies. Marketing teams love this—different segments get different campaigns.

**Preprocessing**
Create cluster assignments as features for supervised learning. "Which cluster does this customer belong to?" becomes a powerful input feature.

**Compression**
Represent each data point by its cluster center, reducing storage and computation requirements.

**Anomaly Detection**
Points that don't fit well into any cluster may be outliers or anomalies worth investigating.

### The Clustering Mindset

Before running any algorithm, ask yourself:

1. **What does "similar" mean for my data?** Distance metrics matter enormously. Two customers might be similar in purchase frequency but different in purchase amount.

2. **How many clusters should exist?** Sometimes domain knowledge guides you (you want exactly 5 customer tiers). Often, you don't know and must discover the right number.

3. **What will I do with the clusters?** Clustering is a means to an end. If you can't interpret or act on the clusters, they're useless.

## The K-Means Algorithm

K-Means is the workhorse of clustering algorithms. It's simple to understand, fast to run, and works well for many real-world problems. Let's build intuition before diving into code.

### How K-Means Works

K-Means partitions n data points into k clusters, where k is specified in advance. The algorithm minimizes the within-cluster variance—points should be close to their cluster's center.

**The Algorithm:**

1. **Initialize:** Choose k random points as initial cluster centers (centroids).

2. **Assign:** For each data point, calculate its distance to every centroid. Assign the point to the nearest centroid's cluster.

3. **Update:** Recalculate each centroid as the mean of all points assigned to that cluster.

4. **Repeat:** Continue steps 2-3 until centroids stop moving (or move negligibly).

[FIGURE: K-Means iteration visualization showing three iterations. In iteration 1, random centroids are placed and points are colored by nearest centroid. In iteration 2, centroids move toward cluster centers and some points change color. In iteration 3, centroids stabilize and clusters are well-defined.]

### A Concrete Example

Imagine plotting customers by two features: annual spending (x-axis) and visit frequency (y-axis). You suspect there are three natural groups. K-Means with k=3 would:

1. Place three random centroids on the plot
2. Color each customer point based on its nearest centroid
3. Move each centroid to the center of its colored points
4. Recolor points based on the new centroid positions
5. Repeat until the centroids stabilize

After convergence, you might discover:
- **Cluster A (blue):** Low spending, low frequency — occasional browsers
- **Cluster B (green):** Moderate spending, high frequency — loyal regulars
- **Cluster C (red):** High spending, moderate frequency — big spenders

### Why K-Means Works

K-Means optimizes an objective function: minimize the sum of squared distances from each point to its assigned centroid. Mathematically:

```
J = Σᵢ Σₓ∈Cᵢ ||x - μᵢ||²
```

Where Cᵢ is cluster i and μᵢ is the centroid of cluster i. Each iteration of assign-update is guaranteed to decrease (or maintain) this objective. The algorithm converges because:

- Assigning points to nearest centroids minimizes distances given fixed centroids
- Moving centroids to cluster means minimizes distances given fixed assignments

### Limitations of K-Means

No algorithm is perfect. K-Means has important limitations:

**Requires specifying k.** You must know (or guess) the number of clusters in advance. We'll address this shortly.

**Sensitive to initialization.** Random initial centroids can lead to poor local optima. Modern implementations run multiple initializations and keep the best result.

**Assumes spherical clusters.** K-Means works best when clusters are roughly circular/spherical. It struggles with elongated or irregular shapes.

**Sensitive to outliers.** A few extreme points can pull centroids away from the true cluster centers.

**Sensitive to feature scales.** Features with larger numeric ranges dominate distance calculations. *Always scale your features.*

## Feature Scaling: A Critical Step

This deserves its own section because it's the most common clustering mistake. Consider two features:

- **Age:** ranges from 18 to 80 (range of 62)
- **Annual Income:** ranges from $20,000 to $200,000 (range of 180,000)

Without scaling, income completely dominates distance calculations. A difference of $1,000 in income would outweigh a difference of 50 years in age, even though both might be equally important for segmentation.

### Scaling Methods

**StandardScaler (Z-score normalization)**
Transforms features to have mean=0 and standard deviation=1:
```
x_scaled = (x - mean) / std_dev
```

**MinMaxScaler (Min-Max normalization)**
Transforms features to a [0, 1] range:
```
x_scaled = (x - min) / (max - min)
```

**For clustering, StandardScaler is usually preferred.** It handles outliers better than MinMax and doesn't artificially compress the data into a fixed range.

```csharp
// In ML.NET, normalization is part of the pipeline
var pipeline = mlContext.Transforms.NormalizeMinMax("Features")
    .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));
```

## K-Means in ML.NET

Let's implement K-Means clustering in ML.NET. We'll start with a simple example before tackling the full customer segmentation project.

### Setting Up

First, define your data structures:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

// Input data class
public class CustomerData
{
    [LoadColumn(0)]
    public float Recency { get; set; }  // Days since last purchase
    
    [LoadColumn(1)]
    public float Frequency { get; set; }  // Number of purchases
    
    [LoadColumn(2)]
    public float Monetary { get; set; }  // Total spending
}

// Prediction output class
public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }
    
    [ColumnName("Score")]
    public float[] Distances { get; set; }  // Distance to each centroid
}
```

### Building the Pipeline

```csharp
var mlContext = new MLContext(seed: 42);

// Load data
IDataView dataView = mlContext.Data.LoadFromTextFile<CustomerData>(
    "customers.csv",
    hasHeader: true,
    separatorChar: ',');

// Build the pipeline
var pipeline = mlContext.Transforms
    // Combine features into a single vector
    .Concatenate("Features", nameof(CustomerData.Recency), 
                             nameof(CustomerData.Frequency), 
                             nameof(CustomerData.Monetary))
    // CRITICAL: Normalize features
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    // Apply K-Means clustering
    .Append(mlContext.Clustering.Trainers.KMeans(
        featureColumnName: "Features",
        numberOfClusters: 4));
```

### Training and Prediction

```csharp
// Train the model
Console.WriteLine("Training K-Means model...");
var model = pipeline.Fit(dataView);

// Make predictions on the same data
var predictions = model.Transform(dataView);

// Convert to enumerable for analysis
var results = mlContext.Data
    .CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false)
    .ToList();

// Count points per cluster
var clusterCounts = results
    .GroupBy(p => p.ClusterId)
    .Select(g => new { Cluster = g.Key, Count = g.Count() })
    .OrderBy(x => x.Cluster);

Console.WriteLine("\nCluster Distribution:");
foreach (var cluster in clusterCounts)
{
    Console.WriteLine($"  Cluster {cluster.Cluster}: {cluster.Count} customers");
}
```

### Accessing Cluster Centroids

After training, you can examine the cluster centroids:

```csharp
// Get the clustering model parameters
VBuffer<float>[] centroids = default;
var kmeansModel = model.LastTransformer.Model;
kmeansModel.GetClusterCentroids(ref centroids, out int k);

Console.WriteLine($"\nCluster Centroids (k={k}):");
for (int i = 0; i < centroids.Length; i++)
{
    var values = centroids[i].GetValues().ToArray();
    Console.WriteLine($"  Cluster {i + 1}: [{string.Join(", ", values.Select(v => v.ToString("F3")))}]");
}
```

## Choosing the Right K

Selecting the number of clusters is both art and science. Too few clusters oversimplify; too many create meaningless distinctions. Two popular methods help guide this decision.

### The Elbow Method

Run K-Means for different values of k and plot the within-cluster sum of squares (WCSS) against k. As k increases, WCSS decreases (more clusters = points closer to centroids). Look for the "elbow"—the point where adding more clusters provides diminishing returns.

```csharp
public static List<(int K, double WCSS)> ComputeElbowData(
    MLContext mlContext,
    IDataView data,
    int maxK)
{
    var results = new List<(int K, double WCSS)>();
    
    for (int k = 2; k <= maxK; k++)
    {
        var pipeline = mlContext.Transforms
            .Concatenate("Features", "Recency", "Frequency", "Monetary")
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: k));
        
        var model = pipeline.Fit(data);
        
        // Calculate WCSS from predictions
        var predictions = model.Transform(data);
        var wcss = CalculateWCSS(mlContext, predictions);
        
        results.Add((k, wcss));
        Console.WriteLine($"k={k}: WCSS={wcss:F2}");
    }
    
    return results;
}

private static double CalculateWCSS(MLContext mlContext, IDataView predictions)
{
    // Sum of minimum distances (each point to its assigned centroid)
    var distances = mlContext.Data
        .CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false)
        .Sum(p => p.Distances.Min());  // Distance to assigned cluster
    
    return distances;
}
```

[FIGURE: Elbow plot showing WCSS (y-axis) versus number of clusters k (x-axis) from 2 to 10. The curve shows a sharp bend at k=4, indicating the optimal number of clusters. Annotation points to the "elbow" at k=4.]

### Silhouette Score

The silhouette score measures how similar each point is to its own cluster compared to other clusters. For each point:

- **a** = average distance to other points in the same cluster
- **b** = average distance to points in the nearest different cluster
- **silhouette** = (b - a) / max(a, b)

Values range from -1 to 1:
- **+1:** Point is well-matched to its cluster, far from others
- **0:** Point is on the boundary between clusters  
- **-1:** Point is probably in the wrong cluster

```csharp
public static double CalculateSilhouetteScore(
    MLContext mlContext,
    IDataView data,
    ITransformer model)
{
    var predictions = model.Transform(data);
    
    // Get features and cluster assignments
    var featuresColumn = predictions.GetColumn<float[]>("Features").ToArray();
    var clusterColumn = predictions.GetColumn<uint>("PredictedLabel").ToArray();
    
    double totalSilhouette = 0;
    int n = featuresColumn.Length;
    
    for (int i = 0; i < n; i++)
    {
        uint myCluster = clusterColumn[i];
        
        // Calculate a: mean distance to same-cluster points
        var sameCluster = Enumerable.Range(0, n)
            .Where(j => j != i && clusterColumn[j] == myCluster)
            .Select(j => EuclideanDistance(featuresColumn[i], featuresColumn[j]))
            .ToList();
        
        double a = sameCluster.Any() ? sameCluster.Average() : 0;
        
        // Calculate b: mean distance to nearest other cluster
        var otherClusters = Enumerable.Range(0, n)
            .Where(j => clusterColumn[j] != myCluster)
            .GroupBy(j => clusterColumn[j])
            .Select(g => g.Select(j => 
                EuclideanDistance(featuresColumn[i], featuresColumn[j])).Average())
            .ToList();
        
        double b = otherClusters.Any() ? otherClusters.Min() : 0;
        
        // Silhouette for this point
        double silhouette = (b - a) / Math.Max(a, b);
        totalSilhouette += double.IsNaN(silhouette) ? 0 : silhouette;
    }
    
    return totalSilhouette / n;
}

private static double EuclideanDistance(float[] a, float[] b)
{
    return Math.Sqrt(a.Zip(b, (x, y) => Math.Pow(x - y, 2)).Sum());
}
```

**Rule of thumb for silhouette scores:**
- **> 0.7:** Strong structure
- **0.5 - 0.7:** Reasonable structure
- **0.25 - 0.5:** Weak structure, may be artificial
- **< 0.25:** No substantial structure

### Combining Methods with Domain Knowledge

Numbers don't tell the whole story. Consider:

- **Business constraints:** Marketing can realistically target 3-5 segments, not 15.
- **Interpretability:** Can you explain each cluster meaningfully?
- **Stability:** Do clusters remain consistent across different data samples?

The best k balances statistical validity with practical utility.

## Interpreting Cluster Results

Finding clusters is only half the battle. You must interpret them—understand *what* each cluster represents and *why* it matters.

### Profiling Clusters

For each cluster, calculate summary statistics:

```csharp
public class ClusterProfile
{
    public uint ClusterId { get; set; }
    public int Count { get; set; }
    public double AvgRecency { get; set; }
    public double AvgFrequency { get; set; }
    public double AvgMonetary { get; set; }
    public string Label { get; set; }  // Human-readable name
}

public static List<ClusterProfile> ProfileClusters(
    List<CustomerData> customers,
    List<ClusterPrediction> predictions)
{
    var combined = customers.Zip(predictions, (c, p) => new { Customer = c, Cluster = p.ClusterId });
    
    return combined
        .GroupBy(x => x.Cluster)
        .Select(g => new ClusterProfile
        {
            ClusterId = g.Key,
            Count = g.Count(),
            AvgRecency = g.Average(x => x.Customer.Recency),
            AvgFrequency = g.Average(x => x.Customer.Frequency),
            AvgMonetary = g.Average(x => x.Customer.Monetary)
        })
        .OrderBy(p => p.ClusterId)
        .ToList();
}
```

### Naming Your Clusters

Transform statistics into stories:

| Cluster | Recency | Frequency | Monetary | Interpretation |
|---------|---------|-----------|----------|----------------|
| 1 | Low (recent) | High | High | **Champions** — Best customers |
| 2 | Medium | Medium | Medium | **Loyal Customers** — Consistent buyers |
| 3 | Low | Low | Low | **New Customers** — Recently acquired |
| 4 | High (long ago) | High | High | **At Risk** — Were great, now quiet |
| 5 | High | Low | Low | **Lost** — Churned customers |

These names make clusters actionable. "Send a win-back campaign to Cluster 4" becomes "Re-engage our At-Risk customers who used to be champions."

## Visualizing Clusters

High-dimensional data is hard to visualize. Customers described by 10+ features can't be plotted directly. We use **dimensionality reduction** to compress features into 2-3 dimensions for visualization.

### Principal Component Analysis (PCA)

PCA finds new axes (principal components) that capture maximum variance in the data. The first component captures the most variance, the second captures the next most (orthogonal to the first), and so on.

```csharp
// Add PCA to reduce dimensions for visualization
var visualizationPipeline = mlContext.Transforms
    .Concatenate("Features", "Recency", "Frequency", "Monetary")
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.Transforms.ProjectToPrincipalComponents(
        outputColumnName: "PCAFeatures",
        inputColumnName: "Features",
        rank: 2))  // Reduce to 2 dimensions
    .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 4));
```

### Creating Visualizations

Export data for visualization in your preferred tool:

```csharp
public static void ExportForVisualization(
    MLContext mlContext,
    IDataView predictions,
    string outputPath)
{
    var data = mlContext.Data.CreateEnumerable<VisualizationData>(
        predictions, reuseRowObject: false);
    
    using var writer = new StreamWriter(outputPath);
    writer.WriteLine("PC1,PC2,Cluster");
    
    foreach (var point in data)
    {
        writer.WriteLine($"{point.PCAFeatures[0]},{point.PCAFeatures[1]},{point.ClusterId}");
    }
}

public class VisualizationData
{
    [ColumnName("PCAFeatures")]
    public float[] PCAFeatures { get; set; }
    
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }
}
```

[FIGURE: Scatter plot showing customer segments after PCA reduction to 2D. Four distinct clusters are visible, colored differently. Cluster 1 (Champions) appears in the upper right, Cluster 2 (Loyal) in the center, Cluster 3 (New) in the lower left, and Cluster 4 (At Risk) in the upper left. Centroids are marked with stars.]

## Common Pitfalls and How to Avoid Them

### The Curse of Dimensionality

As dimensions increase, distances become less meaningful. In 100-dimensional space, most points are roughly equidistant from each other. K-Means (and most distance-based algorithms) struggle in high dimensions.

**Solutions:**
- Reduce dimensions with PCA before clustering
- Select only the most relevant features
- Use domain knowledge to limit input features

### Forgetting to Scale Features

We've emphasized this, but it bears repeating: **unscaled features produce meaningless clusters.** Always normalize.

### Interpreting Random Initialization

K-Means results can vary with different random seeds. Run clustering multiple times and look for consistent patterns, or use `seed` parameter for reproducibility:

```csharp
var mlContext = new MLContext(seed: 42);  // Reproducible results
```

### Assuming Clusters Are "True"

Clusters are a model of reality, not reality itself. Two different algorithms (or different k values) might produce equally valid but different groupings. Validate clusters against external criteria when possible.

### Overfitting the Cluster Count

A model with k = n (one cluster per point) has zero within-cluster variance but tells you nothing. More clusters isn't always better. Optimize for insight, not metrics.

## Project: Customer Segmentation for E-Commerce

Let's apply everything we've learned to a real business problem. You're a data scientist at an e-commerce company. Marketing wants to understand customer segments to personalize campaigns. You have transaction history but no predefined segments—a perfect clustering problem.

### Understanding RFM Analysis

RFM (Recency, Frequency, Monetary) is a proven customer segmentation framework:

- **Recency:** How recently did the customer purchase? (Lower is better)
- **Frequency:** How often do they purchase? (Higher is better)
- **Monetary:** How much do they spend total? (Higher is better)

These three dimensions capture customer value and behavior succinctly.

### The Dataset

We'll work with a realistic e-commerce dataset. Each row represents aggregated customer behavior:

```csv
CustomerId,Recency,Frequency,Monetary,DaysSinceFirstPurchase
C001,5,23,4521.50,365
C002,89,2,125.00,180
C003,12,45,8932.00,720
...
```

### Step 1: Data Preparation

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

public class CustomerRFM
{
    [LoadColumn(0)]
    public string CustomerId { get; set; }
    
    [LoadColumn(1)]
    public float Recency { get; set; }
    
    [LoadColumn(2)]
    public float Frequency { get; set; }
    
    [LoadColumn(3)]
    public float Monetary { get; set; }
    
    [LoadColumn(4)]
    public float DaysSinceFirstPurchase { get; set; }
}

public class CustomerClusterResult
{
    public string CustomerId { get; set; }
    public float Recency { get; set; }
    public float Frequency { get; set; }
    public float Monetary { get; set; }
    public uint ClusterId { get; set; }
    public float[] Distances { get; set; }
    public float[] PCAFeatures { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext(seed: 42);
        
        // Load customer data
        Console.WriteLine("Loading customer data...");
        var dataView = mlContext.Data.LoadFromTextFile<CustomerRFM>(
            "ecommerce_customers.csv",
            hasHeader: true,
            separatorChar: ',');
        
        // Check data quality
        var customers = mlContext.Data
            .CreateEnumerable<CustomerRFM>(dataView, reuseRowObject: false)
            .ToList();
        
        Console.WriteLine($"Loaded {customers.Count} customers");
        Console.WriteLine($"Recency range: {customers.Min(c => c.Recency)} - {customers.Max(c => c.Recency)}");
        Console.WriteLine($"Frequency range: {customers.Min(c => c.Frequency)} - {customers.Max(c => c.Frequency)}");
        Console.WriteLine($"Monetary range: {customers.Min(c => c.Monetary):C0} - {customers.Max(c => c.Monetary):C0}");
```

### Step 2: Finding the Optimal K

```csharp
        // Find optimal k using elbow method
        Console.WriteLine("\n--- Finding Optimal K ---");
        var elbowResults = new List<(int K, double WCSS, double Silhouette)>();
        
        for (int k = 2; k <= 8; k++)
        {
            var testPipeline = mlContext.Transforms
                .Concatenate("Features", "Recency", "Frequency", "Monetary")
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: k));
            
            var testModel = testPipeline.Fit(dataView);
            var testPredictions = testModel.Transform(dataView);
            
            // Calculate metrics
            var metrics = mlContext.Clustering.Evaluate(testPredictions);
            double silhouette = CalculateSilhouetteApprox(mlContext, testPredictions);
            
            elbowResults.Add((k, metrics.AverageDistance, silhouette));
            Console.WriteLine($"k={k}: Avg Distance={metrics.AverageDistance:F4}, Silhouette≈{silhouette:F3}");
        }
        
        // Select k=4 based on elbow and silhouette (or adjust based on results)
        int optimalK = 4;
        Console.WriteLine($"\nSelected k={optimalK} for final model");
```

### Step 3: Training the Final Model

```csharp
        // Build final pipeline with PCA for visualization
        Console.WriteLine("\n--- Training Final Model ---");
        var pipeline = mlContext.Transforms
            .Concatenate("Features", "Recency", "Frequency", "Monetary")
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.Transforms.ProjectToPrincipalComponents(
                outputColumnName: "PCAFeatures",
                inputColumnName: "Features",
                rank: 2))
            .Append(mlContext.Clustering.Trainers.KMeans(
                featureColumnName: "Features",
                numberOfClusters: optimalK));
        
        var model = pipeline.Fit(dataView);
        var predictions = model.Transform(dataView);
        
        // Evaluate
        var finalMetrics = mlContext.Clustering.Evaluate(predictions);
        Console.WriteLine($"Average Distance: {finalMetrics.AverageDistance:F4}");
        Console.WriteLine($"Davies-Bouldin Index: {finalMetrics.DaviesBouldinIndex:F4}");
```

### Step 4: Profiling and Interpreting Clusters

```csharp
        // Extract results for analysis
        Console.WriteLine("\n--- Cluster Profiles ---");
        var results = ExtractResults(mlContext, predictions, customers);
        
        var profiles = results
            .GroupBy(r => r.ClusterId)
            .Select(g => new
            {
                Cluster = g.Key,
                Count = g.Count(),
                Percentage = g.Count() * 100.0 / results.Count,
                AvgRecency = g.Average(r => r.Recency),
                AvgFrequency = g.Average(r => r.Frequency),
                AvgMonetary = g.Average(r => r.Monetary),
                TotalRevenue = g.Sum(r => r.Monetary)
            })
            .OrderBy(p => p.Cluster)
            .ToList();
        
        Console.WriteLine("\n{0,-10} {1,-8} {2,-10} {3,-12} {4,-12} {5,-15} {6,-12}",
            "Cluster", "Count", "% of Total", "Avg Recency", "Avg Frequency", "Avg Monetary", "Total Revenue");
        Console.WriteLine(new string('-', 90));
        
        foreach (var p in profiles)
        {
            Console.WriteLine("{0,-10} {1,-8} {2,-10:F1}% {3,-12:F1} {4,-12:F1} {5,-15:C0} {6,-12:C0}",
                p.Cluster, p.Count, p.Percentage, p.AvgRecency, p.AvgFrequency, p.AvgMonetary, p.TotalRevenue);
        }
```

### Step 5: Naming Segments and Business Recommendations

Based on RFM profiles, we assign meaningful names and develop targeted strategies:

```csharp
        // Assign segment names based on profiles
        Console.WriteLine("\n--- Customer Segments ---\n");
        
        var segmentDefinitions = new Dictionary<uint, (string Name, string Description, string Strategy)>
        {
            { 1, ("Champions", 
                  "Best customers: recent, frequent, high spend",
                  "Reward loyalty, early access to new products, referral programs") },
            { 2, ("Loyal Customers", 
                  "Consistent buyers with good lifetime value",
                  "Upsell premium products, loyalty program enrollment") },
            { 3, ("New Customers", 
                  "Recently acquired, exploring the platform",
                  "Onboarding sequences, first-purchase discounts, product education") },
            { 4, ("At Risk", 
                  "Previously active customers showing decline",
                  "Win-back campaigns, personalized offers, feedback surveys") }
        };
        
        foreach (var profile in profiles)
        {
            if (segmentDefinitions.TryGetValue(profile.Cluster, out var segment))
            {
                Console.WriteLine($"SEGMENT {profile.Cluster}: {segment.Name}");
                Console.WriteLine($"  Description: {segment.Description}");
                Console.WriteLine($"  Size: {profile.Count} customers ({profile.Percentage:F1}%)");
                Console.WriteLine($"  Revenue Contribution: {profile.TotalRevenue:C0}");
                Console.WriteLine($"  Recommended Strategy: {segment.Strategy}");
                Console.WriteLine();
            }
        }
```

### Step 6: Visualization Export

```csharp
        // Export for visualization
        Console.WriteLine("--- Exporting Visualization Data ---");
        ExportVisualizationData(results, "cluster_visualization.csv");
        ExportClusterSummary(profiles, segmentDefinitions, "cluster_summary.csv");
        
        Console.WriteLine("Exported: cluster_visualization.csv (for scatter plot)");
        Console.WriteLine("Exported: cluster_summary.csv (for reporting)");
        
        // Save model for future predictions
        mlContext.Model.Save(model, dataView.Schema, "customer_segmentation_model.zip");
        Console.WriteLine("\nModel saved: customer_segmentation_model.zip");
    }
    
    static void ExportVisualizationData(List<CustomerClusterResult> results, string path)
    {
        using var writer = new StreamWriter(path);
        writer.WriteLine("CustomerId,PC1,PC2,Cluster,Recency,Frequency,Monetary");
        
        foreach (var r in results)
        {
            writer.WriteLine($"{r.CustomerId},{r.PCAFeatures[0]:F4},{r.PCAFeatures[1]:F4}," +
                           $"{r.ClusterId},{r.Recency},{r.Frequency},{r.Monetary:F2}");
        }
    }
}
```

[FIGURE: Complete visualization dashboard showing: (1) 2D scatter plot of customer segments after PCA, with four color-coded clusters; (2) Bar chart comparing average RFM values across segments; (3) Pie chart showing customer count distribution; (4) Stacked bar showing revenue contribution by segment, with Champions contributing 45% despite being only 15% of customers.]

### Sample Output

```
Loading customer data...
Loaded 5000 customers
Recency range: 1 - 365
Frequency range: 1 - 89
Monetary range: $15 - $12,450

--- Finding Optimal K ---
k=2: Avg Distance=0.4521, Silhouette≈0.523
k=3: Avg Distance=0.3102, Silhouette≈0.587
k=4: Avg Distance=0.2234, Silhouette≈0.612  <- Elbow point
k=5: Avg Distance=0.2089, Silhouette≈0.598
k=6: Avg Distance=0.1923, Silhouette≈0.571

Selected k=4 for final model

--- Training Final Model ---
Average Distance: 0.2234
Davies-Bouldin Index: 0.8912

--- Cluster Profiles ---

Cluster    Count    % of Total Avg Recency  Avg Frequency Avg Monetary    Total Revenue
------------------------------------------------------------------------------------------
1          750      15.0%      8.2          42.3          $4,250          $3,187,500
2          1,850    37.0%      45.6         18.7          $1,420          $2,627,000
3          1,200    24.0%      15.3         3.2           $185            $222,000
4          1,200    24.0%      125.8        28.4          $2,890          $3,468,000

--- Customer Segments ---

SEGMENT 1: Champions
  Description: Best customers: recent, frequent, high spend
  Size: 750 customers (15.0%)
  Revenue Contribution: $3,187,500
  Recommended Strategy: Reward loyalty, early access to new products, referral programs

SEGMENT 2: Loyal Customers
  Description: Consistent buyers with good lifetime value
  Size: 1850 customers (37.0%)
  Revenue Contribution: $2,627,000
  Recommended Strategy: Upsell premium products, loyalty program enrollment

SEGMENT 3: New Customers
  Description: Recently acquired, exploring the platform
  Size: 1200 customers (24.0%)
  Revenue Contribution: $222,000
  Recommended Strategy: Onboarding sequences, first-purchase discounts, product education

SEGMENT 4: At Risk
  Description: Previously active customers showing decline
  Size: 1200 customers (24.0%)
  Revenue Contribution: $3,468,000
  Recommended Strategy: Win-back campaigns, personalized offers, feedback surveys
```

### Business Impact

This segmentation reveals critical insights:

1. **Champions (15%) drive disproportionate value.** They generate nearly as much revenue as At Risk customers, despite being fewer. Invest in retention.

2. **At Risk customers represent the biggest opportunity.** With $3.5M in historical revenue, re-engaging even 20% would significantly impact the bottom line.

3. **New Customers need nurturing.** Their low monetary value is expected—the goal is converting them to Loyal Customers.

4. **Loyal Customers are the backbone.** At 37% of the base, maintaining their satisfaction prevents churn to At Risk.

## Predicting Segments for New Customers

Once trained, use the model to classify new customers:

```csharp
public class CustomerSegmentPredictor
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    private readonly PredictionEngine<CustomerRFM, ClusterPrediction> _predictionEngine;
    
    public CustomerSegmentPredictor(string modelPath)
    {
        _mlContext = new MLContext();
        _model = _mlContext.Model.Load(modelPath, out var schema);
        _predictionEngine = _mlContext.Model
            .CreatePredictionEngine<CustomerRFM, ClusterPrediction>(_model);
    }
    
    public (uint Segment, string SegmentName) Predict(CustomerRFM customer)
    {
        var prediction = _predictionEngine.Predict(customer);
        string name = prediction.ClusterId switch
        {
            1 => "Champions",
            2 => "Loyal Customers",
            3 => "New Customers",
            4 => "At Risk",
            _ => "Unknown"
        };
        return (prediction.ClusterId, name);
    }
}

// Usage
var predictor = new CustomerSegmentPredictor("customer_segmentation_model.zip");
var newCustomer = new CustomerRFM 
{ 
    Recency = 7, 
    Frequency = 35, 
    Monetary = 3500 
};

var (segment, name) = predictor.Predict(newCustomer);
Console.WriteLine($"Customer assigned to segment {segment}: {name}");
// Output: Customer assigned to segment 1: Champions
```

## Beyond K-Means: Other Clustering Algorithms

K-Means is powerful but not universal. Consider alternatives for specific scenarios:

| Algorithm | Best For | Limitations |
|-----------|----------|-------------|
| **K-Means** | Spherical clusters, large datasets | Requires k, sensitive to outliers |
| **DBSCAN** | Irregular shapes, automatic k | Struggles with varying density |
| **Hierarchical** | Unknown k, dendrogram visualization | Computationally expensive for large n |
| **Gaussian Mixture** | Soft clustering, elliptical shapes | Assumes Gaussian distribution |

ML.NET focuses on K-Means, but these alternatives are available through Python interop or specialized libraries when needed.

## Key Takeaways

1. **Clustering finds structure without labels.** When you don't know what categories exist, let the algorithm discover them.

2. **K-Means is simple and effective.** Assign points to nearest centroids, move centroids to cluster means, repeat until convergence.

3. **Always scale your features.** Unscaled features produce meaningless clusters dominated by large-range variables.

4. **Choosing k requires judgment.** Use elbow method and silhouette scores as guides, but let domain knowledge and interpretability drive the final decision.

5. **Clusters must be interpretable.** Profile each cluster, give it a meaningful name, and develop actionable strategies.

6. **Visualize with PCA.** Reduce dimensions to 2D for plotting, helping you understand cluster separation and overlap.

7. **Beware high dimensions.** The curse of dimensionality makes clustering harder. Select relevant features or reduce dimensions first.

## What's Next?

Clustering revealed hidden structure in our customer data. But we projected from 3 dimensions to 2 for visualization—what exactly did that projection do? In Chapter 11, we'll explore **Dimensionality Reduction** in depth. You'll learn how PCA works mathematically, when to use other techniques like t-SNE, and how dimensionality reduction powers everything from data visualization to feature engineering for machine learning pipelines.

## Exercises

1. **Experiment with K:** Run the customer segmentation with k=3 and k=5. How do the segments change? Which segmentation is most actionable for marketing?

2. **Add Features:** Include `DaysSinceFirstPurchase` as a fourth feature. Does this improve cluster separation? Do the segment profiles change meaningfully?

3. **Handle Outliers:** Some customers have extremely high monetary values. Implement a strategy to reduce outlier impact (log transformation, capping at 99th percentile) and compare results.

4. **Silhouette Analysis:** Calculate full silhouette scores for each point and identify which customers are on cluster boundaries. These might be "swing" customers worth special attention.

5. **Stability Check:** Run clustering with 10 different random seeds. How consistent are the cluster assignments? What percentage of customers always land in the same cluster?

---

*Clustering transforms raw data into actionable segments. In our customer analysis, we went from 5,000 anonymous transaction records to four distinct customer personas, each with clear characteristics and targeted strategies. That's the power of unsupervised learning—finding the story hiding in your data, even when no one told you what story to look for.*
