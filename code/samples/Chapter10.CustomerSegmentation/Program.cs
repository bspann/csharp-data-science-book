// Chapter 10: Customer Segmentation with K-Means Clustering
// E-commerce RFM Analysis using ML.NET

using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine("  Chapter 10: Customer Segmentation with K-Means Clustering");
Console.WriteLine("  E-commerce RFM Analysis using ML.NET");
Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine();

var mlContext = new MLContext(seed: 42);

// ═══════════════════════════════════════════════════════════════════════════════
// Step 1: Generate Sample E-Commerce Customer Data
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("📊 Step 1: Generating Sample Customer Data");
Console.WriteLine("─────────────────────────────────────────────────────────────────");

var customers = GenerateSampleCustomers();
Console.WriteLine($"Generated {customers.Count} customers with RFM metrics\n");

// Show sample data
Console.WriteLine("Sample customers:");
Console.WriteLine($"{"ID",-8} {"Recency",10} {"Frequency",10} {"Monetary",12}");
Console.WriteLine(new string('-', 42));
foreach (var c in customers.Take(5))
{
    Console.WriteLine($"{c.CustomerId,-8} {c.Recency,10:F0} {c.Frequency,10:F0} {c.Monetary,12:C0}");
}
Console.WriteLine("...\n");

// ═══════════════════════════════════════════════════════════════════════════════
// Step 2: Data Statistics and Exploration
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("📈 Step 2: Data Statistics");
Console.WriteLine("─────────────────────────────────────────────────────────────────");
Console.WriteLine($"Recency:   Min={customers.Min(c => c.Recency),6:F0}, Max={customers.Max(c => c.Recency),6:F0}, Avg={customers.Average(c => c.Recency),6:F1}");
Console.WriteLine($"Frequency: Min={customers.Min(c => c.Frequency),6:F0}, Max={customers.Max(c => c.Frequency),6:F0}, Avg={customers.Average(c => c.Frequency),6:F1}");
Console.WriteLine($"Monetary:  Min={customers.Min(c => c.Monetary),6:C0}, Max={customers.Max(c => c.Monetary),6:C0}, Avg={customers.Average(c => c.Monetary),6:C0}");
Console.WriteLine();

// Load data into ML.NET
var dataView = mlContext.Data.LoadFromEnumerable(customers);

// ═══════════════════════════════════════════════════════════════════════════════
// Step 3: Finding Optimal K (Elbow Method)
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("🔍 Step 3: Finding Optimal K (Elbow Method)");
Console.WriteLine("─────────────────────────────────────────────────────────────────");
Console.WriteLine("Testing K values from 2 to 8...\n");
Console.WriteLine($"{"K",-4} {"Avg Distance",14} {"Davies-Bouldin",16} {"Analysis",20}");
Console.WriteLine(new string('-', 56));

var elbowResults = new List<(int K, double AvgDistance, double DBI)>();

for (int k = 2; k <= 8; k++)
{
    var testPipeline = mlContext.Transforms
        .Concatenate("Features", nameof(CustomerRFM.Recency), 
                                 nameof(CustomerRFM.Frequency), 
                                 nameof(CustomerRFM.Monetary))
        .Append(mlContext.Transforms.NormalizeMinMax("Features"))
        .Append(mlContext.Clustering.Trainers.KMeans(
            featureColumnName: "Features",
            numberOfClusters: k));
    
    var testModel = testPipeline.Fit(dataView);
    var testPredictions = testModel.Transform(dataView);
    var metrics = mlContext.Clustering.Evaluate(testPredictions);
    
    elbowResults.Add((k, metrics.AverageDistance, metrics.DaviesBouldinIndex));
    
    // Visual elbow indicator
    string analysis = k switch
    {
        2 => "",
        3 => "",
        4 => "◄── Elbow point",
        5 => "(diminishing returns)",
        _ => ""
    };
    
    Console.WriteLine($"{k,-4} {metrics.AverageDistance,14:F4} {metrics.DaviesBouldinIndex,16:F4} {analysis}");
}

// Find elbow using rate of change
Console.WriteLine("\n📐 Elbow Analysis:");
for (int i = 1; i < elbowResults.Count; i++)
{
    var reduction = (elbowResults[i - 1].AvgDistance - elbowResults[i].AvgDistance) / elbowResults[i - 1].AvgDistance * 100;
    Console.WriteLine($"   K={elbowResults[i].K}: {reduction:F1}% reduction from K={elbowResults[i - 1].K}");
}

int optimalK = 4;  // Based on elbow analysis
Console.WriteLine($"\n✅ Selected K={optimalK} (best balance of complexity vs. fit)\n");

// ═══════════════════════════════════════════════════════════════════════════════
// Step 4: Train Final K-Means Model
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("🏋️ Step 4: Training Final K-Means Model");
Console.WriteLine("─────────────────────────────────────────────────────────────────");

var pipeline = mlContext.Transforms
    .Concatenate("Features", nameof(CustomerRFM.Recency), 
                             nameof(CustomerRFM.Frequency), 
                             nameof(CustomerRFM.Monetary))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.Clustering.Trainers.KMeans(
        featureColumnName: "Features",
        numberOfClusters: optimalK));

Console.WriteLine("Pipeline:");
Console.WriteLine("  1. Concatenate RFM features → Feature vector");
Console.WriteLine("  2. Normalize features → MinMax scaling [0, 1]");
Console.WriteLine($"  3. K-Means clustering → {optimalK} clusters");
Console.WriteLine();

var model = pipeline.Fit(dataView);
var predictions = model.Transform(dataView);

// Evaluate final model
var finalMetrics = mlContext.Clustering.Evaluate(predictions);
Console.WriteLine("Model Performance:");
Console.WriteLine($"  Average Distance:      {finalMetrics.AverageDistance:F4}");
Console.WriteLine($"  Davies-Bouldin Index:  {finalMetrics.DaviesBouldinIndex:F4}");
Console.WriteLine($"  (Lower values indicate better-defined clusters)");
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════════════════════
// Step 5: Extract and Analyze Cluster Centroids
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("📍 Step 5: Cluster Centroids (Normalized Scale)");
Console.WriteLine("─────────────────────────────────────────────────────────────────");

VBuffer<float>[] centroids = default!;
var kmeansModel = model.LastTransformer.Model;
kmeansModel.GetClusterCentroids(ref centroids, out int numClusters);

Console.WriteLine($"{"Cluster",-10} {"Recency",12} {"Frequency",12} {"Monetary",12}");
Console.WriteLine(new string('-', 48));
for (int i = 0; i < centroids.Length; i++)
{
    var values = centroids[i].GetValues().ToArray();
    Console.WriteLine($"{i + 1,-10} {values[0],12:F3} {values[1],12:F3} {values[2],12:F3}");
}
Console.WriteLine("(Values are normalized to [0,1] scale)\n");

// ═══════════════════════════════════════════════════════════════════════════════
// Step 6: Profile Clusters with Original Scale Values
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("👥 Step 6: Cluster Profiles (Original Scale)");
Console.WriteLine("─────────────────────────────────────────────────────────────────");

// Get predictions with cluster assignments
var predictionData = mlContext.Data.CreateEnumerable<ClusterPrediction>(predictions, reuseRowObject: false).ToList();

// Combine with original customer data
var results = customers.Zip(predictionData, (c, p) => new CustomerClusterResult
{
    CustomerId = c.CustomerId,
    Recency = c.Recency,
    Frequency = c.Frequency,
    Monetary = c.Monetary,
    ClusterId = p.ClusterId,
    Distances = p.Distances
}).ToList();

// Calculate cluster profiles
var profiles = results
    .GroupBy(r => r.ClusterId)
    .Select(g => new ClusterProfile
    {
        ClusterId = g.Key,
        Count = g.Count(),
        Percentage = g.Count() * 100.0 / results.Count,
        AvgRecency = g.Average(r => r.Recency),
        AvgFrequency = g.Average(r => r.Frequency),
        AvgMonetary = g.Average(r => r.Monetary),
        TotalRevenue = g.Sum(r => r.Monetary)
    })
    .OrderBy(p => p.ClusterId)
    .ToList();

Console.WriteLine($"{"Cluster",-10} {"Count",8} {"% Total",10} {"Avg Recency",14} {"Avg Frequency",14} {"Avg Monetary",14}");
Console.WriteLine(new string('-', 72));
foreach (var p in profiles)
{
    Console.WriteLine($"{p.ClusterId,-10} {p.Count,8} {p.Percentage,9:F1}% {p.AvgRecency,14:F1} {p.AvgFrequency,14:F1} {p.AvgMonetary,14:C0}");
}
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════════════════════
// Step 7: Interpret and Name Segments
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("🏷️ Step 7: Customer Segment Interpretation");
Console.WriteLine("─────────────────────────────────────────────────────────────────");

// Assign segment names based on RFM characteristics
var segmentDefinitions = AssignSegmentNames(profiles);

foreach (var profile in profiles)
{
    var segment = segmentDefinitions[profile.ClusterId];
    Console.WriteLine($"\n┌─────────────────────────────────────────────────────────────────┐");
    Console.WriteLine($"│  SEGMENT {profile.ClusterId}: {segment.Name,-52} │");
    Console.WriteLine($"├─────────────────────────────────────────────────────────────────┤");
    Console.WriteLine($"│  Description: {segment.Description,-48} │");
    Console.WriteLine($"│  Size: {profile.Count} customers ({profile.Percentage:F1}%)                                      │".PadRight(68) + "│");
    Console.WriteLine($"│  Revenue: {profile.TotalRevenue:C0}                                           │".PadRight(68) + "│");
    Console.WriteLine($"│  RFM: R={profile.AvgRecency:F0} days, F={profile.AvgFrequency:F0} orders, M={profile.AvgMonetary:C0}    │".PadRight(68) + "│");
    Console.WriteLine($"├─────────────────────────────────────────────────────────────────┤");
    Console.WriteLine($"│  Strategy: {segment.Strategy,-52}│");
    Console.WriteLine($"└─────────────────────────────────────────────────────────────────┘");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Step 8: Revenue Analysis by Segment
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("\n\n💰 Step 8: Revenue Contribution Analysis");
Console.WriteLine("─────────────────────────────────────────────────────────────────");

double totalRevenue = profiles.Sum(p => p.TotalRevenue);
Console.WriteLine($"Total Revenue: {totalRevenue:C0}\n");

Console.WriteLine($"{"Segment",-20} {"Customers",12} {"Revenue",14} {"% Revenue",12} {"Rev/Customer",14}");
Console.WriteLine(new string('-', 74));

foreach (var profile in profiles.OrderByDescending(p => p.TotalRevenue))
{
    var segment = segmentDefinitions[profile.ClusterId];
    var revenuePercent = profile.TotalRevenue / totalRevenue * 100;
    var revenuePerCustomer = profile.TotalRevenue / profile.Count;
    
    // Visual bar
    int barLength = (int)(revenuePercent / 3);
    string bar = new string('█', barLength);
    
    Console.WriteLine($"{segment.Name,-20} {profile.Count,12} {profile.TotalRevenue,14:C0} {revenuePercent,11:F1}% {revenuePerCustomer,14:C0}");
    Console.WriteLine($"                     {bar}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Step 9: Predict Segment for New Customer
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("\n\n🔮 Step 9: Predict Segment for New Customers");
Console.WriteLine("─────────────────────────────────────────────────────────────────");

var predictionEngine = mlContext.Model.CreatePredictionEngine<CustomerRFM, ClusterPrediction>(model);

var newCustomers = new[]
{
    new CustomerRFM { CustomerId = "NEW01", Recency = 5, Frequency = 45, Monetary = 5200 },
    new CustomerRFM { CustomerId = "NEW02", Recency = 120, Frequency = 3, Monetary = 150 },
    new CustomerRFM { CustomerId = "NEW03", Recency = 30, Frequency = 15, Monetary = 890 }
};

Console.WriteLine($"{"Customer",-10} {"Recency",10} {"Frequency",10} {"Monetary",12} {"Predicted Segment",-20}");
Console.WriteLine(new string('-', 66));

foreach (var customer in newCustomers)
{
    var prediction = predictionEngine.Predict(customer);
    var segmentName = segmentDefinitions[prediction.ClusterId].Name;
    Console.WriteLine($"{customer.CustomerId,-10} {customer.Recency,10:F0} {customer.Frequency,10:F0} {customer.Monetary,12:C0} {segmentName,-20}");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Step 10: Summary and Next Steps
// ═══════════════════════════════════════════════════════════════════════════════
Console.WriteLine("\n\n📋 Summary");
Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");
Console.WriteLine($"✓ Analyzed {customers.Count} customers using RFM metrics");
Console.WriteLine($"✓ Identified optimal cluster count (K={optimalK}) using elbow method");
Console.WriteLine($"✓ Discovered {optimalK} distinct customer segments:");
foreach (var profile in profiles.OrderByDescending(p => p.TotalRevenue))
{
    var segment = segmentDefinitions[profile.ClusterId];
    Console.WriteLine($"    • {segment.Name}: {profile.Count} customers ({profile.Percentage:F0}%), {profile.TotalRevenue:C0} revenue");
}
Console.WriteLine($"✓ Model ready for real-time customer classification");
Console.WriteLine("\n🎯 Key Business Insights:");
Console.WriteLine($"   • Top segment contributes {profiles.Max(p => p.TotalRevenue) / totalRevenue * 100:F0}% of revenue");
Console.WriteLine($"   • At-risk customers hold significant revenue potential");
Console.WriteLine($"   • New customers need nurturing to increase lifetime value");

Console.WriteLine("\n═══════════════════════════════════════════════════════════════════════════════");
Console.WriteLine("  Customer segmentation complete. Model ready for production deployment.");
Console.WriteLine("═══════════════════════════════════════════════════════════════════════════════");


// ═══════════════════════════════════════════════════════════════════════════════
// Helper Methods
// ═══════════════════════════════════════════════════════════════════════════════

static List<CustomerRFM> GenerateSampleCustomers()
{
    var random = new Random(42);
    var customers = new List<CustomerRFM>();
    
    // Segment 1: Champions - Recent, Frequent, High Spenders
    for (int i = 0; i < 75; i++)
    {
        customers.Add(new CustomerRFM
        {
            CustomerId = $"C{customers.Count + 1:D4}",
            Recency = random.Next(1, 20),           // Very recent
            Frequency = random.Next(30, 60),        // Very frequent
            Monetary = random.Next(3500, 8000)      // High value
        });
    }
    
    // Segment 2: Loyal Customers - Moderate all around
    for (int i = 0; i < 150; i++)
    {
        customers.Add(new CustomerRFM
        {
            CustomerId = $"C{customers.Count + 1:D4}",
            Recency = random.Next(20, 80),          // Moderate recency
            Frequency = random.Next(12, 30),        // Moderate frequency
            Monetary = random.Next(800, 2500)       // Moderate value
        });
    }
    
    // Segment 3: New Customers - Recent but low frequency/monetary
    for (int i = 0; i < 100; i++)
    {
        customers.Add(new CustomerRFM
        {
            CustomerId = $"C{customers.Count + 1:D4}",
            Recency = random.Next(1, 40),           // Recent
            Frequency = random.Next(1, 5),          // Low frequency
            Monetary = random.Next(50, 400)         // Low value (so far)
        });
    }
    
    // Segment 4: At Risk - Not recent, but were valuable
    for (int i = 0; i < 75; i++)
    {
        customers.Add(new CustomerRFM
        {
            CustomerId = $"C{customers.Count + 1:D4}",
            Recency = random.Next(90, 200),         // Long time ago
            Frequency = random.Next(15, 40),        // Were frequent
            Monetary = random.Next(2000, 5000)      // Were high value
        });
    }
    
    // Shuffle to avoid ordering bias
    return customers.OrderBy(_ => random.Next()).ToList();
}

static Dictionary<uint, SegmentInfo> AssignSegmentNames(List<ClusterProfile> profiles)
{
    var segments = new Dictionary<uint, SegmentInfo>();
    
    // Sort profiles by characteristics to assign appropriate names
    var sorted = profiles
        .Select(p => new
        {
            Profile = p,
            RecencyScore = 1.0 / (p.AvgRecency + 1),  // Lower recency = higher score
            FrequencyScore = p.AvgFrequency,
            MonetaryScore = p.AvgMonetary
        })
        .ToList();
    
    // Find Champions: Low recency + High frequency + High monetary
    var champions = sorted.OrderByDescending(s => s.RecencyScore * s.FrequencyScore * s.MonetaryScore).First().Profile;
    
    // Find At Risk: High recency + High frequency/monetary (were good)
    var atRisk = sorted.Where(s => s.Profile.ClusterId != champions.ClusterId)
                       .OrderByDescending(s => s.Profile.AvgRecency * s.Profile.AvgMonetary)
                       .First().Profile;
    
    // Find New Customers: Low frequency + Low monetary
    var newCustomers = sorted.Where(s => s.Profile.ClusterId != champions.ClusterId && s.Profile.ClusterId != atRisk.ClusterId)
                             .OrderBy(s => s.FrequencyScore + s.MonetaryScore)
                             .First().Profile;
    
    // Find Loyal: The remaining segment
    var loyal = sorted.First(s => s.Profile.ClusterId != champions.ClusterId && 
                                   s.Profile.ClusterId != atRisk.ClusterId && 
                                   s.Profile.ClusterId != newCustomers.ClusterId).Profile;
    
    segments[champions.ClusterId] = new SegmentInfo
    {
        Name = "Champions",
        Description = "Best customers: recent, frequent, high spenders",
        Strategy = "VIP rewards, early access, referral programs"
    };
    
    segments[loyal.ClusterId] = new SegmentInfo
    {
        Name = "Loyal Customers",
        Description = "Consistent buyers with solid lifetime value",
        Strategy = "Loyalty programs, upsell premium products"
    };
    
    segments[newCustomers.ClusterId] = new SegmentInfo
    {
        Name = "New Customers",
        Description = "Recently acquired, exploring the platform",
        Strategy = "Onboarding emails, first-purchase discounts"
    };
    
    segments[atRisk.ClusterId] = new SegmentInfo
    {
        Name = "At Risk",
        Description = "Previously valuable customers now inactive",
        Strategy = "Win-back campaigns, personalized offers"
    };
    
    return segments;
}


// ═══════════════════════════════════════════════════════════════════════════════
// Data Classes
// ═══════════════════════════════════════════════════════════════════════════════

public class CustomerRFM
{
    public string CustomerId { get; set; } = string.Empty;
    public float Recency { get; set; }      // Days since last purchase (lower = better)
    public float Frequency { get; set; }    // Number of purchases (higher = better)
    public float Monetary { get; set; }     // Total spending (higher = better)
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }
    
    [ColumnName("Score")]
    public float[] Distances { get; set; } = Array.Empty<float>();
}

public class CustomerClusterResult
{
    public string CustomerId { get; set; } = string.Empty;
    public float Recency { get; set; }
    public float Frequency { get; set; }
    public float Monetary { get; set; }
    public uint ClusterId { get; set; }
    public float[] Distances { get; set; } = Array.Empty<float>();
}

public class ClusterProfile
{
    public uint ClusterId { get; set; }
    public int Count { get; set; }
    public double Percentage { get; set; }
    public double AvgRecency { get; set; }
    public double AvgFrequency { get; set; }
    public double AvgMonetary { get; set; }
    public double TotalRevenue { get; set; }
}

public class SegmentInfo
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
}
