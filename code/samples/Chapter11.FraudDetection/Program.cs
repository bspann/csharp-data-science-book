using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

// Chapter 11: Fraud Detection with Anomaly Detection
// Demonstrates Z-score, IQR, and ML.NET spike detection

Console.WriteLine("=== Chapter 11: Fraud Detection ===\n");

// Embedded sample transaction data
var transactions = new List<Transaction>
{
    new(1, "2024-01-15 09:23", 45.99m, "Grocery", "NYC"),
    new(2, "2024-01-15 10:15", 12.50m, "Coffee", "NYC"),
    new(3, "2024-01-15 14:30", 89.99m, "Restaurant", "NYC"),
    new(4, "2024-01-15 23:45", 2500.00m, "Electronics", "Miami"),  // Suspicious: high amount, late, different city
    new(5, "2024-01-16 08:00", 35.00m, "Gas", "NYC"),
    new(6, "2024-01-16 08:05", 38.00m, "Gas", "LA"),               // Suspicious: impossible travel
    new(7, "2024-01-16 12:00", 55.00m, "Grocery", "NYC"),
    new(8, "2024-01-16 15:30", 4200.00m, "Wire Transfer", "NYC"),  // Suspicious: very high amount
    new(9, "2024-01-17 09:00", 22.00m, "Coffee", "NYC"),
    new(10, "2024-01-17 11:30", 67.50m, "Restaurant", "NYC"),
    new(11, "2024-01-17 14:00", 120.00m, "Clothing", "NYC"),
    new(12, "2024-01-17 14:01", 115.00m, "Clothing", "NYC"),       // Suspicious: rapid succession
    new(13, "2024-01-17 14:02", 130.00m, "Clothing", "NYC"),       // Suspicious: rapid succession
    new(14, "2024-01-18 10:00", 42.00m, "Grocery", "NYC"),
    new(15, "2024-01-18 16:00", 8500.00m, "Jewelry", "Vegas"),     // Suspicious: extremely high
};

// === Method 1: Z-Score Anomaly Detection ===
Console.WriteLine("--- Z-Score Anomaly Detection ---");
var amounts = transactions.Select(t => (double)t.Amount).ToArray();
var mean = amounts.Average();
var stdDev = Math.Sqrt(amounts.Average(x => Math.Pow(x - mean, 2)));

Console.WriteLine($"Mean: ${mean:F2}, StdDev: ${stdDev:F2}\n");

foreach (var t in transactions)
{
    var zScore = (double)(t.Amount - (decimal)mean) / stdDev;
    if (Math.Abs(zScore) > 2.0)
    {
        Console.WriteLine($"⚠️  TX#{t.Id}: ${t.Amount:F2} (Z-score: {zScore:F2}) - ANOMALY");
    }
}

// === Method 2: IQR-Based Outlier Detection ===
Console.WriteLine("\n--- IQR Outlier Detection ---");
var sorted = amounts.OrderBy(x => x).ToArray();
var q1 = Percentile(sorted, 25);
var q3 = Percentile(sorted, 75);
var iqr = q3 - q1;
var lowerBound = q1 - 1.5 * iqr;
var upperBound = q3 + 1.5 * iqr;

Console.WriteLine($"Q1: ${q1:F2}, Q3: ${q3:F2}, IQR: ${iqr:F2}");
Console.WriteLine($"Bounds: ${lowerBound:F2} - ${upperBound:F2}\n");

foreach (var t in transactions)
{
    if ((double)t.Amount < lowerBound || (double)t.Amount > upperBound)
    {
        Console.WriteLine($"⚠️  TX#{t.Id}: ${t.Amount:F2} - OUTLIER (outside IQR bounds)");
    }
}

// === Method 3: ML.NET IID Spike Detection ===
Console.WriteLine("\n--- ML.NET Spike Detection ---");
var mlContext = new MLContext(seed: 42);

var spikeData = transactions.Select(t => new SpikeInput { Value = (float)t.Amount }).ToList();
var dataView = mlContext.Data.LoadFromEnumerable(spikeData);

var pipeline = mlContext.Transforms.DetectIidSpike(
    outputColumnName: nameof(SpikePrediction.Prediction),
    inputColumnName: nameof(SpikeInput.Value),
    confidence: 95.0,
    pvalueHistoryLength: 5);

var model = pipeline.Fit(dataView);
var transformedData = model.Transform(dataView);
var predictions = mlContext.Data.CreateEnumerable<SpikePrediction>(transformedData, reuseRowObject: false).ToList();

for (int i = 0; i < transactions.Count; i++)
{
    var pred = predictions[i].Prediction;
    // Prediction: [Alert (0/1), Raw Score, P-Value]
    if (pred[0] == 1)
    {
        Console.WriteLine($"⚠️  TX#{transactions[i].Id}: ${transactions[i].Amount:F2} - SPIKE DETECTED (p-value: {pred[2]:F4})");
    }
}

// === Combined Risk Scoring ===
Console.WriteLine("\n--- Combined Risk Assessment ---");
Console.WriteLine("ID   | Amount      | Z-Score | IQR    | Spike | Risk Score | Status");
Console.WriteLine(new string('-', 75));

foreach (var t in transactions)
{
    var idx = transactions.IndexOf(t);
    var zScore = (double)(t.Amount - (decimal)mean) / stdDev;
    var isIqrOutlier = (double)t.Amount < lowerBound || (double)t.Amount > upperBound;
    var isSpike = predictions[idx].Prediction[0] == 1;
    
    // Calculate composite risk score (0-100)
    var riskScore = 0;
    if (Math.Abs(zScore) > 3) riskScore += 40;
    else if (Math.Abs(zScore) > 2) riskScore += 25;
    else if (Math.Abs(zScore) > 1.5) riskScore += 10;
    
    if (isIqrOutlier) riskScore += 30;
    if (isSpike) riskScore += 30;
    
    // Additional heuristics
    var hour = int.Parse(t.Timestamp.Split(' ')[1].Split(':')[0]);
    if (hour >= 23 || hour <= 5) riskScore += 15; // Late night
    if (t.Amount > 1000) riskScore += 10;
    
    var status = riskScore switch
    {
        >= 70 => "🚨 HIGH RISK",
        >= 40 => "⚠️  MEDIUM",
        >= 20 => "📋 LOW",
        _ => "✅ NORMAL"
    };
    
    Console.WriteLine($"{t.Id,4} | ${t.Amount,9:F2} | {zScore,7:F2} | {(isIqrOutlier ? "YES" : "NO"),-6} | {(isSpike ? "YES" : "NO"),-5} | {riskScore,10} | {status}");
}

// Summary
var flagged = transactions.Count(t =>
{
    var idx = transactions.IndexOf(t);
    var zScore = Math.Abs((double)(t.Amount - (decimal)mean) / stdDev);
    var isIqr = (double)t.Amount < lowerBound || (double)t.Amount > upperBound;
    var isSpike = predictions[idx].Prediction[0] == 1;
    return zScore > 2 || isIqr || isSpike;
});

Console.WriteLine($"\n📊 Summary: {flagged} of {transactions.Count} transactions flagged for review");

// Helper function for percentile calculation
static double Percentile(double[] sorted, int percentile)
{
    var index = (percentile / 100.0) * (sorted.Length - 1);
    var lower = (int)Math.Floor(index);
    var upper = (int)Math.Ceiling(index);
    if (lower == upper) return sorted[lower];
    return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
}

// Data classes
record Transaction(int Id, string Timestamp, decimal Amount, string Category, string Location);

class SpikeInput
{
    public float Value { get; set; }
}

class SpikePrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; } = [];
}
