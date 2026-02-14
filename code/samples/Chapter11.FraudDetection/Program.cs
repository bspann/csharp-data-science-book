// Chapter 11: Anomaly Detection - Fraud Detection Sample
// Demonstrates IID Spike Detection, Statistical Approaches (Z-score, IQR),
// and ML.NET Anomaly Detection APIs for processing transaction data

using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine("  Chapter 11: Anomaly Detection - Fraud Detection Demo");
Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine();

// Generate sample transaction data
var transactions = GenerateSampleTransactions();
Console.WriteLine($"Generated {transactions.Count} sample transactions");
Console.WriteLine($"  - Legitimate: {transactions.Count(t => !t.IsFraud)}");
Console.WriteLine($"  - Fraudulent: {transactions.Count(t => t.IsFraud)}");
Console.WriteLine();

// ============================================================================
// Part 1: Statistical Approaches
// ============================================================================
Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  Part 1: Statistical Anomaly Detection                       ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

// Z-Score Detection
Console.WriteLine("┌─────────────────────────────────────────────────────────────┐");
Console.WriteLine("│ Z-Score Method                                              │");
Console.WriteLine("└─────────────────────────────────────────────────────────────┘");

var zScoreDetector = new ZScoreDetector(threshold: 2.5);
var amounts = transactions.Select(t => (double)t.Amount).ToArray();
var zScoreResults = zScoreDetector.Detect(amounts);

var zScoreAnomalies = zScoreResults.Where(r => r.IsAnomaly).ToList();
Console.WriteLine($"Z-Score anomalies detected: {zScoreAnomalies.Count}");
Console.WriteLine("Top 5 anomalies by Z-score:");
foreach (var anomaly in zScoreAnomalies.OrderByDescending(a => a.Score).Take(5))
{
    var tx = transactions[anomaly.Index];
    Console.WriteLine($"  Transaction {tx.TransactionId}: ${tx.Amount:F2} (Z={anomaly.Score:F2}) - Actual Fraud: {tx.IsFraud}");
}
Console.WriteLine();

// IQR Detection
Console.WriteLine("┌─────────────────────────────────────────────────────────────┐");
Console.WriteLine("│ IQR (Interquartile Range) Method                           │");
Console.WriteLine("└─────────────────────────────────────────────────────────────┘");

var iqrDetector = new IqrDetector(multiplier: 1.5);
var iqrResults = iqrDetector.Detect(amounts);

var iqrAnomalies = iqrResults.Where(r => r.IsAnomaly).ToList();
Console.WriteLine($"IQR anomalies detected: {iqrAnomalies.Count}");
Console.WriteLine("Top 5 anomalies by IQR score:");
foreach (var anomaly in iqrAnomalies.OrderByDescending(a => a.Score).Take(5))
{
    var tx = transactions[anomaly.Index];
    Console.WriteLine($"  Transaction {tx.TransactionId}: ${tx.Amount:F2} (IQR Score={anomaly.Score:F2}) - Actual Fraud: {tx.IsFraud}");
}
Console.WriteLine();

// ============================================================================
// Part 2: ML.NET IID Spike Detection
// ============================================================================
Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  Part 2: ML.NET IID Spike Detection                          ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

var mlContext = new MLContext(seed: 42);

// Prepare data for ML.NET
var transactionAmounts = transactions
    .Select(t => new TransactionAmount { Amount = t.Amount })
    .ToList();

var dataView = mlContext.Data.LoadFromEnumerable(transactionAmounts);

// IID Spike Detection Pipeline
Console.WriteLine("┌─────────────────────────────────────────────────────────────┐");
Console.WriteLine("│ IID Spike Detection                                         │");
Console.WriteLine("└─────────────────────────────────────────────────────────────┘");

var spikePipeline = mlContext.Transforms.DetectIidSpike(
    outputColumnName: nameof(SpikePrediction.Prediction),
    inputColumnName: nameof(TransactionAmount.Amount),
    confidence: 95.0,
    pvalueHistoryLength: 30);

var spikeModel = spikePipeline.Fit(dataView);
var spikeTransformed = spikeModel.Transform(dataView);
var spikePredictions = mlContext.Data
    .CreateEnumerable<SpikePrediction>(spikeTransformed, reuseRowObject: false)
    .ToList();

var spikes = spikePredictions
    .Select((p, i) => (Index: i, Prediction: p, Transaction: transactions[i]))
    .Where(x => x.Prediction.Prediction[0] == 1)
    .ToList();

Console.WriteLine($"Spikes detected: {spikes.Count}");
Console.WriteLine("Detected spikes:");
foreach (var spike in spikes.Take(10))
{
    Console.WriteLine($"  Transaction {spike.Transaction.TransactionId}: " +
        $"${spike.Transaction.Amount:F2} (Score={spike.Prediction.Prediction[1]:F3}, " +
        $"P-Value={spike.Prediction.Prediction[2]:F4}) - Actual Fraud: {spike.Transaction.IsFraud}");
}
Console.WriteLine();

// IID Change Point Detection
Console.WriteLine("┌─────────────────────────────────────────────────────────────┐");
Console.WriteLine("│ IID Change Point Detection                                  │");
Console.WriteLine("└─────────────────────────────────────────────────────────────┘");

var changePointPipeline = mlContext.Transforms.DetectIidChangePoint(
    outputColumnName: nameof(ChangePointPrediction.Prediction),
    inputColumnName: nameof(TransactionAmount.Amount),
    confidence: 95.0,
    changeHistoryLength: 20);

var changePointModel = changePointPipeline.Fit(dataView);
var changePointTransformed = changePointModel.Transform(dataView);
var changePointPredictions = mlContext.Data
    .CreateEnumerable<ChangePointPrediction>(changePointTransformed, reuseRowObject: false)
    .ToList();

var changePoints = changePointPredictions
    .Select((p, i) => (Index: i, Prediction: p, Transaction: transactions[i]))
    .Where(x => x.Prediction.Prediction[0] == 1)
    .ToList();

Console.WriteLine($"Change points detected: {changePoints.Count}");
if (changePoints.Any())
{
    Console.WriteLine("Detected change points:");
    foreach (var cp in changePoints.Take(5))
    {
        Console.WriteLine($"  Index {cp.Index}: Transaction {cp.Transaction.TransactionId} " +
            $"(Martingale={cp.Prediction.Prediction[3]:F3})");
    }
}
Console.WriteLine();

// ============================================================================
// Part 3: Multi-Feature Statistical Fraud Detection
// ============================================================================
Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  Part 3: Multi-Feature Statistical Fraud Detection          ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

var statisticalFraudDetector = new StatisticalFraudDetector(zThreshold: 2.5);

// Train on legitimate transactions only
var legitimateTransactions = transactions.Where(t => !t.IsFraud).ToList();
statisticalFraudDetector.Train(legitimateTransactions);

// Test on all transactions
var fraudPredictions = transactions
    .Select(t => (Transaction: t, Prediction: statisticalFraudDetector.Predict(t)))
    .ToList();

var predictedFraud = fraudPredictions.Where(p => p.Prediction.IsPredictedFraud).ToList();

// Calculate metrics
int truePositives = fraudPredictions.Count(p => p.Prediction.IsPredictedFraud && p.Transaction.IsFraud);
int falsePositives = fraudPredictions.Count(p => p.Prediction.IsPredictedFraud && !p.Transaction.IsFraud);
int trueNegatives = fraudPredictions.Count(p => !p.Prediction.IsPredictedFraud && !p.Transaction.IsFraud);
int falseNegatives = fraudPredictions.Count(p => !p.Prediction.IsPredictedFraud && p.Transaction.IsFraud);

double precision = truePositives + falsePositives > 0 
    ? (double)truePositives / (truePositives + falsePositives) : 0;
double recall = truePositives + falseNegatives > 0 
    ? (double)truePositives / (truePositives + falseNegatives) : 0;
double f1Score = precision + recall > 0 
    ? 2 * (precision * recall) / (precision + recall) : 0;

Console.WriteLine("Detection Results:");
Console.WriteLine($"  True Positives:  {truePositives}");
Console.WriteLine($"  False Positives: {falsePositives}");
Console.WriteLine($"  True Negatives:  {trueNegatives}");
Console.WriteLine($"  False Negatives: {falseNegatives}");
Console.WriteLine();
Console.WriteLine($"  Precision: {precision:P2}");
Console.WriteLine($"  Recall:    {recall:P2}");
Console.WriteLine($"  F1 Score:  {f1Score:F3}");
Console.WriteLine();

// Show some example predictions with risk factors
Console.WriteLine("Example fraud predictions with risk factors:");
foreach (var pred in predictedFraud.OrderByDescending(p => p.Prediction.Score).Take(5))
{
    Console.WriteLine($"\n  Transaction {pred.Transaction.TransactionId}:");
    Console.WriteLine($"    Amount: ${pred.Transaction.Amount:F2}");
    Console.WriteLine($"    Score: {pred.Prediction.Score:F3}");
    Console.WriteLine($"    Actual Fraud: {pred.Transaction.IsFraud}");
    Console.WriteLine($"    Risk Factors:");
    foreach (var factor in pred.Prediction.RiskFactors)
    {
        Console.WriteLine($"      - {factor}");
    }
}
Console.WriteLine();

// ============================================================================
// Part 4: Streaming Anomaly Detection Simulation
// ============================================================================
Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  Part 4: Streaming Anomaly Detection                         ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

var streamingDetector = new StreamingAnomalyDetector(windowSize: 30, zThreshold: 2.5);

// Initialize with historical data
var historicalData = amounts.Take(50).ToArray();
streamingDetector.Initialize(historicalData);

Console.WriteLine("Processing streaming transactions...");
Console.WriteLine();

var alerts = new List<AnomalyAlert>();
var simulatedStream = amounts.Skip(50).ToList();

for (int i = 0; i < simulatedStream.Count; i++)
{
    var timestamp = DateTime.Now.AddMinutes(-simulatedStream.Count + i);
    var alert = streamingDetector.ProcessDataPoint(simulatedStream[i], timestamp);
    
    if (alert != null)
    {
        alerts.Add(alert);
    }
}

Console.WriteLine($"Streaming alerts generated: {alerts.Count}");
Console.WriteLine("Recent alerts:");
foreach (var alert in alerts.TakeLast(5))
{
    Console.WriteLine($"  [{alert.Timestamp:HH:mm:ss}] Value: ${alert.Value:F2}, " +
        $"Score: {alert.Score:F3}, Severity: {alert.Severity}");
}
Console.WriteLine();

// ============================================================================
// Part 5: Threshold Analysis
// ============================================================================
Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  Part 5: Threshold Analysis                                  ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

var thresholdEvaluator = new ThresholdEvaluator();

// Get probability scores for all transactions using statistical detector
var probabilityPredictions = transactions
    .Select(t => (Actual: t.IsFraud, Probability: (float)statisticalFraudDetector.Predict(t).Score))
    .ToList();

var thresholds = new[] { 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 };
var thresholdMetrics = thresholdEvaluator.EvaluateThresholds(probabilityPredictions, thresholds);

thresholdEvaluator.PrintEvaluationReport(thresholdMetrics);
Console.WriteLine();

// ============================================================================
// Part 6: Hybrid Detection Demo
// ============================================================================
Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║  Part 6: Hybrid Detection (Statistical + Pattern-Based)     ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

var hybridDetector = new HybridFraudDetector(
    new StatisticalFraudDetector(zThreshold: 2.5),
    new PatternBasedDetector());

// Train the hybrid detector
hybridDetector.Train(legitimateTransactions);

// Evaluate on test transactions
var hybridResults = transactions.Select(t => (
    Transaction: t,
    Decision: hybridDetector.Evaluate(t)
)).ToList();

// Calculate hybrid metrics
int hybridTP = hybridResults.Count(r => r.Decision.IsFraud && r.Transaction.IsFraud);
int hybridFP = hybridResults.Count(r => r.Decision.IsFraud && !r.Transaction.IsFraud);
int hybridFN = hybridResults.Count(r => !r.Decision.IsFraud && r.Transaction.IsFraud);

double hybridPrecision = hybridTP + hybridFP > 0 
    ? (double)hybridTP / (hybridTP + hybridFP) : 0;
double hybridRecall = hybridTP + hybridFN > 0 
    ? (double)hybridTP / (hybridTP + hybridFN) : 0;
double hybridF1 = hybridPrecision + hybridRecall > 0 
    ? 2 * (hybridPrecision * hybridRecall) / (hybridPrecision + hybridRecall) : 0;

Console.WriteLine("Hybrid Detection Results:");
Console.WriteLine($"  Precision: {hybridPrecision:P2}");
Console.WriteLine($"  Recall:    {hybridRecall:P2}");
Console.WriteLine($"  F1 Score:  {hybridF1:F3}");
Console.WriteLine();

// Show action distribution
var actionCounts = hybridResults
    .GroupBy(r => r.Decision.Action)
    .ToDictionary(g => g.Key, g => g.Count());

Console.WriteLine("Action Distribution:");
foreach (var action in actionCounts.OrderBy(a => a.Key))
{
    Console.WriteLine($"  {action.Key}: {action.Value} transactions");
}
Console.WriteLine();

Console.WriteLine("Sample decisions:");
foreach (var result in hybridResults
    .Where(r => r.Decision.Action != FraudAction.Approve)
    .OrderByDescending(r => r.Decision.Confidence)
    .Take(5))
{
    Console.WriteLine($"\n  Transaction {result.Transaction.TransactionId}:");
    Console.WriteLine($"    Amount: ${result.Transaction.Amount:F2}");
    Console.WriteLine($"    Action: {result.Decision.Action}");
    Console.WriteLine($"    Confidence: {result.Decision.Confidence:P1}");
    Console.WriteLine($"    Method: {result.Decision.DetectionMethod}");
    Console.WriteLine($"    Actual Fraud: {result.Transaction.IsFraud}");
}

Console.WriteLine();
Console.WriteLine("═══════════════════════════════════════════════════════════════");
Console.WriteLine("  Demo Complete!");
Console.WriteLine("═══════════════════════════════════════════════════════════════");

// ============================================================================
// Data Generation
// ============================================================================
static List<Transaction> GenerateSampleTransactions()
{
    var random = new Random(42);
    var transactions = new List<Transaction>();
    var baseTime = DateTime.Now.AddHours(-24);

    // Generate legitimate transactions
    for (int i = 0; i < 180; i++)
    {
        var hourOffset = random.NextDouble() * 24;
        var hour = (float)(baseTime.AddHours(hourOffset).Hour + 
            baseTime.AddHours(hourOffset).Minute / 60.0);
        
        transactions.Add(new Transaction
        {
            TransactionId = $"TX{i:D4}",
            Timestamp = baseTime.AddHours(hourOffset),
            Amount = (float)(random.NextDouble() * 200 + 20), // $20-$220
            TimeSinceLastTransaction = (float)(random.NextDouble() * 12 + 0.5),
            MerchantCategory = random.Next(1, 10),
            DistanceFromHome = (float)(random.NextDouble() * 50),
            DistanceFromLastTransaction = (float)(random.NextDouble() * 30),
            UsedChip = random.NextDouble() > 0.2,
            OnlineTransaction = random.NextDouble() > 0.7,
            TransactionHour = hour,
            IsFraud = false
        });
    }

    // Generate fraudulent transactions with suspicious patterns
    var fraudPatterns = new (string Pattern, Func<Random, Transaction> Generator)[]
    {
        // High amount fraud
        ("HighAmount", r => new Transaction
        {
            TransactionId = $"TX{180 + r.Next(1000):D4}",
            Timestamp = baseTime.AddHours(r.NextDouble() * 24),
            Amount = (float)(r.NextDouble() * 5000 + 2000), // $2000-$7000
            TimeSinceLastTransaction = (float)(r.NextDouble() * 0.5),
            MerchantCategory = r.Next(1, 10),
            DistanceFromHome = (float)(r.NextDouble() * 100 + 200),
            DistanceFromLastTransaction = (float)(r.NextDouble() * 500 + 100),
            UsedChip = false,
            OnlineTransaction = false,
            TransactionHour = (float)(r.NextDouble() * 5), // Late night
            IsFraud = true
        }),
        
        // High velocity fraud (rapid transactions far apart)
        ("HighVelocity", r => new Transaction
        {
            TransactionId = $"TX{280 + r.Next(1000):D4}",
            Timestamp = baseTime.AddHours(r.NextDouble() * 24),
            Amount = (float)(r.NextDouble() * 300 + 100),
            TimeSinceLastTransaction = (float)(r.NextDouble() * 0.1 + 0.01), // Minutes
            MerchantCategory = r.Next(1, 10),
            DistanceFromHome = (float)(r.NextDouble() * 500 + 300),
            DistanceFromLastTransaction = (float)(r.NextDouble() * 1000 + 500), // Impossible travel
            UsedChip = false,
            OnlineTransaction = false,
            TransactionHour = (float)(r.NextDouble() * 24),
            IsFraud = true
        }),
        
        // Late night + far from home
        ("LateNightFarAway", r => new Transaction
        {
            TransactionId = $"TX{380 + r.Next(1000):D4}",
            Timestamp = baseTime.AddHours(r.NextDouble() * 5), // 0-5 AM
            Amount = (float)(r.NextDouble() * 800 + 200),
            TimeSinceLastTransaction = (float)(r.NextDouble() * 2),
            MerchantCategory = r.Next(1, 10),
            DistanceFromHome = (float)(r.NextDouble() * 800 + 400),
            DistanceFromLastTransaction = (float)(r.NextDouble() * 200),
            UsedChip = false,
            OnlineTransaction = false,
            TransactionHour = (float)(r.NextDouble() * 5),
            IsFraud = true
        })
    };

    // Add 20 fraudulent transactions
    for (int i = 0; i < 20; i++)
    {
        var pattern = fraudPatterns[i % fraudPatterns.Length];
        transactions.Add(pattern.Generator(random));
    }

    // Shuffle transactions
    return transactions.OrderBy(_ => random.Next()).ToList();
}

// ============================================================================
// Data Classes
// ============================================================================
public class Transaction
{
    public string TransactionId { get; set; } = "";
    public DateTime Timestamp { get; set; }
    public float Amount { get; set; }
    public float TimeSinceLastTransaction { get; set; }
    public int MerchantCategory { get; set; }
    public float DistanceFromHome { get; set; }
    public float DistanceFromLastTransaction { get; set; }
    public bool UsedChip { get; set; }
    public bool OnlineTransaction { get; set; }
    public float TransactionHour { get; set; }
    public bool IsFraud { get; set; }

    public float VelocityScore => DistanceFromLastTransaction / 
        Math.Max(TimeSinceLastTransaction, 0.1f);
}

public class TransactionAmount
{
    public float Amount { get; set; }
}

public class SpikePrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; } = [];
}

public class ChangePointPrediction
{
    [VectorType(4)]
    public double[] Prediction { get; set; } = [];
}

public record AnomalyResult
{
    public int Index { get; init; }
    public double Value { get; init; }
    public double Score { get; init; }
    public bool IsAnomaly { get; init; }
}

public record FraudPrediction
{
    public string TransactionId { get; init; } = "";
    public double Score { get; init; }
    public bool IsPredictedFraud { get; init; }
    public List<string> RiskFactors { get; init; } = [];
}

public record AnomalyAlert
{
    public DateTime Timestamp { get; init; }
    public double Value { get; init; }
    public double Score { get; init; }
    public AlertSeverity Severity { get; init; }
}

public enum AlertSeverity { Low, Medium, High, Critical }

public record ThresholdMetrics
{
    public double Threshold { get; init; }
    public double Precision { get; init; }
    public double Recall { get; init; }
    public double F1Score { get; init; }
    public int FalsePositives { get; init; }
    public int FalseNegatives { get; init; }
    public double EstimatedCost { get; init; }
}

public record FraudDecision
{
    public string TransactionId { get; init; } = "";
    public bool IsFraud { get; init; }
    public float Confidence { get; init; }
    public FraudAction Action { get; init; }
    public List<string> RiskFactors { get; init; } = [];
    public string DetectionMethod { get; init; } = "";
}

public enum FraudAction { Approve, Block, Review }

// ============================================================================
// Statistical Detectors
// ============================================================================
public class ZScoreDetector
{
    private readonly double _threshold;

    public ZScoreDetector(double threshold = 3.0)
    {
        _threshold = threshold;
    }

    public List<AnomalyResult> Detect(double[] data)
    {
        var results = new List<AnomalyResult>();

        double mean = data.Average();
        double stdDev = CalculateStandardDeviation(data, mean);

        if (stdDev == 0)
            return results;

        for (int i = 0; i < data.Length; i++)
        {
            double zScore = Math.Abs((data[i] - mean) / stdDev);

            results.Add(new AnomalyResult
            {
                Index = i,
                Value = data[i],
                Score = zScore,
                IsAnomaly = zScore > _threshold
            });
        }

        return results;
    }

    private double CalculateStandardDeviation(double[] data, double mean)
    {
        double sumSquaredDifferences = data.Sum(x => Math.Pow(x - mean, 2));
        return Math.Sqrt(sumSquaredDifferences / data.Length);
    }
}

public class IqrDetector
{
    private readonly double _multiplier;

    public IqrDetector(double multiplier = 1.5)
    {
        _multiplier = multiplier;
    }

    public List<AnomalyResult> Detect(double[] data)
    {
        var sorted = data.OrderBy(x => x).ToArray();

        double q1 = GetPercentile(sorted, 25);
        double q3 = GetPercentile(sorted, 75);
        double iqr = q3 - q1;

        double lowerBound = q1 - (_multiplier * iqr);
        double upperBound = q3 + (_multiplier * iqr);

        var results = new List<AnomalyResult>();

        for (int i = 0; i < data.Length; i++)
        {
            bool isAnomaly = data[i] < lowerBound || data[i] > upperBound;
            double score = isAnomaly
                ? Math.Max(Math.Abs(data[i] - lowerBound),
                          Math.Abs(data[i] - upperBound)) / Math.Max(iqr, 1)
                : 0;

            results.Add(new AnomalyResult
            {
                Index = i,
                Value = data[i],
                Score = score,
                IsAnomaly = isAnomaly
            });
        }

        return results;
    }

    private double GetPercentile(double[] sortedData, double percentile)
    {
        double index = (percentile / 100.0) * (sortedData.Length - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);

        if (lower == upper)
            return sortedData[lower];

        return sortedData[lower] + (index - lower) *
               (sortedData[upper] - sortedData[lower]);
    }
}

// ============================================================================
// Multi-Feature Statistical Fraud Detector
// ============================================================================
public class StatisticalFraudDetector
{
    private double _meanAmount;
    private double _stdAmount;
    private double _meanVelocity;
    private double _stdVelocity;
    private double _meanDistance;
    private double _stdDistance;
    private readonly double _zThreshold;

    public StatisticalFraudDetector(double zThreshold = 3.0)
    {
        _zThreshold = zThreshold;
    }

    public void Train(IEnumerable<Transaction> legitimateTransactions)
    {
        var transactions = legitimateTransactions.ToList();

        _meanAmount = transactions.Average(t => t.Amount);
        _stdAmount = Math.Sqrt(transactions
            .Average(t => Math.Pow(t.Amount - _meanAmount, 2)));

        var velocities = transactions
            .Select(t => (double)t.VelocityScore)
            .Where(v => !double.IsInfinity(v) && !double.IsNaN(v))
            .ToList();

        _meanVelocity = velocities.Average();
        _stdVelocity = Math.Sqrt(velocities
            .Average(v => Math.Pow(v - _meanVelocity, 2)));

        _meanDistance = transactions.Average(t => t.DistanceFromHome);
        _stdDistance = Math.Sqrt(transactions
            .Average(t => Math.Pow(t.DistanceFromHome - _meanDistance, 2)));
    }

    public FraudPrediction Predict(Transaction transaction)
    {
        double amountZScore = _stdAmount > 0
            ? Math.Abs((transaction.Amount - _meanAmount) / _stdAmount)
            : 0;

        double velocityZScore = _stdVelocity > 0
            ? Math.Abs((transaction.VelocityScore - _meanVelocity) / _stdVelocity)
            : 0;

        double distanceZScore = _stdDistance > 0
            ? Math.Abs((transaction.DistanceFromHome - _meanDistance) / _stdDistance)
            : 0;

        // Weighted combined score
        double combinedScore = (amountZScore * 0.4 + velocityZScore * 0.35 + distanceZScore * 0.25);

        var riskFactors = new List<string>();
        
        if (amountZScore > _zThreshold)
            riskFactors.Add($"Unusual amount (z={amountZScore:F2})");
        if (velocityZScore > _zThreshold)
            riskFactors.Add($"Suspicious velocity (z={velocityZScore:F2})");
        if (distanceZScore > _zThreshold * 0.8)
            riskFactors.Add($"Far from typical location (z={distanceZScore:F2})");
        if (transaction.DistanceFromHome > 500)
            riskFactors.Add("Far from home location");
        if (!transaction.UsedChip && !transaction.OnlineTransaction)
            riskFactors.Add("Magnetic stripe transaction");
        if (transaction.TransactionHour >= 0 && transaction.TransactionHour < 6)
            riskFactors.Add("Late night transaction");
        if (transaction.TimeSinceLastTransaction < 0.1)
            riskFactors.Add("Rapid successive transaction");

        return new FraudPrediction
        {
            TransactionId = transaction.TransactionId,
            Score = combinedScore,
            IsPredictedFraud = combinedScore > _zThreshold || riskFactors.Count >= 3,
            RiskFactors = riskFactors
        };
    }
}

// ============================================================================
// Streaming Anomaly Detector (using rolling window Z-score)
// ============================================================================
public class StreamingAnomalyDetector
{
    private readonly Queue<double> _windowData;
    private readonly int _windowSize;
    private readonly double _zThreshold;
    private bool _initialized;

    public StreamingAnomalyDetector(int windowSize = 50, double zThreshold = 3.0)
    {
        _windowData = new Queue<double>();
        _windowSize = windowSize;
        _zThreshold = zThreshold;
    }

    public void Initialize(double[] historicalData)
    {
        _windowData.Clear();
        foreach (var value in historicalData.TakeLast(_windowSize))
        {
            _windowData.Enqueue(value);
        }
        _initialized = true;
    }

    public AnomalyAlert? ProcessDataPoint(double value, DateTime timestamp)
    {
        if (!_initialized)
            throw new InvalidOperationException(
                "Initialize must be called before processing data points");

        // Calculate Z-score based on current window
        var windowArray = _windowData.ToArray();
        double mean = windowArray.Average();
        double stdDev = Math.Sqrt(windowArray.Average(x => Math.Pow(x - mean, 2)));
        
        double zScore = stdDev > 0 ? Math.Abs((value - mean) / stdDev) : 0;

        // Add new value and maintain window size
        _windowData.Enqueue(value);
        if (_windowData.Count > _windowSize)
            _windowData.Dequeue();

        if (zScore > _zThreshold)
        {
            return new AnomalyAlert
            {
                Timestamp = timestamp,
                Value = value,
                Score = zScore,
                Severity = ClassifySeverity(zScore)
            };
        }

        return null;
    }

    private AlertSeverity ClassifySeverity(double score)
    {
        return score switch
        {
            > 5.0 => AlertSeverity.Critical,
            > 4.0 => AlertSeverity.High,
            > 3.0 => AlertSeverity.Medium,
            _ => AlertSeverity.Low
        };
    }
}

// ============================================================================
// Threshold Evaluator
// ============================================================================
public class ThresholdEvaluator
{
    public List<ThresholdMetrics> EvaluateThresholds(
        List<(bool Actual, float Probability)> predictions,
        double[] thresholds)
    {
        var results = new List<ThresholdMetrics>();

        foreach (var threshold in thresholds)
        {
            int truePositives = 0;
            int falsePositives = 0;
            int trueNegatives = 0;
            int falseNegatives = 0;

            foreach (var (actual, probability) in predictions)
            {
                bool predicted = probability >= threshold;

                if (predicted && actual) truePositives++;
                else if (predicted && !actual) falsePositives++;
                else if (!predicted && !actual) trueNegatives++;
                else falseNegatives++;
            }

            double precision = truePositives + falsePositives > 0
                ? (double)truePositives / (truePositives + falsePositives)
                : 0;

            double recall = truePositives + falseNegatives > 0
                ? (double)truePositives / (truePositives + falseNegatives)
                : 0;

            double f1 = precision + recall > 0
                ? 2 * (precision * recall) / (precision + recall)
                : 0;

            double cost = (falseNegatives * 100) + (falsePositives * 1);

            results.Add(new ThresholdMetrics
            {
                Threshold = threshold,
                Precision = precision,
                Recall = recall,
                F1Score = f1,
                FalsePositives = falsePositives,
                FalseNegatives = falseNegatives,
                EstimatedCost = cost
            });
        }

        return results;
    }

    public void PrintEvaluationReport(List<ThresholdMetrics> metrics)
    {
        Console.WriteLine("Threshold Analysis:");
        Console.WriteLine(new string('-', 85));
        Console.WriteLine($"{"Threshold",10} {"Precision",11} {"Recall",10} " +
                         $"{"F1",8} {"FP",6} {"FN",6} {"Cost",12}");
        Console.WriteLine(new string('-', 85));

        foreach (var m in metrics)
        {
            Console.WriteLine($"{m.Threshold,10:F2} {m.Precision,11:P1} " +
                            $"{m.Recall,10:P1} {m.F1Score,8:F3} " +
                            $"{m.FalsePositives,6} {m.FalseNegatives,6} " +
                            $"{m.EstimatedCost,12:C0}");
        }

        var optimal = metrics.MinBy(m => m.EstimatedCost);
        Console.WriteLine();
        Console.WriteLine($"Optimal threshold (lowest cost): {optimal?.Threshold:F2}");
    }
}

// ============================================================================
// Pattern-Based Detector (for Hybrid approach)
// ============================================================================
public class PatternBasedDetector
{
    public (bool IsSuspicious, float Confidence, string Reason) Analyze(Transaction transaction)
    {
        var suspiciousPatterns = new List<(bool Match, float Weight, string Reason)>
        {
            (transaction.Amount > 1000 && transaction.TransactionHour < 6,
             0.8f, "High amount during unusual hours"),
            
            (transaction.VelocityScore > 500,
             0.9f, "Impossible travel velocity"),
            
            (!transaction.UsedChip && transaction.Amount > 500,
             0.6f, "High-value magnetic stripe transaction"),
            
            (transaction.DistanceFromHome > 300 && transaction.TimeSinceLastTransaction < 1,
             0.85f, "Rapid distant transaction"),
            
            (transaction.DistanceFromLastTransaction > 500 && transaction.TimeSinceLastTransaction < 2,
             0.95f, "Physically impossible transaction sequence")
        };

        var matches = suspiciousPatterns.Where(p => p.Match).ToList();
        
        if (matches.Count == 0)
            return (false, 0, "No suspicious patterns");

        var maxMatch = matches.MaxBy(m => m.Weight);
        float combinedConfidence = Math.Min(1.0f, matches.Sum(m => m.Weight) / matches.Count);
        
        return (true, combinedConfidence, maxMatch.Reason);
    }
}

// ============================================================================
// Hybrid Fraud Detector
// ============================================================================
public class HybridFraudDetector
{
    private readonly StatisticalFraudDetector _statisticalDetector;
    private readonly PatternBasedDetector _patternDetector;

    public HybridFraudDetector(
        StatisticalFraudDetector statisticalDetector,
        PatternBasedDetector patternDetector)
    {
        _statisticalDetector = statisticalDetector;
        _patternDetector = patternDetector;
    }

    public void Train(IEnumerable<Transaction> legitimateTransactions)
    {
        _statisticalDetector.Train(legitimateTransactions);
    }

    public FraudDecision Evaluate(Transaction transaction)
    {
        var statisticalResult = _statisticalDetector.Predict(transaction);
        var (patternSuspicious, patternConfidence, patternReason) = 
            _patternDetector.Analyze(transaction);

        // High-confidence pattern match - block immediately
        if (patternSuspicious && patternConfidence > 0.9)
        {
            return new FraudDecision
            {
                TransactionId = transaction.TransactionId,
                IsFraud = true,
                Confidence = patternConfidence,
                Action = FraudAction.Block,
                RiskFactors = [patternReason],
                DetectionMethod = "Pattern (high confidence)"
            };
        }

        // High statistical score - block
        if (statisticalResult.Score > 4.0)
        {
            return new FraudDecision
            {
                TransactionId = transaction.TransactionId,
                IsFraud = true,
                Confidence = 0.95f,
                Action = FraudAction.Block,
                RiskFactors = statisticalResult.RiskFactors,
                DetectionMethod = "Statistical (high confidence)"
            };
        }

        // Both methods flag as suspicious - block
        if (statisticalResult.IsPredictedFraud && patternSuspicious)
        {
            var allFactors = statisticalResult.RiskFactors.ToList();
            allFactors.Add(patternReason);
            
            return new FraudDecision
            {
                TransactionId = transaction.TransactionId,
                IsFraud = true,
                Confidence = Math.Max((float)statisticalResult.Score / 4, patternConfidence),
                Action = FraudAction.Block,
                RiskFactors = allFactors,
                DetectionMethod = "Hybrid (both methods agree)"
            };
        }

        // Moderate suspicion - review
        if (statisticalResult.Score > 2.0 || (patternSuspicious && patternConfidence > 0.5))
        {
            var factors = statisticalResult.RiskFactors.ToList();
            if (patternSuspicious) factors.Add(patternReason);
            
            return new FraudDecision
            {
                TransactionId = transaction.TransactionId,
                IsFraud = false,
                Confidence = (float)statisticalResult.Score / 4,
                Action = FraudAction.Review,
                RiskFactors = factors,
                DetectionMethod = "Hybrid (moderate suspicion)"
            };
        }

        // Low risk - approve
        return new FraudDecision
        {
            TransactionId = transaction.TransactionId,
            IsFraud = false,
            Confidence = 0.1f,
            Action = FraudAction.Approve,
            RiskFactors = [],
            DetectionMethod = "Statistical (low risk)"
        };
    }
}
