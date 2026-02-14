# Chapter 11: Anomaly Detection

The email arrived at 3:47 AM on a Tuesday: "ALERT: Unusual transaction pattern detected." Within seconds, the fraud detection system had flagged a suspicious sequence—someone in São Paulo was attempting to use a credit card that had been swiped in Chicago just two hours earlier. The transaction was blocked automatically, and the cardholder received a notification asking to verify the activity. By morning, what could have been a $15,000 loss was prevented entirely.

This is anomaly detection in action. It's the silent guardian running behind every credit card transaction, the watchful eye monitoring server health metrics, and the invisible shield protecting networks from intrusion. For C# developers, understanding anomaly detection opens doors to building systems that can identify the unusual, the suspicious, and the potentially catastrophic—before they cause real damage.

## What Is Anomaly Detection?

Anomaly detection is the practice of identifying data points, events, or observations that deviate significantly from expected behavior. These outliers might represent fraud, system failures, security breaches, or simply interesting phenomena worth investigating.

### The Three Faces of Anomalies

Anomalies come in different forms, and understanding their nature helps us choose appropriate detection strategies:

**Point Anomalies** are individual data points that lie far outside the normal range. A single credit card transaction of $50,000 when the customer typically spends $200 is a point anomaly. These are the most straightforward to detect—a value that simply doesn't fit.

**Contextual Anomalies** depend on the surrounding context. A temperature of 90°F in Phoenix during summer is perfectly normal; the same temperature in January is anomalous. Similarly, a server CPU spike at 95% during a scheduled backup window might be expected, while the same spike at 3 AM on a Sunday warrants investigation.

**Collective Anomalies** are patterns where individual points seem normal, but together they reveal something suspicious. Consider a user who logs in at 9 AM, checks email, and logs out at 5 PM—completely normal behavior. But if that same pattern repeats for 30 consecutive days without a single deviation, including weekends and holidays, you might be looking at an automated script rather than a human user.

### Real-World Applications

Anomaly detection permeates nearly every industry:

**Financial Services**: Fraud detection remains the poster child for anomaly detection. Banks analyze transaction amounts, locations, merchant categories, and timing to flag suspicious activity. A card used at a gas station followed immediately by a jewelry store purchase in a different city raises red flags.

**Cybersecurity**: Network intrusion detection systems monitor traffic patterns for signs of attacks. Unusual data exfiltration volumes, connections to known malicious IPs, or login attempts from unexpected geographies all trigger alerts.

**Industrial IoT**: Manufacturing plants deploy sensors that monitor equipment health. Subtle changes in vibration patterns, temperature gradients, or power consumption can predict equipment failure days before catastrophic breakdowns occur.

**Healthcare**: Patient monitoring systems track vital signs continuously, alerting medical staff when heart rate, blood pressure, or oxygen saturation deviate from safe ranges.

**E-commerce**: Recommendation engines need to filter out anomalous user behavior—bot activity, price scrapers, or coordinated review fraud—to maintain accurate personalization models.

## Statistical Approaches to Anomaly Detection

Before diving into machine learning, it's worth mastering statistical methods. They're interpretable, computationally efficient, and often sufficient for many use cases.

### Z-Score Method

The Z-score measures how many standard deviations a data point lies from the mean. For normally distributed data, about 99.7% of observations fall within three standard deviations of the mean—making anything beyond that threshold a potential anomaly.

```csharp
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
        
        // Avoid division by zero for constant data
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

public record AnomalyResult
{
    public int Index { get; init; }
    public double Value { get; init; }
    public double Score { get; init; }
    public bool IsAnomaly { get; init; }
}
```

The Z-score method works beautifully for normally distributed data but struggles with skewed distributions or data containing multiple modes.

### Interquartile Range (IQR) Method

The IQR method is more robust to outliers than Z-scores because it relies on quartiles rather than the mean and standard deviation, which outliers can heavily influence.

```csharp
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
                          Math.Abs(data[i] - upperBound)) / iqr
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
```

A multiplier of 1.5 identifies mild outliers; using 3.0 catches only extreme outliers. The IQR method handles skewed data gracefully but assumes anomalies appear in the tails of the distribution.

### When Statistical Methods Fall Short

Statistical approaches assume anomalies are rare and differ significantly from normal observations. They struggle when:

- Normal behavior spans multiple clusters or modes
- The relationship between features is complex and nonlinear
- Anomalies are subtle combinations of normal-looking individual values
- The data distribution changes over time (concept drift)

This is where machine learning techniques shine.

## ML.NET Anomaly Detection APIs

ML.NET provides powerful, production-ready anomaly detection capabilities specifically designed for time series data. These algorithms are optimized for scenarios like server monitoring, financial analysis, and IoT sensor data.

### IID Spike Detection

IID (Independent and Identically Distributed) Spike Detection identifies sudden, temporary spikes in data. Think of a website's traffic spiking when a viral tweet mentions your product, or a power grid experiencing a momentary surge.

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

public class ServerMetric
{
    public float CpuUsage { get; set; }
}

public class SpikePrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; } = Array.Empty<double>();
}

public class SpikeDetector
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    
    public SpikeDetector(int confidence = 95, int pValueHistoryLength = 30)
    {
        _mlContext = new MLContext();
        
        // We'll build the pipeline when we have data
        _confidence = confidence;
        _pValueHistoryLength = pValueHistoryLength;
    }
    
    private readonly int _confidence;
    private readonly int _pValueHistoryLength;
    
    public List<(int Index, float Value, bool IsSpike)> DetectSpikes(
        float[] cpuReadings)
    {
        var data = cpuReadings
            .Select(x => new ServerMetric { CpuUsage = x })
            .ToList();
        
        var dataView = _mlContext.Data.LoadFromEnumerable(data);
        
        var pipeline = _mlContext.Transforms
            .DetectIidSpike(
                outputColumnName: nameof(SpikePrediction.Prediction),
                inputColumnName: nameof(ServerMetric.CpuUsage),
                confidence: _confidence,
                pvalueHistoryLength: _pValueHistoryLength);
        
        var model = pipeline.Fit(dataView);
        var transformedData = model.Transform(dataView);
        
        var predictions = _mlContext.Data
            .CreateEnumerable<SpikePrediction>(transformedData, reuseRowObject: false)
            .ToList();
        
        var results = new List<(int, float, bool)>();
        
        for (int i = 0; i < predictions.Count; i++)
        {
            // Prediction array: [Alert (0/1), Score, P-Value]
            bool isSpike = predictions[i].Prediction[0] == 1;
            results.Add((i, cpuReadings[i], isSpike));
        }
        
        return results;
    }
}
```

The `confidence` parameter (0-100) controls sensitivity—higher values mean fewer false positives but potentially missed anomalies. The `pvalueHistoryLength` determines how much historical context the algorithm considers.

### IID Change Point Detection

While spikes are temporary deviations, change points represent permanent shifts in the data's underlying distribution. A company implementing a new caching strategy might see average response times drop from 200ms to 50ms—a change point rather than a spike.

```csharp
public class ChangePointPrediction
{
    [VectorType(4)]
    public double[] Prediction { get; set; } = Array.Empty<double>();
}

public class ChangePointDetector
{
    private readonly MLContext _mlContext;
    
    public ChangePointDetector()
    {
        _mlContext = new MLContext();
    }
    
    public List<ChangePointResult> DetectChangePoints(
        float[] values,
        int confidence = 95,
        int changeHistoryLength = 20)
    {
        var data = values
            .Select(x => new ServerMetric { CpuUsage = x })
            .ToList();
        
        var dataView = _mlContext.Data.LoadFromEnumerable(data);
        
        var pipeline = _mlContext.Transforms
            .DetectIidChangePoint(
                outputColumnName: nameof(ChangePointPrediction.Prediction),
                inputColumnName: nameof(ServerMetric.CpuUsage),
                confidence: confidence,
                changeHistoryLength: changeHistoryLength);
        
        var model = pipeline.Fit(dataView);
        var transformedData = model.Transform(dataView);
        
        var predictions = _mlContext.Data
            .CreateEnumerable<ChangePointPrediction>(
                transformedData, reuseRowObject: false)
            .ToList();
        
        var results = new List<ChangePointResult>();
        
        for (int i = 0; i < predictions.Count; i++)
        {
            // Prediction array: [Alert, Score, P-Value, Martingale Value]
            var pred = predictions[i].Prediction;
            
            results.Add(new ChangePointResult
            {
                Index = i,
                Value = values[i],
                IsChangePoint = pred[0] == 1,
                Score = pred[1],
                PValue = pred[2],
                MartingaleValue = pred[3]
            });
        }
        
        return results;
    }
}

public record ChangePointResult
{
    public int Index { get; init; }
    public float Value { get; init; }
    public bool IsChangePoint { get; init; }
    public double Score { get; init; }
    public double PValue { get; init; }
    public double MartingaleValue { get; init; }
}
```

### SSA (Singular Spectrum Analysis) for Seasonal Data

Real-world time series often exhibit seasonality—web traffic peaks during business hours, retail sales surge during holidays, and energy consumption follows weekly patterns. SSA decomposes time series into trend, seasonal, and noise components, enabling anomaly detection that respects these patterns.

```csharp
public class SsaAnomalyDetector
{
    private readonly MLContext _mlContext;
    private readonly int _trainingWindowSize;
    private readonly int _seasonalityWindowSize;
    
    public SsaAnomalyDetector(
        int trainingWindowSize = 168,  // 7 days of hourly data
        int seasonalityWindowSize = 24) // Daily seasonality
    {
        _mlContext = new MLContext();
        _trainingWindowSize = trainingWindowSize;
        _seasonalityWindowSize = seasonalityWindowSize;
    }
    
    public List<SsaAnomalyResult> DetectAnomalies(float[] values)
    {
        var data = values
            .Select(x => new ServerMetric { CpuUsage = x })
            .ToList();
        
        var dataView = _mlContext.Data.LoadFromEnumerable(data);
        
        var pipeline = _mlContext.Transforms
            .DetectAnomalyBySrCnn(
                outputColumnName: "Prediction",
                inputColumnName: nameof(ServerMetric.CpuUsage),
                windowSize: 64,
                backAddWindowSize: 5,
                lookaheadWindowSize: 5,
                averagingWindowSize: 3,
                judgementWindowSize: 21,
                threshold: 0.3);
        
        var model = pipeline.Fit(dataView);
        var transformedData = model.Transform(dataView);
        
        var predictions = _mlContext.Data
            .CreateEnumerable<SsaPrediction>(
                transformedData, reuseRowObject: false)
            .ToList();
        
        var results = new List<SsaAnomalyResult>();
        
        for (int i = 0; i < predictions.Count; i++)
        {
            var pred = predictions[i].Prediction;
            
            results.Add(new SsaAnomalyResult
            {
                Index = i,
                Value = values[i],
                IsAnomaly = pred[0] == 1,
                RawScore = pred[1],
                Magnitude = pred.Length > 2 ? pred[2] : 0
            });
        }
        
        return results;
    }
}

public class SsaPrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; } = Array.Empty<double>();
}

public record SsaAnomalyResult
{
    public int Index { get; init; }
    public float Value { get; init; }
    public bool IsAnomaly { get; init; }
    public double RawScore { get; init; }
    public double Magnitude { get; init; }
}
```

The SR-CNN (Spectral Residual with Convolutional Neural Network) algorithm is particularly effective at detecting anomalies while ignoring expected seasonal variations.

## Time Series Anomaly Detection in Practice

Time series data presents unique challenges: values are correlated with their neighbors, patterns repeat at various frequencies, and the definition of "normal" can evolve over time.

### Streaming Detection with ML.NET

For real-time systems, you need streaming anomaly detection—processing data points as they arrive rather than analyzing batch datasets:

```csharp
public class StreamingAnomalyDetector : IDisposable
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private TimeSeriesPredictionEngine<ServerMetric, SpikePrediction>? _engine;
    
    public StreamingAnomalyDetector()
    {
        _mlContext = new MLContext();
    }
    
    public void Initialize(float[] historicalData)
    {
        var data = historicalData
            .Select(x => new ServerMetric { CpuUsage = x })
            .ToList();
        
        var dataView = _mlContext.Data.LoadFromEnumerable(data);
        
        var pipeline = _mlContext.Transforms
            .DetectIidSpike(
                outputColumnName: nameof(SpikePrediction.Prediction),
                inputColumnName: nameof(ServerMetric.CpuUsage),
                confidence: 95,
                pvalueHistoryLength: 50);
        
        _model = pipeline.Fit(dataView);
        
        // Create streaming prediction engine
        _engine = _mlContext.Model
            .CreateTimeSeriesPredictionEngine<ServerMetric, SpikePrediction>(
                _model);
    }
    
    public AnomalyAlert? ProcessDataPoint(float value, DateTime timestamp)
    {
        if (_engine == null)
            throw new InvalidOperationException(
                "Initialize must be called before processing data points");
        
        var prediction = _engine.Predict(new ServerMetric { CpuUsage = value });
        
        if (prediction.Prediction[0] == 1)
        {
            return new AnomalyAlert
            {
                Timestamp = timestamp,
                Value = value,
                Score = prediction.Prediction[1],
                PValue = prediction.Prediction[2],
                Severity = ClassifySeverity(prediction.Prediction[1])
            };
        }
        
        return null;
    }
    
    private AlertSeverity ClassifySeverity(double score)
    {
        return score switch
        {
            > 0.95 => AlertSeverity.Critical,
            > 0.8 => AlertSeverity.High,
            > 0.6 => AlertSeverity.Medium,
            _ => AlertSeverity.Low
        };
    }
    
    public void SaveCheckpoint(string path)
    {
        _engine?.CheckPoint(_mlContext, path);
    }
    
    public void LoadCheckpoint(string path)
    {
        _engine = _mlContext.Model
            .CreateTimeSeriesPredictionEngine<ServerMetric, SpikePrediction>(
                _mlContext.Model.Load(path, out _));
    }
    
    public void Dispose()
    {
        (_engine as IDisposable)?.Dispose();
    }
}

public record AnomalyAlert
{
    public DateTime Timestamp { get; init; }
    public float Value { get; init; }
    public double Score { get; init; }
    public double PValue { get; init; }
    public AlertSeverity Severity { get; init; }
}

public enum AlertSeverity { Low, Medium, High, Critical }
```

The `TimeSeriesPredictionEngine` maintains state between predictions, allowing it to learn from each new data point while detecting anomalies. The checkpoint functionality enables system restarts without losing learned context.

## Setting Thresholds and Handling False Positives

The most challenging aspect of anomaly detection isn't the algorithm—it's calibration. Set thresholds too low, and you'll drown in false alarms. Set them too high, and real anomalies slip through unnoticed.

### The Cost Asymmetry Problem

In fraud detection, the costs of errors are wildly asymmetric:

- **False Positive**: A legitimate transaction is blocked. The customer is inconvenienced, maybe calls support, possibly churns. Cost: $10-50 in support time, potential lifetime value loss.
- **False Negative**: A fraudulent transaction is approved. The bank absorbs the loss. Cost: Potentially thousands of dollars plus investigation costs.

This asymmetry means we typically prefer more false positives (higher sensitivity) in high-stakes scenarios.

### Dynamic Thresholding

Static thresholds fail when data patterns evolve. Implement adaptive thresholds that adjust based on recent data:

```csharp
public class AdaptiveThresholdManager
{
    private readonly Queue<double> _recentScores;
    private readonly int _windowSize;
    private readonly double _baseThreshold;
    private readonly double _adaptationRate;
    
    public AdaptiveThresholdManager(
        int windowSize = 1000,
        double baseThreshold = 0.8,
        double adaptationRate = 0.1)
    {
        _recentScores = new Queue<double>();
        _windowSize = windowSize;
        _baseThreshold = baseThreshold;
        _adaptationRate = adaptationRate;
    }
    
    public double CurrentThreshold { get; private set; }
    
    public bool IsAnomaly(double score, out double dynamicThreshold)
    {
        _recentScores.Enqueue(score);
        
        if (_recentScores.Count > _windowSize)
            _recentScores.Dequeue();
        
        // Calculate threshold as percentile of recent scores
        var sorted = _recentScores.OrderBy(x => x).ToArray();
        int percentileIndex = (int)(sorted.Length * _baseThreshold);
        double adaptedThreshold = sorted[Math.Min(percentileIndex, sorted.Length - 1)];
        
        // Smooth the transition
        CurrentThreshold = CurrentThreshold == 0 
            ? adaptedThreshold
            : (1 - _adaptationRate) * CurrentThreshold + 
              _adaptationRate * adaptedThreshold;
        
        dynamicThreshold = CurrentThreshold;
        return score > CurrentThreshold;
    }
}
```

### Alert Fatigue and Aggregation

Operations teams suffer from alert fatigue when anomaly systems generate too many notifications. Combat this with intelligent aggregation:

```csharp
public class AlertAggregator
{
    private readonly Dictionary<string, AlertState> _alertStates = new();
    private readonly TimeSpan _cooldownPeriod;
    private readonly int _requiredConsecutiveAnomalies;
    
    public AlertAggregator(
        TimeSpan cooldownPeriod,
        int requiredConsecutiveAnomalies = 3)
    {
        _cooldownPeriod = cooldownPeriod;
        _requiredConsecutiveAnomalies = requiredConsecutiveAnomalies;
    }
    
    public bool ShouldAlert(string metricName, AnomalyAlert alert)
    {
        if (!_alertStates.TryGetValue(metricName, out var state))
        {
            state = new AlertState();
            _alertStates[metricName] = state;
        }
        
        // Check cooldown
        if (state.LastAlertTime != null && 
            DateTime.UtcNow - state.LastAlertTime < _cooldownPeriod)
        {
            state.ConsecutiveAnomalies = 0;
            return false;
        }
        
        state.ConsecutiveAnomalies++;
        
        // Require multiple consecutive anomalies before alerting
        if (state.ConsecutiveAnomalies >= _requiredConsecutiveAnomalies)
        {
            state.LastAlertTime = DateTime.UtcNow;
            state.ConsecutiveAnomalies = 0;
            return true;
        }
        
        return false;
    }
    
    public void RecordNormal(string metricName)
    {
        if (_alertStates.TryGetValue(metricName, out var state))
        {
            state.ConsecutiveAnomalies = 0;
        }
    }
    
    private class AlertState
    {
        public int ConsecutiveAnomalies { get; set; }
        public DateTime? LastAlertTime { get; set; }
    }
}
```

## Project: Detecting Fraudulent Transactions

Let's build a complete fraud detection system that demonstrates both statistical and ML-based approaches, with proper evaluation and production considerations.

### Understanding the Dataset

We'll work with a credit card transaction dataset containing features like transaction amount, time since last transaction, merchant category, and whether the transaction was fraudulent:

```csharp
public class Transaction
{
    public string TransactionId { get; set; } = "";
    public DateTime Timestamp { get; set; }
    public float Amount { get; set; }
    public float TimeSinceLastTransaction { get; set; } // Hours
    public int MerchantCategory { get; set; }
    public float DistanceFromHome { get; set; } // Miles
    public float DistanceFromLastTransaction { get; set; }
    public bool UsedChip { get; set; }
    public bool OnlineTransaction { get; set; }
    public float TransactionHour { get; set; }
    public bool IsFraud { get; set; }
    
    // Derived features
    public float VelocityScore => DistanceFromLastTransaction / 
        Math.Max(TimeSinceLastTransaction, 0.1f);
}
```

### Statistical Fraud Detection

First, implement a multi-feature statistical detector:

```csharp
public class StatisticalFraudDetector
{
    private double _meanAmount;
    private double _stdAmount;
    private double _meanVelocity;
    private double _stdVelocity;
    private readonly double _zThreshold;
    
    public StatisticalFraudDetector(double zThreshold = 3.0)
    {
        _zThreshold = zThreshold;
    }
    
    public void Train(IEnumerable<Transaction> legitimateTransactions)
    {
        var transactions = legitimateTransactions.ToList();
        
        // Calculate statistics for amount
        _meanAmount = transactions.Average(t => t.Amount);
        _stdAmount = Math.Sqrt(transactions
            .Average(t => Math.Pow(t.Amount - _meanAmount, 2)));
        
        // Calculate statistics for velocity
        var velocities = transactions
            .Select(t => (double)t.VelocityScore)
            .Where(v => !double.IsInfinity(v) && !double.IsNaN(v))
            .ToList();
        
        _meanVelocity = velocities.Average();
        _stdVelocity = Math.Sqrt(velocities
            .Average(v => Math.Pow(v - _meanVelocity, 2)));
    }
    
    public FraudPrediction Predict(Transaction transaction)
    {
        double amountZScore = _stdAmount > 0 
            ? Math.Abs((transaction.Amount - _meanAmount) / _stdAmount)
            : 0;
        
        double velocityZScore = _stdVelocity > 0 
            ? Math.Abs((transaction.VelocityScore - _meanVelocity) / _stdVelocity)
            : 0;
        
        // Combined score (simple average of normalized scores)
        double combinedScore = (amountZScore + velocityZScore) / 2;
        
        // Risk factors
        var riskFactors = new List<string>();
        if (amountZScore > _zThreshold)
            riskFactors.Add($"Unusual amount (z={amountZScore:F2})");
        if (velocityZScore > _zThreshold)
            riskFactors.Add($"Suspicious velocity (z={velocityZScore:F2})");
        if (transaction.DistanceFromHome > 500)
            riskFactors.Add("Far from home location");
        if (!transaction.UsedChip && !transaction.OnlineTransaction)
            riskFactors.Add("Magnetic stripe transaction");
        if (transaction.TransactionHour >= 0 && transaction.TransactionHour < 6)
            riskFactors.Add("Late night transaction");
        
        return new FraudPrediction
        {
            TransactionId = transaction.TransactionId,
            Score = combinedScore / _zThreshold, // Normalize to 0-1 range
            IsPredictedFraud = combinedScore > _zThreshold,
            RiskFactors = riskFactors
        };
    }
}

public record FraudPrediction
{
    public string TransactionId { get; init; } = "";
    public double Score { get; init; }
    public bool IsPredictedFraud { get; init; }
    public List<string> RiskFactors { get; init; } = new();
}
```

### ML.NET Binary Classification Approach

For more sophisticated detection, train a binary classifier:

```csharp
public class MLFraudDetector
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private PredictionEngine<TransactionFeatures, FraudOutput>? _predictionEngine;
    
    public MLFraudDetector()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public void Train(IEnumerable<Transaction> transactions)
    {
        var features = transactions.Select(t => new TransactionFeatures
        {
            Amount = t.Amount,
            TimeSinceLastTransaction = t.TimeSinceLastTransaction,
            MerchantCategory = t.MerchantCategory,
            DistanceFromHome = t.DistanceFromHome,
            DistanceFromLastTransaction = t.DistanceFromLastTransaction,
            UsedChip = t.UsedChip ? 1 : 0,
            OnlineTransaction = t.OnlineTransaction ? 1 : 0,
            TransactionHour = t.TransactionHour,
            VelocityScore = t.VelocityScore,
            IsFraud = t.IsFraud
        }).ToList();
        
        var dataView = _mlContext.Data.LoadFromEnumerable(features);
        var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        
        var pipeline = _mlContext.Transforms
            .Concatenate("Features",
                nameof(TransactionFeatures.Amount),
                nameof(TransactionFeatures.TimeSinceLastTransaction),
                nameof(TransactionFeatures.MerchantCategory),
                nameof(TransactionFeatures.DistanceFromHome),
                nameof(TransactionFeatures.DistanceFromLastTransaction),
                nameof(TransactionFeatures.UsedChip),
                nameof(TransactionFeatures.OnlineTransaction),
                nameof(TransactionFeatures.TransactionHour),
                nameof(TransactionFeatures.VelocityScore))
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: nameof(TransactionFeatures.IsFraud),
                featureColumnName: "Features",
                numberOfLeaves: 20,
                minimumExampleCountPerLeaf: 10,
                learningRate: 0.1));
        
        _model = pipeline.Fit(split.TrainSet);
        
        // Evaluate
        var predictions = _model.Transform(split.TestSet);
        var metrics = _mlContext.BinaryClassification.Evaluate(
            predictions, 
            labelColumnName: nameof(TransactionFeatures.IsFraud));
        
        Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");
        
        _predictionEngine = _mlContext.Model
            .CreatePredictionEngine<TransactionFeatures, FraudOutput>(_model);
    }
    
    public (bool IsFraud, float Probability) Predict(Transaction transaction)
    {
        if (_predictionEngine == null)
            throw new InvalidOperationException("Model not trained");
        
        var features = new TransactionFeatures
        {
            Amount = transaction.Amount,
            TimeSinceLastTransaction = transaction.TimeSinceLastTransaction,
            MerchantCategory = transaction.MerchantCategory,
            DistanceFromHome = transaction.DistanceFromHome,
            DistanceFromLastTransaction = transaction.DistanceFromLastTransaction,
            UsedChip = transaction.UsedChip ? 1 : 0,
            OnlineTransaction = transaction.OnlineTransaction ? 1 : 0,
            TransactionHour = transaction.TransactionHour,
            VelocityScore = transaction.VelocityScore
        };
        
        var result = _predictionEngine.Predict(features);
        return (result.PredictedLabel, result.Probability);
    }
}

public class TransactionFeatures
{
    public float Amount { get; set; }
    public float TimeSinceLastTransaction { get; set; }
    public int MerchantCategory { get; set; }
    public float DistanceFromHome { get; set; }
    public float DistanceFromLastTransaction { get; set; }
    public int UsedChip { get; set; }
    public int OnlineTransaction { get; set; }
    public float TransactionHour { get; set; }
    public float VelocityScore { get; set; }
    public bool IsFraud { get; set; }
}

public class FraudOutput
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }
    public float Probability { get; set; }
}
```

### Evaluating at Different Thresholds

The default 0.5 threshold isn't always optimal. Evaluate precision and recall across thresholds:

```csharp
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
            
            // Calculate cost (example: FN costs 100, FP costs 1)
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
        Console.WriteLine("\nThreshold Analysis:");
        Console.WriteLine(new string('-', 80));
        Console.WriteLine($"{"Threshold",10} {"Precision",10} {"Recall",10} " +
                         $"{"F1",10} {"FP",8} {"FN",8} {"Cost",12}");
        Console.WriteLine(new string('-', 80));
        
        foreach (var m in metrics)
        {
            Console.WriteLine($"{m.Threshold,10:F2} {m.Precision,10:P1} " +
                            $"{m.Recall,10:P1} {m.F1Score,10:F3} " +
                            $"{m.FalsePositives,8} {m.FalseNegatives,8} " +
                            $"{m.EstimatedCost,12:C0}");
        }
        
        var optimal = metrics.MinBy(m => m.EstimatedCost);
        Console.WriteLine($"\nOptimal threshold (lowest cost): {optimal?.Threshold:F2}");
    }
}

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
```

### Production Deployment Architecture

A production fraud detection system needs more than just an algorithm:

```csharp
public class ProductionFraudDetectionService
{
    private readonly MLFraudDetector _mlDetector;
    private readonly StatisticalFraudDetector _statisticalDetector;
    private readonly AlertAggregator _alertAggregator;
    private readonly ILogger<ProductionFraudDetectionService> _logger;
    
    // Configuration
    private readonly double _mlThreshold;
    private readonly bool _useEnsemble;
    
    public ProductionFraudDetectionService(
        ILogger<ProductionFraudDetectionService> logger,
        IConfiguration config)
    {
        _mlDetector = new MLFraudDetector();
        _statisticalDetector = new StatisticalFraudDetector();
        _alertAggregator = new AlertAggregator(
            TimeSpan.FromMinutes(15),
            requiredConsecutiveAnomalies: 1);
        _logger = logger;
        
        _mlThreshold = config.GetValue<double>("FraudDetection:Threshold", 0.7);
        _useEnsemble = config.GetValue<bool>("FraudDetection:UseEnsemble", true);
    }
    
    public async Task<FraudDecision> EvaluateTransaction(Transaction transaction)
    {
        var stopwatch = Stopwatch.StartNew();
        
        try
        {
            // Get predictions from both models
            var (mlFraud, mlProbability) = _mlDetector.Predict(transaction);
            var statisticalResult = _statisticalDetector.Predict(transaction);
            
            // Ensemble decision
            FraudDecision decision;
            
            if (_useEnsemble)
            {
                // Both models must agree for low-confidence predictions
                bool ensembleFraud = mlProbability > _mlThreshold && 
                                    statisticalResult.IsPredictedFraud;
                
                // High-confidence ML prediction overrides
                if (mlProbability > 0.95)
                    ensembleFraud = true;
                
                decision = new FraudDecision
                {
                    TransactionId = transaction.TransactionId,
                    IsFraud = ensembleFraud,
                    Confidence = mlProbability,
                    Action = ensembleFraud ? FraudAction.Block : FraudAction.Approve,
                    RiskFactors = statisticalResult.RiskFactors,
                    ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                };
            }
            else
            {
                decision = new FraudDecision
                {
                    TransactionId = transaction.TransactionId,
                    IsFraud = mlProbability > _mlThreshold,
                    Confidence = mlProbability,
                    Action = mlProbability > _mlThreshold 
                        ? FraudAction.Block 
                        : FraudAction.Approve,
                    RiskFactors = statisticalResult.RiskFactors,
                    ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                };
            }
            
            // Add review queue for borderline cases
            if (mlProbability > 0.4 && mlProbability < _mlThreshold)
            {
                decision = decision with { Action = FraudAction.Review };
            }
            
            // Log for monitoring
            _logger.LogInformation(
                "Transaction {TransactionId}: Fraud={IsFraud}, " +
                "Confidence={Confidence:P1}, Action={Action}, " +
                "ProcessingTime={ProcessingTimeMs}ms",
                decision.TransactionId,
                decision.IsFraud,
                decision.Confidence,
                decision.Action,
                decision.ProcessingTimeMs);
            
            return decision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, 
                "Error evaluating transaction {TransactionId}", 
                transaction.TransactionId);
            
            // Fail-safe: approve but flag for review
            return new FraudDecision
            {
                TransactionId = transaction.TransactionId,
                IsFraud = false,
                Confidence = 0,
                Action = FraudAction.Review,
                RiskFactors = new List<string> { "Evaluation error - manual review required" },
                ProcessingTimeMs = stopwatch.ElapsedMilliseconds
            };
        }
    }
}

public record FraudDecision
{
    public string TransactionId { get; init; } = "";
    public bool IsFraud { get; init; }
    public float Confidence { get; init; }
    public FraudAction Action { get; init; }
    public List<string> RiskFactors { get; init; } = new();
    public long ProcessingTimeMs { get; init; }
}

public enum FraudAction { Approve, Block, Review }
```

### Key Production Considerations

**Latency Requirements**: Credit card authorizations must complete in under 100ms. The model must be fast, which sometimes means trading accuracy for speed.

**Model Monitoring**: Track prediction distributions over time. If the model suddenly flags 20% of transactions instead of the usual 2%, something is wrong—either fraud is spiking or the model is drifting.

**Feedback Loops**: When fraud investigators confirm or dispute predictions, feed that data back into retraining. But be careful—confirmed fraud is certain, but "not fraud" just means "not caught yet."

**Explainability**: When a customer calls asking why their transaction was blocked, you need to provide reasons. The statistical approach's risk factors help here, as does feature importance from the ML model.

**A/B Testing**: Roll out model changes gradually, comparing performance against the production model before full deployment.

### Model Monitoring and Continuous Learning

Production fraud detection systems require continuous monitoring to remain effective. Fraudsters constantly adapt their tactics, and models that performed brilliantly at launch can degrade within months.

```csharp
public class ModelMonitor
{
    private readonly Queue<PredictionRecord> _recentPredictions;
    private readonly int _windowSize;
    private double _baselineFraudRate;
    
    public ModelMonitor(int windowSize = 10000, double baselineFraudRate = 0.02)
    {
        _recentPredictions = new Queue<PredictionRecord>();
        _windowSize = windowSize;
        _baselineFraudRate = baselineFraudRate;
    }
    
    public void RecordPrediction(string transactionId, float probability, 
        bool predicted, bool? actualFraud = null)
    {
        _recentPredictions.Enqueue(new PredictionRecord
        {
            TransactionId = transactionId,
            Probability = probability,
            PredictedFraud = predicted,
            ActualFraud = actualFraud,
            Timestamp = DateTime.UtcNow
        });
        
        while (_recentPredictions.Count > _windowSize)
            _recentPredictions.Dequeue();
    }
    
    public ModelHealthReport GetHealthReport()
    {
        var predictions = _recentPredictions.ToList();
        
        // Calculate current fraud rate (predictions)
        double currentPredictedRate = predictions.Count > 0
            ? predictions.Count(p => p.PredictedFraud) / (double)predictions.Count
            : 0;
        
        // Check for rate anomalies
        double rateDeviation = Math.Abs(currentPredictedRate - _baselineFraudRate) 
            / _baselineFraudRate;
        
        // Calculate confirmed accuracy where we have feedback
        var confirmed = predictions.Where(p => p.ActualFraud.HasValue).ToList();
        double? accuracy = confirmed.Count > 100
            ? confirmed.Count(p => p.PredictedFraud == p.ActualFraud) 
                / (double)confirmed.Count
            : null;
        
        // Check score distribution health
        var scores = predictions.Select(p => (double)p.Probability).ToList();
        double meanScore = scores.Average();
        double stdScore = Math.Sqrt(scores.Average(s => Math.Pow(s - meanScore, 2)));
        
        return new ModelHealthReport
        {
            Timestamp = DateTime.UtcNow,
            SampleSize = predictions.Count,
            CurrentPredictedFraudRate = currentPredictedRate,
            BaselineFraudRate = _baselineFraudRate,
            RateDeviationPercent = rateDeviation * 100,
            ConfirmedAccuracy = accuracy,
            MeanPredictionScore = meanScore,
            ScoreStandardDeviation = stdScore,
            RequiresAttention = rateDeviation > 0.5 || (accuracy.HasValue && accuracy < 0.9),
            AlertMessage = GenerateAlertMessage(rateDeviation, accuracy)
        };
    }
    
    private string? GenerateAlertMessage(double rateDeviation, double? accuracy)
    {
        var alerts = new List<string>();
        
        if (rateDeviation > 1.0)
            alerts.Add("CRITICAL: Fraud prediction rate has doubled from baseline");
        else if (rateDeviation > 0.5)
            alerts.Add("WARNING: Fraud prediction rate deviating significantly");
        
        if (accuracy.HasValue && accuracy < 0.8)
            alerts.Add("CRITICAL: Model accuracy has dropped below 80%");
        else if (accuracy.HasValue && accuracy < 0.9)
            alerts.Add("WARNING: Model accuracy declining");
        
        return alerts.Count > 0 ? string.Join("; ", alerts) : null;
    }
    
    private record PredictionRecord
    {
        public string TransactionId { get; init; } = "";
        public float Probability { get; init; }
        public bool PredictedFraud { get; init; }
        public bool? ActualFraud { get; init; }
        public DateTime Timestamp { get; init; }
    }
}

public record ModelHealthReport
{
    public DateTime Timestamp { get; init; }
    public int SampleSize { get; init; }
    public double CurrentPredictedFraudRate { get; init; }
    public double BaselineFraudRate { get; init; }
    public double RateDeviationPercent { get; init; }
    public double? ConfirmedAccuracy { get; init; }
    public double MeanPredictionScore { get; init; }
    public double ScoreStandardDeviation { get; init; }
    public bool RequiresAttention { get; init; }
    public string? AlertMessage { get; init; }
}
```

Implement automated retraining triggers based on performance degradation:

```csharp
public class RetrainingOrchestrator
{
    private readonly ModelMonitor _monitor;
    private readonly MLFraudDetector _detector;
    private readonly ILogger<RetrainingOrchestrator> _logger;
    private DateTime _lastRetraining;
    private readonly TimeSpan _minimumRetrainingInterval = TimeSpan.FromDays(7);
    
    public RetrainingOrchestrator(
        ModelMonitor monitor,
        MLFraudDetector detector,
        ILogger<RetrainingOrchestrator> logger)
    {
        _monitor = monitor;
        _detector = detector;
        _logger = logger;
        _lastRetraining = DateTime.UtcNow;
    }
    
    public async Task<bool> CheckAndRetrain(
        Func<Task<IEnumerable<Transaction>>> getTrainingData)
    {
        var health = _monitor.GetHealthReport();
        
        // Check if retraining is needed
        bool shouldRetrain = health.RequiresAttention &&
            DateTime.UtcNow - _lastRetraining > _minimumRetrainingInterval;
        
        if (!shouldRetrain)
            return false;
        
        _logger.LogWarning(
            "Initiating model retraining due to: {AlertMessage}",
            health.AlertMessage);
        
        try
        {
            var trainingData = await getTrainingData();
            _detector.Train(trainingData);
            _lastRetraining = DateTime.UtcNow;
            
            _logger.LogInformation(
                "Model retraining completed successfully at {Timestamp}",
                _lastRetraining);
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Model retraining failed");
            return false;
        }
    }
}
```

### Comparing Statistical vs ML Approaches

After implementing both approaches, let's analyze when to use each:

| Aspect | Statistical (Z-Score/IQR) | ML (FastTree Classifier) |
|--------|---------------------------|--------------------------|
| **Training Data** | Minimal—just baseline stats | Requires labeled examples |
| **Interpretability** | Highly interpretable | Moderate (feature importance) |
| **Latency** | Sub-millisecond | 1-10ms typical |
| **Feature Interactions** | Cannot capture | Excels at complex patterns |
| **Concept Drift** | Handles naturally with rolling windows | Requires retraining |
| **Cold Start** | Works immediately | Needs historical fraud examples |

The practical recommendation: use statistical methods as a first-pass filter for obvious anomalies, then apply ML models for nuanced detection. This layered approach reduces computational load while maintaining high detection rates.

```csharp
public class HybridFraudDetector
{
    private readonly StatisticalFraudDetector _statisticalDetector;
    private readonly MLFraudDetector _mlDetector;
    
    public HybridFraudDetector(
        StatisticalFraudDetector statisticalDetector,
        MLFraudDetector mlDetector)
    {
        _statisticalDetector = statisticalDetector;
        _mlDetector = mlDetector;
    }
    
    public FraudDecision Evaluate(Transaction transaction)
    {
        // First pass: statistical detection (fast)
        var statisticalResult = _statisticalDetector.Predict(transaction);
        
        // If statistical score is very high, block immediately
        if (statisticalResult.Score > 5.0)
        {
            return new FraudDecision
            {
                TransactionId = transaction.TransactionId,
                IsFraud = true,
                Confidence = 0.99f,
                Action = FraudAction.Block,
                RiskFactors = statisticalResult.RiskFactors,
                DetectionMethod = "Statistical (high confidence)"
            };
        }
        
        // If statistical score is moderate, use ML for deeper analysis
        if (statisticalResult.Score > 1.0)
        {
            var (mlFraud, mlProbability) = _mlDetector.Predict(transaction);
            
            return new FraudDecision
            {
                TransactionId = transaction.TransactionId,
                IsFraud = mlProbability > 0.7,
                Confidence = mlProbability,
                Action = mlProbability > 0.7 ? FraudAction.Block 
                       : mlProbability > 0.4 ? FraudAction.Review 
                       : FraudAction.Approve,
                RiskFactors = statisticalResult.RiskFactors,
                DetectionMethod = "ML (statistical flagged)"
            };
        }
        
        // Low statistical score: approve without ML overhead
        return new FraudDecision
        {
            TransactionId = transaction.TransactionId,
            IsFraud = false,
            Confidence = 0.1f,
            Action = FraudAction.Approve,
            RiskFactors = new List<string>(),
            DetectionMethod = "Statistical (low risk)"
        };
    }
}
```

This hybrid approach processes 80% of transactions with just statistical methods (sub-millisecond), reserving ML inference for the 20% that need deeper scrutiny.

## Summary

Anomaly detection transforms data streams into actionable intelligence. We've explored statistical foundations with Z-scores and IQR methods, ML.NET's sophisticated time series APIs for spike and change point detection, and built a complete fraud detection system comparing multiple approaches.

The key insights for production deployment:

1. **Choose thresholds based on business costs**, not arbitrary statistical boundaries. A false negative that costs $10,000 justifies many $10 false positive investigations.

2. **Layer your detection** with fast statistical methods as a first pass and ML models for nuanced analysis.

3. **Implement proper aggregation** to avoid alert fatigue—operations teams ignore systems that cry wolf.

4. **Monitor model health continuously**. Fraudsters adapt, and yesterday's perfect model becomes tomorrow's vulnerability.

5. **Build feedback loops** from fraud investigations back into retraining pipelines.

6. **Design for graceful degradation**. When your ML model fails, fall back to statistical methods rather than blocking all transactions.

Remember that anomaly detection is as much about operational design as it is about algorithms. The best model in the world is useless if alert fatigue causes operators to ignore it, or if false positives drive away legitimate customers.

In the next chapter, we'll explore ensemble methods and model composition—combining multiple models to achieve better performance than any single approach could deliver alone.
