# Chapter 14: Time Series Forecasting

Every business runs on predictions about the future. How many units will we sell next month? What will our server load look like during the holiday season? When will this machine need maintenance? These questions share a common thread: they involve data that changes over time, and answering them requires understanding patterns in historical sequences.

Time series forecasting is one of the most practically valuable skills in data science. Unlike the classification and regression problems we've covered, time series data has a critical property: **order matters**. The sequence of events, the rhythm of measurements, the march of time itself—these become features you can exploit for prediction.

If you've worked with financial data, IoT sensors, user analytics, or operational metrics, you've already touched time series. In this chapter, we'll formalize that intuition and build real forecasting systems using ML.NET's time series capabilities.

## What Makes Time Series Special

Before we dive into code, let's establish why time series deserves its own chapter—why you can't just throw temporal data at a standard regression model and call it a day.

### The Autocorrelation Problem

In traditional machine learning, we assume observations are *independent*. Each training example is its own island, unrelated to others. This assumption lets us shuffle data, split it randomly, and treat each prediction as isolated.

Time series violates this assumption fundamentally. Today's stock price depends on yesterday's. This hour's website traffic correlates with the same hour last week. The temperature right now is heavily influenced by the temperature an hour ago.

This dependency—where a value correlates with its own past values—is called **autocorrelation**, and it's both a challenge and an opportunity. It's a challenge because naive train/test splits will leak information. It's an opportunity because we can exploit these patterns for prediction.

```csharp
// This is WRONG for time series
var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
// Random splitting breaks temporal order and causes data leakage!

// This is RIGHT for time series
// Use the first 80% chronologically for training, last 20% for testing
var trainData = data.Take(totalRows * 0.8);
var testData = data.Skip(totalRows * 0.8);
```

### The Three Components of Time Series

Most time series can be decomposed into three fundamental components:

**1. Trend**
The long-term direction of the data. Is your metric generally increasing, decreasing, or staying flat over months or years? Trend captures the underlying trajectory ignoring short-term fluctuations.

**2. Seasonality**
Regular, predictable patterns that repeat at fixed intervals. Retail sales spike in December. Energy consumption peaks in summer afternoons. Website traffic drops on weekends. Seasonality can occur at multiple scales: daily, weekly, monthly, or yearly.

**3. Noise (Residuals)**
The random variation left over after accounting for trend and seasonality. This is the unpredictable component—the chaos that no model can forecast. Some time series are mostly noise; others have strong patterns that make forecasting easier.

[FIGURE: Time series decomposition diagram showing a sales chart at the top, broken down into three stacked components: a smooth upward trend line, a repeating seasonal wave pattern, and irregular noise fluctuations]

Understanding this decomposition is crucial because different forecasting methods handle these components differently. Some methods excel at capturing trends; others are better at seasonality. The best approach depends on which components dominate your data.

### Stationarity: The Hidden Requirement

Many classical forecasting methods assume your time series is *stationary*—meaning its statistical properties (mean, variance) don't change over time. A stationary series fluctuates around a constant mean and has consistent volatility.

Real-world data is almost never stationary. Sales grow over time. User counts increase. Prices have trends. This non-stationarity must be addressed before applying many forecasting techniques.

Common approaches include:
- **Differencing**: Instead of predicting the value, predict the *change* from the previous value
- **Detrending**: Remove the trend component, forecast the remainder, then add trend back
- **Log transformation**: Stabilize variance that grows with the level

```csharp
// Manual differencing example
public static double[] Difference(double[] series)
{
    var differenced = new double[series.Length - 1];
    for (int i = 1; i < series.Length; i++)
    {
        differenced[i - 1] = series[i] - series[i - 1];
    }
    return differenced;
}

// Now differenced series represents changes, not levels
// Often more stationary than the original
```

## Time Series in ML.NET

ML.NET provides time series forecasting through the `Microsoft.ML.TimeSeries` package. Unlike some ML.NET components that follow the standard pipeline pattern, time series forecasting uses a specialized API designed for sequential data.

### Setting Up Your Project

Create a new project and add the required packages:

```bash
dotnet new console -n SalesForecasting
cd SalesForecasting
dotnet add package Microsoft.ML --version 5.0.0
dotnet add package Microsoft.ML.TimeSeries --version 5.0.0
```

Your `.csproj` should include:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.ML" Version="5.0.0" />
    <PackageReference Include="Microsoft.ML.TimeSeries" Version="5.0.0" />
  </ItemGroup>
</Project>
```

### The SSA Algorithm: Singular Spectrum Analysis

ML.NET's primary forecasting algorithm is **Singular Spectrum Analysis (SSA)**. Don't let the mathematical name intimidate you—the concept is intuitive.

SSA works by:
1. Converting your time series into a matrix of overlapping windows
2. Using matrix decomposition to identify underlying patterns
3. Separating signal (trend + seasonality) from noise
4. Projecting those patterns forward for forecasting

Think of it like this: SSA finds the "skeleton" of your time series—the core patterns that repeat—and uses that skeleton to predict future values. The noise gets filtered out, leaving a smoother, more forecastable signal.

Why SSA instead of classical methods like ARIMA? SSA has several advantages for practical applications:
- **No stationarity requirement**: SSA handles trends and seasonality naturally
- **Multiple seasonalities**: Can capture patterns at different frequencies
- **Robust to noise**: The decomposition acts as built-in smoothing
- **Simple API**: Fewer hyperparameters than ARIMA-style models

### Your First Forecast

Let's build a minimal forecasting example to understand the API:

```csharp
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

// Define input and output schemas
public class SalesData
{
    public float Sales { get; set; }
}

public class SalesForecast
{
    public float[] ForecastedSales { get; set; } = Array.Empty<float>();
    public float[] LowerBound { get; set; } = Array.Empty<float>();
    public float[] UpperBound { get; set; } = Array.Empty<float>();
}

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext(seed: 42);
        
        // Sample data: monthly sales with trend and seasonality
        var salesData = new List<SalesData>
        {
            new() { Sales = 100 }, new() { Sales = 120 }, new() { Sales = 135 },
            new() { Sales = 110 }, new() { Sales = 95 },  new() { Sales = 85 },
            new() { Sales = 90 },  new() { Sales = 100 }, new() { Sales = 140 },
            new() { Sales = 160 }, new() { Sales = 180 }, new() { Sales = 200 },
            // Year 2 - same pattern but higher (trend)
            new() { Sales = 130 }, new() { Sales = 150 }, new() { Sales = 165 },
            new() { Sales = 140 }, new() { Sales = 125 }, new() { Sales = 115 },
            new() { Sales = 120 }, new() { Sales = 130 }, new() { Sales = 170 },
            new() { Sales = 190 }, new() { Sales = 210 }, new() { Sales = 230 },
        };
        
        var dataView = mlContext.Data.LoadFromEnumerable(salesData);
        
        // Configure the forecasting model
        var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
            outputColumnName: nameof(SalesForecast.ForecastedSales),
            inputColumnName: nameof(SalesData.Sales),
            windowSize: 6,                    // Pattern detection window
            seriesLength: salesData.Count,    // Total history length
            trainSize: salesData.Count,       // Training data size
            horizon: 3,                       // How many periods to forecast
            confidenceLevel: 0.95f,           // 95% confidence intervals
            confidenceLowerBoundColumn: nameof(SalesForecast.LowerBound),
            confidenceUpperBoundColumn: nameof(SalesForecast.UpperBound)
        );
        
        // Train the model
        var model = forecastingPipeline.Fit(dataView);
        
        // Create forecasting engine
        var forecastEngine = model.CreateTimeSeriesEngine<SalesData, SalesForecast>(mlContext);
        
        // Generate forecast
        var forecast = forecastEngine.Predict();
        
        Console.WriteLine("Sales Forecast for Next 3 Months:");
        Console.WriteLine("----------------------------------");
        for (int i = 0; i < forecast.ForecastedSales.Length; i++)
        {
            Console.WriteLine($"Month {i + 1}: {forecast.ForecastedSales[i]:F0} " +
                            $"(95% CI: {forecast.LowerBound[i]:F0} - {forecast.UpperBound[i]:F0})");
        }
    }
}
```

Output:
```
Sales Forecast for Next 3 Months:
----------------------------------
Month 1: 158 (95% CI: 128 - 188)
Month 2: 178 (95% CI: 143 - 213)
Month 3: 192 (95% CI: 152 - 232)
```

### Understanding the SSA Parameters

The `ForecastBySsa` method has several important parameters. Let's understand what each does:

| Parameter | Purpose | Guidance |
|-----------|---------|----------|
| `windowSize` | Size of the sliding window for pattern detection | Typically `seriesLength / 2` or related to your seasonality period |
| `seriesLength` | Length of series used for training | Your historical data size |
| `trainSize` | Portion of series used for model fitting | Usually same as `seriesLength` |
| `horizon` | Number of future periods to forecast | How far ahead you need predictions |
| `confidenceLevel` | Probability coverage for intervals | 0.95 for 95% confidence intervals |

The `windowSize` parameter is the most important for capturing seasonality. If you have monthly data with yearly seasonality, a window size of 6-12 works well. For daily data with weekly seasonality, try 7 or a multiple of 7.

```csharp
// For monthly data with yearly seasonality
int windowSize = 6;  // Half a year captures the pattern

// For daily data with weekly seasonality
int windowSize = 7;  // One full week

// For hourly data with daily seasonality
int windowSize = 24; // One full day
```

## Analyzing Your Time Series

Before building a forecasting model, you should understand your data's characteristics. This exploratory phase identifies patterns, anomalies, and the appropriate model parameters.

### Visualizing Trends and Patterns

While ML.NET doesn't include visualization, you can export data for plotting or use Polyglot Notebooks with plotting libraries:

```csharp
using Microsoft.Data.Analysis;

// Load time series into DataFrame for exploration
var df = DataFrame.LoadCsv("sales_data.csv");

// Calculate rolling statistics
var sales = df["Sales"].Cast<float>().ToArray();
var rollingMean = CalculateRollingMean(sales, windowSize: 12);
var rollingStd = CalculateRollingStd(sales, windowSize: 12);

// Check for trend: is the rolling mean changing over time?
Console.WriteLine($"First year average: {rollingMean.Take(12).Average():F2}");
Console.WriteLine($"Last year average: {rollingMean.TakeLast(12).Average():F2}");

// Check for changing variance: is volatility increasing?
Console.WriteLine($"First year std dev: {rollingStd.Take(12).Average():F2}");
Console.WriteLine($"Last year std dev: {rollingStd.TakeLast(12).Average():F2}");

static float[] CalculateRollingMean(float[] data, int windowSize)
{
    var result = new float[data.Length];
    for (int i = 0; i < data.Length; i++)
    {
        int start = Math.Max(0, i - windowSize + 1);
        int count = i - start + 1;
        result[i] = data.Skip(start).Take(count).Average();
    }
    return result;
}

static float[] CalculateRollingStd(float[] data, int windowSize)
{
    var result = new float[data.Length];
    for (int i = 0; i < data.Length; i++)
    {
        int start = Math.Max(0, i - windowSize + 1);
        var window = data.Skip(start).Take(i - start + 1).ToArray();
        if (window.Length > 1)
        {
            float mean = window.Average();
            result[i] = (float)Math.Sqrt(window.Average(x => (x - mean) * (x - mean)));
        }
    }
    return result;
}
```

### Detecting Seasonality

To identify seasonal patterns, look for repeating cycles in your data. A simple approach is to compute the autocorrelation at different lags:

```csharp
public static double[] CalculateAutocorrelation(float[] series, int maxLag)
{
    var result = new double[maxLag];
    double mean = series.Average();
    double variance = series.Average(x => (x - mean) * (x - mean));
    
    for (int lag = 1; lag <= maxLag; lag++)
    {
        double sum = 0;
        int count = 0;
        
        for (int i = lag; i < series.Length; i++)
        {
            sum += (series[i] - mean) * (series[i - lag] - mean);
            count++;
        }
        
        result[lag - 1] = (sum / count) / variance;
    }
    
    return result;
}

// Usage
var autocorr = CalculateAutocorrelation(sales, maxLag: 24);

Console.WriteLine("Autocorrelation Analysis:");
Console.WriteLine("-------------------------");
for (int i = 0; i < autocorr.Length; i++)
{
    int lag = i + 1;
    string bar = new string('█', (int)(Math.Abs(autocorr[i]) * 20));
    Console.WriteLine($"Lag {lag,2}: {autocorr[i]:F3} {bar}");
}
```

High autocorrelation at specific lags indicates seasonality. If lag 12 has high correlation in monthly data, you have yearly seasonality. If lag 7 has high correlation in daily data, you have weekly seasonality.

[FIGURE: Autocorrelation plot (correlogram) showing bars at different lags, with prominent spikes at lags 12 and 24 indicating yearly seasonality in monthly data]

### Handling Anomalies and Missing Values

Real-world time series often have gaps and outliers. ML.NET's SSA can handle some noise, but extreme anomalies should be addressed:

```csharp
public class TimeSeriesCleaner
{
    public static float[] HandleMissingValues(float[] series, MissingValueStrategy strategy)
    {
        var result = new float[series.Length];
        
        for (int i = 0; i < series.Length; i++)
        {
            if (float.IsNaN(series[i]) || series[i] == 0) // Assuming 0 means missing
            {
                result[i] = strategy switch
                {
                    MissingValueStrategy.ForwardFill => 
                        i > 0 ? result[i - 1] : series.First(x => !float.IsNaN(x) && x != 0),
                    MissingValueStrategy.LinearInterpolation => 
                        InterpolateValue(series, i),
                    MissingValueStrategy.SeasonalValue => 
                        GetSeasonalValue(series, i, seasonLength: 12),
                    _ => series[i]
                };
            }
            else
            {
                result[i] = series[i];
            }
        }
        
        return result;
    }
    
    private static float InterpolateValue(float[] series, int index)
    {
        // Find previous valid value
        int prevIdx = index - 1;
        while (prevIdx >= 0 && (float.IsNaN(series[prevIdx]) || series[prevIdx] == 0))
            prevIdx--;
            
        // Find next valid value
        int nextIdx = index + 1;
        while (nextIdx < series.Length && (float.IsNaN(series[nextIdx]) || series[nextIdx] == 0))
            nextIdx++;
            
        if (prevIdx < 0) return series[nextIdx];
        if (nextIdx >= series.Length) return series[prevIdx];
        
        // Linear interpolation
        float ratio = (float)(index - prevIdx) / (nextIdx - prevIdx);
        return series[prevIdx] + ratio * (series[nextIdx] - series[prevIdx]);
    }
    
    private static float GetSeasonalValue(float[] series, int index, int seasonLength)
    {
        // Use same period from previous season
        int previousSeasonIdx = index - seasonLength;
        if (previousSeasonIdx >= 0 && !float.IsNaN(series[previousSeasonIdx]))
            return series[previousSeasonIdx];
            
        // Fallback to next season
        int nextSeasonIdx = index + seasonLength;
        if (nextSeasonIdx < series.Length && !float.IsNaN(series[nextSeasonIdx]))
            return series[nextSeasonIdx];
            
        return InterpolateValue(series, index);
    }
}

public enum MissingValueStrategy
{
    ForwardFill,
    LinearInterpolation,
    SeasonalValue
}
```

## Evaluation Metrics for Time Series

Evaluating forecasting models requires specialized metrics that account for temporal structure. You can't use accuracy or F1-score—these are continuous predictions, not classifications.

### Mean Absolute Error (MAE)

The average absolute difference between predicted and actual values. Easy to interpret in the same units as your data.

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

```csharp
public static double CalculateMAE(float[] actual, float[] predicted)
{
    if (actual.Length != predicted.Length)
        throw new ArgumentException("Arrays must have same length");
        
    double sum = 0;
    for (int i = 0; i < actual.Length; i++)
    {
        sum += Math.Abs(actual[i] - predicted[i]);
    }
    return sum / actual.Length;
}
```

**Interpretation**: MAE of 50 on sales data means your predictions are off by 50 units on average. Lower is better. Use when you want a simple, interpretable error measure.

### Root Mean Squared Error (RMSE)

The square root of the average squared differences. RMSE penalizes large errors more heavily than MAE.

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

```csharp
public static double CalculateRMSE(float[] actual, float[] predicted)
{
    if (actual.Length != predicted.Length)
        throw new ArgumentException("Arrays must have same length");
        
    double sumSquared = 0;
    for (int i = 0; i < actual.Length; i++)
    {
        double error = actual[i] - predicted[i];
        sumSquared += error * error;
    }
    return Math.Sqrt(sumSquared / actual.Length);
}
```

**Interpretation**: RMSE is always greater than or equal to MAE. If RMSE is much larger than MAE, your model has some big misses. Use when large errors are particularly costly.

### Mean Absolute Percentage Error (MAPE)

The average absolute error as a percentage of actual values. Scale-independent, making it useful for comparing across different metrics.

$$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

```csharp
public static double CalculateMAPE(float[] actual, float[] predicted)
{
    if (actual.Length != predicted.Length)
        throw new ArgumentException("Arrays must have same length");
        
    double sum = 0;
    int count = 0;
    
    for (int i = 0; i < actual.Length; i++)
    {
        if (Math.Abs(actual[i]) > 0.001) // Avoid division by zero
        {
            sum += Math.Abs((actual[i] - predicted[i]) / actual[i]);
            count++;
        }
    }
    
    return (sum / count) * 100;
}
```

**Interpretation**: MAPE of 10% means your predictions are off by 10% on average. Use when you want a percentage-based metric that works across different scales.

**Warning**: MAPE is undefined when actual values are zero and can be misleading when values are close to zero. It also asymmetrically penalizes over-predictions more than under-predictions.

### Comparing Metrics in Practice

```csharp
public class ForecastEvaluator
{
    public static void PrintEvaluation(float[] actual, float[] predicted, string modelName)
    {
        Console.WriteLine($"\n{modelName} Evaluation:");
        Console.WriteLine(new string('-', 40));
        Console.WriteLine($"MAE:  {CalculateMAE(actual, predicted):F2}");
        Console.WriteLine($"RMSE: {CalculateRMSE(actual, predicted):F2}");
        Console.WriteLine($"MAPE: {CalculateMAPE(actual, predicted):F2}%");
        
        // Also show baseline comparison
        var naiveBaseline = CreateNaiveForecast(actual);
        var baselineMAE = CalculateMAE(actual.Skip(1).ToArray(), naiveBaseline);
        Console.WriteLine($"\nNaive Baseline MAE: {baselineMAE:F2}");
        
        double improvement = (baselineMAE - CalculateMAE(actual, predicted)) / baselineMAE * 100;
        Console.WriteLine($"Improvement over baseline: {improvement:F1}%");
    }
    
    private static float[] CreateNaiveForecast(float[] actual)
    {
        // Naive forecast: predict previous value
        return actual.Take(actual.Length - 1).ToArray();
    }
}
```

### Backtesting with Walk-Forward Validation

Standard cross-validation doesn't work for time series because it violates temporal order. Instead, use **walk-forward validation**: train on past data, predict the next period, then expand the training window and repeat.

```csharp
public class WalkForwardValidator
{
    private readonly MLContext _mlContext;
    
    public WalkForwardValidator(MLContext mlContext)
    {
        _mlContext = mlContext;
    }
    
    public WalkForwardResult Validate(
        List<SalesData> fullData,
        int initialTrainingSize,
        int horizon,
        int windowSize)
    {
        var allPredictions = new List<float>();
        var allActuals = new List<float>();
        
        int currentTrainingEnd = initialTrainingSize;
        
        while (currentTrainingEnd + horizon <= fullData.Count)
        {
            // Train on data up to currentTrainingEnd
            var trainingData = fullData.Take(currentTrainingEnd).ToList();
            var testData = fullData.Skip(currentTrainingEnd).Take(horizon).ToList();
            
            // Build and train model
            var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
            
            var pipeline = _mlContext.Forecasting.ForecastBySsa(
                outputColumnName: nameof(SalesForecast.ForecastedSales),
                inputColumnName: nameof(SalesData.Sales),
                windowSize: windowSize,
                seriesLength: trainingData.Count,
                trainSize: trainingData.Count,
                horizon: horizon
            );
            
            var model = pipeline.Fit(dataView);
            var engine = model.CreateTimeSeriesEngine<SalesData, SalesForecast>(_mlContext);
            
            // Generate predictions
            var forecast = engine.Predict();
            
            // Store results
            allPredictions.AddRange(forecast.ForecastedSales);
            allActuals.AddRange(testData.Select(d => d.Sales));
            
            // Slide window forward
            currentTrainingEnd += horizon;
        }
        
        return new WalkForwardResult
        {
            Predictions = allPredictions.ToArray(),
            Actuals = allActuals.ToArray(),
            MAE = (float)CalculateMAE(allActuals.ToArray(), allPredictions.ToArray()),
            RMSE = (float)CalculateRMSE(allActuals.ToArray(), allPredictions.ToArray()),
            MAPE = (float)CalculateMAPE(allActuals.ToArray(), allPredictions.ToArray())
        };
    }
}

public class WalkForwardResult
{
    public float[] Predictions { get; set; } = Array.Empty<float>();
    public float[] Actuals { get; set; } = Array.Empty<float>();
    public float MAE { get; set; }
    public float RMSE { get; set; }
    public float MAPE { get; set; }
}
```

[FIGURE: Walk-forward validation diagram showing a timeline with expanding training windows and fixed-size test windows sliding forward through time]

## Project: Retail Sales Forecasting System

Now let's build a complete sales forecasting system for a retail scenario. This project brings together everything we've covered: data loading, exploration, model training, evaluation, and prediction.

### The Business Problem

You're working for a retail company that needs to forecast weekly sales for inventory planning. Accurate forecasts mean:
- **Less overstock**: Don't tie up capital in unsold inventory
- **Fewer stockouts**: Don't lose sales to empty shelves
- **Better staffing**: Schedule employees based on expected demand

The dataset contains 143 weeks of sales data for a single store, including the date and weekly sales amount. We need to forecast the next 8 weeks.

### Step 1: Define the Data Model

```csharp
using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace RetailForecasting;

public class WeeklySales
{
    public DateTime WeekStartDate { get; set; }
    public float Sales { get; set; }
    public bool IsHoliday { get; set; }
}

public class SalesInput
{
    public float Sales { get; set; }
}

public class SalesPrediction
{
    public float[] ForecastedSales { get; set; } = Array.Empty<float>();
    public float[] LowerBound { get; set; } = Array.Empty<float>();
    public float[] UpperBound { get; set; } = Array.Empty<float>();
}
```

### Step 2: Load and Explore the Data

```csharp
public class SalesDataLoader
{
    public static List<WeeklySales> LoadFromCsv(string path)
    {
        var sales = new List<WeeklySales>();
        
        foreach (var line in File.ReadLines(path).Skip(1)) // Skip header
        {
            var parts = line.Split(',');
            sales.Add(new WeeklySales
            {
                WeekStartDate = DateTime.Parse(parts[0]),
                Sales = float.Parse(parts[1]),
                IsHoliday = parts.Length > 2 && bool.Parse(parts[2])
            });
        }
        
        return sales.OrderBy(s => s.WeekStartDate).ToList();
    }
    
    public static void PrintSummary(List<WeeklySales> data)
    {
        Console.WriteLine("=== Sales Data Summary ===");
        Console.WriteLine($"Date range: {data.First().WeekStartDate:d} to {data.Last().WeekStartDate:d}");
        Console.WriteLine($"Total weeks: {data.Count}");
        Console.WriteLine($"Average weekly sales: ${data.Average(d => d.Sales):N0}");
        Console.WriteLine($"Min weekly sales: ${data.Min(d => d.Sales):N0}");
        Console.WriteLine($"Max weekly sales: ${data.Max(d => d.Sales):N0}");
        Console.WriteLine($"Std deviation: ${StandardDeviation(data.Select(d => d.Sales)):N0}");
        Console.WriteLine($"Holiday weeks: {data.Count(d => d.IsHoliday)}");
    }
    
    private static double StandardDeviation(IEnumerable<float> values)
    {
        var list = values.ToList();
        double avg = list.Average();
        double sumSquares = list.Sum(v => (v - avg) * (v - avg));
        return Math.Sqrt(sumSquares / list.Count);
    }
}
```

### Step 3: Analyze Seasonality

```csharp
public class SeasonalityAnalyzer
{
    public static void AnalyzePatterns(List<WeeklySales> data)
    {
        // Weekly pattern within month
        var byWeekOfMonth = data
            .GroupBy(d => GetWeekOfMonth(d.WeekStartDate))
            .OrderBy(g => g.Key)
            .Select(g => new { Week = g.Key, AvgSales = g.Average(x => x.Sales) });
        
        Console.WriteLine("\n=== Sales by Week of Month ===");
        foreach (var week in byWeekOfMonth)
        {
            string bar = new string('█', (int)(week.AvgSales / 5000));
            Console.WriteLine($"Week {week.Week}: ${week.AvgSales:N0} {bar}");
        }
        
        // Monthly pattern
        var byMonth = data
            .GroupBy(d => d.WeekStartDate.Month)
            .OrderBy(g => g.Key)
            .Select(g => new { 
                Month = g.Key, 
                AvgSales = g.Average(x => x.Sales),
                MonthName = new DateTime(2024, g.Key, 1).ToString("MMM")
            });
        
        Console.WriteLine("\n=== Sales by Month ===");
        foreach (var month in byMonth)
        {
            string bar = new string('█', (int)(month.AvgSales / 5000));
            Console.WriteLine($"{month.MonthName}: ${month.AvgSales:N0} {bar}");
        }
        
        // Year-over-year trend
        var byYear = data
            .GroupBy(d => d.WeekStartDate.Year)
            .OrderBy(g => g.Key)
            .Select(g => new { Year = g.Key, TotalSales = g.Sum(x => x.Sales) });
        
        Console.WriteLine("\n=== Annual Sales ===");
        foreach (var year in byYear)
        {
            Console.WriteLine($"{year.Year}: ${year.TotalSales:N0}");
        }
    }
    
    private static int GetWeekOfMonth(DateTime date)
    {
        DateTime firstDay = new DateTime(date.Year, date.Month, 1);
        return (date.Day + (int)firstDay.DayOfWeek - 1) / 7 + 1;
    }
}
```

### Step 4: Build the Forecasting Model

```csharp
public class SalesForecaster
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private TimeSeriesPredictionEngine<SalesInput, SalesPrediction>? _engine;
    
    public SalesForecaster(int seed = 42)
    {
        _mlContext = new MLContext(seed);
    }
    
    public void Train(List<WeeklySales> historicalData, int horizon = 8)
    {
        // Convert to input format
        var inputData = historicalData
            .Select(d => new SalesInput { Sales = d.Sales })
            .ToList();
        
        var dataView = _mlContext.Data.LoadFromEnumerable(inputData);
        
        // Determine optimal window size based on seasonality
        // For weekly data with yearly seasonality, use ~26 weeks (half year)
        int windowSize = Math.Min(26, inputData.Count / 2);
        
        Console.WriteLine($"\nTraining SSA model with {inputData.Count} weeks of data...");
        Console.WriteLine($"Window size: {windowSize}, Horizon: {horizon}");
        
        var pipeline = _mlContext.Forecasting.ForecastBySsa(
            outputColumnName: nameof(SalesPrediction.ForecastedSales),
            inputColumnName: nameof(SalesInput.Sales),
            windowSize: windowSize,
            seriesLength: inputData.Count,
            trainSize: inputData.Count,
            horizon: horizon,
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: nameof(SalesPrediction.LowerBound),
            confidenceUpperBoundColumn: nameof(SalesPrediction.UpperBound)
        );
        
        _model = pipeline.Fit(dataView);
        _engine = _model.CreateTimeSeriesEngine<SalesInput, SalesPrediction>(_mlContext);
        
        Console.WriteLine("Model trained successfully.");
    }
    
    public SalesPrediction Forecast()
    {
        if (_engine == null)
            throw new InvalidOperationException("Model not trained. Call Train() first.");
            
        return _engine.Predict();
    }
    
    public void UpdateWithActual(float actualValue)
    {
        if (_engine == null)
            throw new InvalidOperationException("Model not trained. Call Train() first.");
            
        // Update model with new observation for continuous learning
        _engine.Predict(new SalesInput { Sales = actualValue });
    }
    
    public void SaveModel(string path)
    {
        if (_engine == null)
            throw new InvalidOperationException("Model not trained. Call Train() first.");
            
        _engine.CheckPoint(_mlContext, path);
        Console.WriteLine($"Model saved to {path}");
    }
    
    public void LoadModel(string path)
    {
        _model = _mlContext.Model.Load(path, out _);
        _engine = _model.CreateTimeSeriesEngine<SalesInput, SalesPrediction>(_mlContext);
        Console.WriteLine($"Model loaded from {path}");
    }
}
```

### Step 5: Evaluate Model Performance

```csharp
public class ModelEvaluator
{
    public static EvaluationResults EvaluateWithHoldout(
        List<WeeklySales> fullData, 
        int holdoutWeeks = 8)
    {
        // Split data
        var trainingData = fullData.Take(fullData.Count - holdoutWeeks).ToList();
        var testData = fullData.TakeLast(holdoutWeeks).ToList();
        
        Console.WriteLine($"\nEvaluating model...");
        Console.WriteLine($"Training weeks: {trainingData.Count}");
        Console.WriteLine($"Test weeks: {holdoutWeeks}");
        
        // Train model
        var forecaster = new SalesForecaster();
        forecaster.Train(trainingData, horizon: holdoutWeeks);
        
        // Generate predictions
        var prediction = forecaster.Forecast();
        
        // Calculate metrics
        var actual = testData.Select(d => d.Sales).ToArray();
        var predicted = prediction.ForecastedSales;
        
        var results = new EvaluationResults
        {
            MAE = CalculateMAE(actual, predicted),
            RMSE = CalculateRMSE(actual, predicted),
            MAPE = CalculateMAPE(actual, predicted),
            Actual = actual,
            Predicted = predicted,
            LowerBound = prediction.LowerBound,
            UpperBound = prediction.UpperBound
        };
        
        // Print results
        Console.WriteLine("\n=== Evaluation Results ===");
        Console.WriteLine($"MAE:  ${results.MAE:N0}");
        Console.WriteLine($"RMSE: ${results.RMSE:N0}");
        Console.WriteLine($"MAPE: {results.MAPE:F1}%");
        
        Console.WriteLine("\n=== Actual vs Predicted ===");
        for (int i = 0; i < holdoutWeeks; i++)
        {
            string coverage = actual[i] >= prediction.LowerBound[i] && 
                            actual[i] <= prediction.UpperBound[i] ? "✓" : "✗";
            Console.WriteLine($"Week {i + 1}: Actual=${actual[i]:N0}, " +
                            $"Predicted=${predicted[i]:N0}, " +
                            $"CI=[${prediction.LowerBound[i]:N0}-${prediction.UpperBound[i]:N0}] {coverage}");
        }
        
        // Calculate confidence interval coverage
        int covered = 0;
        for (int i = 0; i < holdoutWeeks; i++)
        {
            if (actual[i] >= prediction.LowerBound[i] && actual[i] <= prediction.UpperBound[i])
                covered++;
        }
        Console.WriteLine($"\n95% CI Coverage: {covered}/{holdoutWeeks} ({100.0 * covered / holdoutWeeks:F0}%)");
        
        return results;
    }
    
    private static double CalculateMAE(float[] actual, float[] predicted)
    {
        return actual.Zip(predicted, (a, p) => Math.Abs(a - p)).Average();
    }
    
    private static double CalculateRMSE(float[] actual, float[] predicted)
    {
        double mse = actual.Zip(predicted, (a, p) => (a - p) * (a - p)).Average();
        return Math.Sqrt(mse);
    }
    
    private static double CalculateMAPE(float[] actual, float[] predicted)
    {
        return actual.Zip(predicted, (a, p) => Math.Abs((a - p) / a)).Average() * 100;
    }
}

public class EvaluationResults
{
    public double MAE { get; set; }
    public double RMSE { get; set; }
    public double MAPE { get; set; }
    public float[] Actual { get; set; } = Array.Empty<float>();
    public float[] Predicted { get; set; } = Array.Empty<float>();
    public float[] LowerBound { get; set; } = Array.Empty<float>();
    public float[] UpperBound { get; set; } = Array.Empty<float>();
}
```

### Step 6: Production Deployment

```csharp
public class ForecastingService
{
    private readonly SalesForecaster _forecaster;
    private readonly string _modelPath;
    
    public ForecastingService(string modelPath)
    {
        _modelPath = modelPath;
        _forecaster = new SalesForecaster();
        
        if (File.Exists(modelPath))
        {
            _forecaster.LoadModel(modelPath);
        }
    }
    
    public ForecastReport GenerateForecast(DateTime startDate, int weeks = 8)
    {
        var prediction = _forecaster.Forecast();
        
        var report = new ForecastReport
        {
            GeneratedAt = DateTime.UtcNow,
            ForecastStartDate = startDate,
            Weeks = new List<WeeklyForecast>()
        };
        
        for (int i = 0; i < weeks && i < prediction.ForecastedSales.Length; i++)
        {
            report.Weeks.Add(new WeeklyForecast
            {
                WeekNumber = i + 1,
                WeekStartDate = startDate.AddDays(i * 7),
                PredictedSales = prediction.ForecastedSales[i],
                LowerBound = prediction.LowerBound[i],
                UpperBound = prediction.UpperBound[i],
                Confidence = 0.95f
            });
        }
        
        return report;
    }
    
    public void UpdateWithActualSales(float actualSales)
    {
        _forecaster.UpdateWithActual(actualSales);
        _forecaster.SaveModel(_modelPath);
    }
    
    public string GenerateMarkdownReport(ForecastReport report)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# Sales Forecast Report");
        sb.AppendLine($"Generated: {report.GeneratedAt:yyyy-MM-dd HH:mm} UTC");
        sb.AppendLine();
        sb.AppendLine($"## Forecast for {report.Weeks.Count} Weeks Starting {report.ForecastStartDate:MMM d, yyyy}");
        sb.AppendLine();
        sb.AppendLine("| Week | Date | Predicted | Low (95%) | High (95%) |");
        sb.AppendLine("|------|------|-----------|-----------|------------|");
        
        foreach (var week in report.Weeks)
        {
            sb.AppendLine($"| {week.WeekNumber} | {week.WeekStartDate:MMM d} | " +
                        $"${week.PredictedSales:N0} | ${week.LowerBound:N0} | ${week.UpperBound:N0} |");
        }
        
        sb.AppendLine();
        sb.AppendLine($"**Total Forecasted Sales:** ${report.Weeks.Sum(w => w.PredictedSales):N0}");
        
        return sb.ToString();
    }
}

public class ForecastReport
{
    public DateTime GeneratedAt { get; set; }
    public DateTime ForecastStartDate { get; set; }
    public List<WeeklyForecast> Weeks { get; set; } = new();
}

public class WeeklyForecast
{
    public int WeekNumber { get; set; }
    public DateTime WeekStartDate { get; set; }
    public float PredictedSales { get; set; }
    public float LowerBound { get; set; }
    public float UpperBound { get; set; }
    public float Confidence { get; set; }
}
```

### Step 7: Complete Program

```csharp
using RetailForecasting;

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════╗");
        Console.WriteLine("║    Retail Sales Forecasting System   ║");
        Console.WriteLine("╚══════════════════════════════════════╝\n");
        
        // Generate sample data (in production, load from database/file)
        var salesData = GenerateSampleData();
        
        // Explore the data
        SalesDataLoader.PrintSummary(salesData);
        SeasonalityAnalyzer.AnalyzePatterns(salesData);
        
        // Evaluate model performance
        var results = ModelEvaluator.EvaluateWithHoldout(salesData, holdoutWeeks: 8);
        
        // Train final model on all data
        Console.WriteLine("\n=== Training Final Model ===");
        var forecaster = new SalesForecaster();
        forecaster.Train(salesData, horizon: 8);
        
        // Generate forecast
        var prediction = forecaster.Forecast();
        
        Console.WriteLine("\n=== 8-Week Sales Forecast ===");
        Console.WriteLine("─────────────────────────────────────────────────────");
        
        var lastDate = salesData.Last().WeekStartDate;
        float totalForecast = 0;
        
        for (int i = 0; i < prediction.ForecastedSales.Length; i++)
        {
            var weekStart = lastDate.AddDays((i + 1) * 7);
            Console.WriteLine($"Week of {weekStart:MMM dd, yyyy}: " +
                            $"${prediction.ForecastedSales[i]:N0} " +
                            $"(95% CI: ${prediction.LowerBound[i]:N0} - ${prediction.UpperBound[i]:N0})");
            totalForecast += prediction.ForecastedSales[i];
        }
        
        Console.WriteLine("─────────────────────────────────────────────────────");
        Console.WriteLine($"Total 8-Week Forecast: ${totalForecast:N0}");
        
        // Save model for production use
        forecaster.SaveModel("sales_model.zip");
    }
    
    static List<WeeklySales> GenerateSampleData()
    {
        var data = new List<WeeklySales>();
        var random = new Random(42);
        var startDate = new DateTime(2022, 1, 3); // First Monday of 2022
        
        for (int week = 0; week < 143; week++)
        {
            var date = startDate.AddDays(week * 7);
            
            // Base sales
            float sales = 50000;
            
            // Trend: 0.3% weekly growth
            sales *= (float)Math.Pow(1.003, week);
            
            // Monthly seasonality (higher at end of month)
            int dayOfMonth = date.Day;
            if (dayOfMonth >= 25) sales *= 1.15f;
            else if (dayOfMonth <= 7) sales *= 0.95f;
            
            // Yearly seasonality (holiday boost)
            int month = date.Month;
            float[] monthlyFactor = { 0.85f, 0.82f, 0.95f, 1.0f, 1.02f, 1.0f, 
                                     0.95f, 0.98f, 1.0f, 1.05f, 1.15f, 1.35f };
            sales *= monthlyFactor[month - 1];
            
            // Random noise
            sales *= (float)(0.92 + random.NextDouble() * 0.16);
            
            // Check if holiday week
            bool isHoliday = (month == 11 && dayOfMonth >= 20) || // Thanksgiving
                           (month == 12 && dayOfMonth >= 15);    // Christmas
            if (isHoliday) sales *= 1.1f;
            
            data.Add(new WeeklySales
            {
                WeekStartDate = date,
                Sales = (float)Math.Round(sales, 0),
                IsHoliday = isHoliday
            });
        }
        
        return data;
    }
}
```

### Sample Output

```
╔══════════════════════════════════════╗
║    Retail Sales Forecasting System   ║
╚══════════════════════════════════════╝

=== Sales Data Summary ===
Date range: 1/3/2022 to 10/23/2024
Total weeks: 143
Average weekly sales: $55,834
Min weekly sales: $38,472
Max weekly sales: $82,156
Std deviation: $9,247
Holiday weeks: 18

=== Sales by Month ===
Jan: $45,892 █████████
Feb: $44,127 ████████
Mar: $51,234 ██████████
Apr: $53,891 ██████████
...
Nov: $62,456 ████████████
Dec: $73,289 ██████████████

=== Evaluation Results ===
MAE:  $3,847
RMSE: $4,562
MAPE: 6.8%

=== 8-Week Sales Forecast ===
─────────────────────────────────────────────────────
Week of Oct 28, 2024: $58,234 (95% CI: $49,123 - $67,345)
Week of Nov 04, 2024: $61,456 (95% CI: $51,234 - $71,678)
Week of Nov 11, 2024: $63,891 (95% CI: $52,456 - $75,326)
Week of Nov 18, 2024: $68,234 (95% CI: $55,891 - $80,577)
Week of Nov 25, 2024: $72,567 (95% CI: $58,234 - $86,900)
Week of Dec 02, 2024: $69,891 (95% CI: $55,456 - $84,326)
Week of Dec 09, 2024: $74,123 (95% CI: $58,891 - $89,355)
Week of Dec 16, 2024: $78,456 (95% CI: $61,234 - $95,678)
─────────────────────────────────────────────────────
Total 8-Week Forecast: $546,852
Model saved to sales_model.zip
```

## Beyond Basic Forecasting

### Handling Multiple Seasonalities

Real-world data often has patterns at multiple scales: daily, weekly, and yearly. SSA can capture multiple seasonalities by using an appropriate window size:

```csharp
// For daily data with both weekly and yearly patterns
// Window should capture at least the longest seasonal cycle
int windowSize = 365; // Full year to capture yearly seasonality

// For hourly data with daily and weekly patterns
int windowSize = 168; // One full week in hours (24 * 7)
```

### External Regressors

Sometimes forecasts can be improved with external information: holidays, promotions, weather. While basic SSA doesn't directly support external regressors, you can:

1. **Pre-adjust the series**: Remove known effects before forecasting, then add them back
2. **Post-adjust forecasts**: Apply multipliers for known future events
3. **Use regression + time series**: Combine ML.NET regression with time series residuals

```csharp
public float AdjustForHoliday(float baseForecast, DateTime forecastDate)
{
    // Historical holiday lift factors
    if (IsBlackFriday(forecastDate)) return baseForecast * 1.45f;
    if (IsChristmasWeek(forecastDate)) return baseForecast * 1.35f;
    if (IsLaborDay(forecastDate)) return baseForecast * 1.15f;
    
    return baseForecast;
}
```

### Online Learning with Checkpoints

ML.NET's time series engine supports updating the model with new observations without retraining from scratch:

```csharp
// Make a prediction and update the model in one step
var newObservation = new SalesInput { Sales = 62500 };
var forecast = engine.Predict(newObservation);

// Save the updated model state
engine.CheckPoint(mlContext, "updated_model.zip");
```

This is crucial for production systems where you want the model to adapt to new patterns over time.

## Summary

Time series forecasting extends your ML.NET toolkit into a domain where order and time matter as much as the values themselves. In this chapter, you learned:

- **Time series fundamentals**: Autocorrelation, trend, seasonality, and noise—the building blocks of temporal data
- **SSA in ML.NET**: How Singular Spectrum Analysis decomposes and forecasts time series without requiring stationarity
- **Evaluation metrics**: MAE, RMSE, and MAPE each tell you something different about forecast quality
- **Walk-forward validation**: The right way to evaluate time series models without leaking future information
- **Production patterns**: Model persistence, online learning, and continuous updates

The retail forecasting project demonstrated a complete workflow from raw data to deployed predictions. The same patterns apply to demand forecasting, capacity planning, financial projections, and any domain where understanding the future helps you prepare for it.

Time series forecasting is as much art as science. The best practitioners develop intuition for their data's patterns, understand the business context that drives those patterns, and know when the forecast can be trusted—and when it can't. Your software engineering background gives you an advantage: you understand systems, you think about edge cases, and you build for production, not just notebooks.

## Exercises

1. **Multi-step Evaluation**: Modify the `WalkForwardValidator` to track error by forecast horizon (1-week ahead vs 8-weeks ahead). Plot how accuracy degrades as you forecast further into the future. Why does this happen?

2. **Hyperparameter Tuning**: Create a systematic experiment to find the optimal `windowSize` for a dataset. Test window sizes from 4 to 52 for weekly data and measure MAPE for each. What window size works best? How does it relate to the data's seasonality?

3. **Anomaly Detection**: SSA can also detect anomalies. Use `mlContext.AnomalyDetection.DetectSpikeBySsa()` to identify unusual weeks in the sales data. Compare detected spikes against known holidays or promotional events.

4. **Ensemble Forecasting**: Train multiple SSA models with different window sizes and average their predictions. Does the ensemble outperform any single model? Implement weighted averaging where weights are based on historical accuracy.

5. **Real-World Application**: Download a public time series dataset (electricity consumption, stock prices, or weather data) and build a complete forecasting pipeline. Include data exploration, model selection, evaluation, and a prediction report. Document which aspects of the data made forecasting easier or harder.

---

*Next Chapter: Anomaly Detection and Monitoring →*
