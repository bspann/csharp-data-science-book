// Chapter 14: Time Series Forecasting - Retail Sales Forecasting with SSA
// Demonstrates ML.NET's Singular Spectrum Analysis for time series forecasting

using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;

namespace Chapter14.SalesForecasting;

// Input data schema for time series
public class MonthlySales
{
    public DateTime Date { get; set; }
    public float Sales { get; set; }
}

// Schema for SSA model input
public class SalesInput
{
    public float Sales { get; set; }
}

// Schema for SSA model output with confidence intervals
public class SalesPrediction
{
    public float[] ForecastedSales { get; set; } = [];
    public float[] LowerBound { get; set; } = [];
    public float[] UpperBound { get; set; } = [];
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("╔════════════════════════════════════════════════════════╗");
        Console.WriteLine("║  Chapter 14: Retail Sales Forecasting with ML.NET SSA  ║");
        Console.WriteLine("╚════════════════════════════════════════════════════════╝\n");

        // Step 1: Generate sample monthly sales data (36 months = 3 years)
        var salesData = GenerateSampleMonthlySalesData();
        
        // Step 2: Explore the data
        PrintDataSummary(salesData);
        AnalyzeSeasonality(salesData);
        
        // Step 3: Split into train/test for evaluation (hold out last 6 months)
        int holdoutMonths = 6;
        var trainData = salesData.Take(salesData.Count - holdoutMonths).ToList();
        var testData = salesData.Skip(salesData.Count - holdoutMonths).ToList();
        
        Console.WriteLine($"\n=== Train/Test Split ===");
        Console.WriteLine($"Training months: {trainData.Count}");
        Console.WriteLine($"Test months: {testData.Count}");
        
        // Step 4: Train and evaluate model
        var mlContext = new MLContext(seed: 42);
        var (model, engine) = TrainSSAModel(mlContext, trainData, horizon: holdoutMonths);
        
        // Step 5: Evaluate on held-out test data
        var testPrediction = engine.Predict();
        var actual = testData.Select(d => d.Sales).ToArray();
        var predicted = testPrediction.ForecastedSales;
        
        Console.WriteLine($"\n=== Model Evaluation (Hold-out Test) ===");
        EvaluateForecast(actual, predicted, testPrediction.LowerBound, testPrediction.UpperBound);
        
        // Step 6: Retrain on full data and forecast future
        Console.WriteLine($"\n=== Training Final Model on All Data ===");
        var (finalModel, finalEngine) = TrainSSAModel(mlContext, salesData, horizon: 6);
        
        // Step 7: Generate forecast for next 6 months
        var forecast = finalEngine.Predict();
        
        Console.WriteLine($"\n=== 6-Month Sales Forecast ===");
        Console.WriteLine("─────────────────────────────────────────────────────────────");
        
        var lastDate = salesData.Last().Date;
        float totalForecast = 0;
        
        for (int i = 0; i < forecast.ForecastedSales.Length; i++)
        {
            var forecastDate = lastDate.AddMonths(i + 1);
            var monthName = forecastDate.ToString("MMMM yyyy");
            Console.WriteLine($"{monthName,-15}: ${forecast.ForecastedSales[i]:N0,10} " +
                            $"(95% CI: ${forecast.LowerBound[i]:N0,10} - ${forecast.UpperBound[i]:N0,10})");
            totalForecast += forecast.ForecastedSales[i];
        }
        
        Console.WriteLine("─────────────────────────────────────────────────────────────");
        Console.WriteLine($"{"Total Forecast",-15}: ${totalForecast:N0}");
        
        // Step 8: Save model
        string modelPath = "sales_forecast_model.zip";
        finalEngine.CheckPoint(mlContext, modelPath);
        Console.WriteLine($"\n✓ Model saved to {modelPath}");
        
        Console.WriteLine("\n=== Forecasting Complete ===");
    }

    /// <summary>
    /// Generates 36 months of sample retail sales data with trend and seasonality.
    /// </summary>
    static List<MonthlySales> GenerateSampleMonthlySalesData()
    {
        var data = new List<MonthlySales>();
        var random = new Random(42);
        var startDate = new DateTime(2022, 1, 1);
        
        // Monthly seasonality factors (retail pattern: holiday spike in Nov/Dec)
        float[] monthlySeasonality = 
        {
            0.85f,  // January - post-holiday slump
            0.82f,  // February - lowest
            0.90f,  // March - spring uptick
            0.95f,  // April
            1.00f,  // May
            1.02f,  // June - summer
            0.98f,  // July
            1.00f,  // August - back to school
            1.05f,  // September
            1.08f,  // October
            1.20f,  // November - Black Friday
            1.45f   // December - holidays
        };
        
        float baseSales = 100000f;  // Base monthly sales
        float trendGrowth = 0.008f; // 0.8% monthly growth (~10% annual)
        
        for (int month = 0; month < 36; month++)
        {
            var date = startDate.AddMonths(month);
            int monthIndex = date.Month - 1;
            
            // Calculate sales: base * trend * seasonality * noise
            float trend = (float)Math.Pow(1 + trendGrowth, month);
            float seasonality = monthlySeasonality[monthIndex];
            float noise = 0.95f + (float)random.NextDouble() * 0.10f; // ±5% noise
            
            float sales = baseSales * trend * seasonality * noise;
            
            data.Add(new MonthlySales
            {
                Date = date,
                Sales = (float)Math.Round(sales, 0)
            });
        }
        
        return data;
    }

    /// <summary>
    /// Prints summary statistics of the sales data.
    /// </summary>
    static void PrintDataSummary(List<MonthlySales> data)
    {
        Console.WriteLine("=== Sales Data Summary ===");
        Console.WriteLine($"Date range:      {data.First().Date:MMM yyyy} to {data.Last().Date:MMM yyyy}");
        Console.WriteLine($"Total months:    {data.Count}");
        Console.WriteLine($"Average sales:   ${data.Average(d => d.Sales):N0}");
        Console.WriteLine($"Min sales:       ${data.Min(d => d.Sales):N0}");
        Console.WriteLine($"Max sales:       ${data.Max(d => d.Sales):N0}");
        Console.WriteLine($"Std deviation:   ${StandardDeviation(data.Select(d => d.Sales)):N0}");
    }

    /// <summary>
    /// Analyzes seasonal patterns by averaging sales by month across years.
    /// </summary>
    static void AnalyzeSeasonality(List<MonthlySales> data)
    {
        Console.WriteLine("\n=== Seasonality Analysis (Average by Month) ===");
        
        var byMonth = data
            .GroupBy(d => d.Date.Month)
            .OrderBy(g => g.Key)
            .Select(g => new 
            { 
                Month = g.Key, 
                AvgSales = g.Average(x => x.Sales),
                MonthName = new DateTime(2024, g.Key, 1).ToString("MMM")
            });
        
        float maxSales = (float)byMonth.Max(m => m.AvgSales);
        
        foreach (var month in byMonth)
        {
            int barLength = (int)(month.AvgSales / maxSales * 30);
            string bar = new string('█', barLength);
            Console.WriteLine($"{month.MonthName}: ${month.AvgSales,10:N0} {bar}");
        }
    }

    /// <summary>
    /// Trains an SSA (Singular Spectrum Analysis) forecasting model.
    /// </summary>
    static (ITransformer, TimeSeriesPredictionEngine<SalesInput, SalesPrediction>) TrainSSAModel(
        MLContext mlContext,
        List<MonthlySales> data,
        int horizon)
    {
        // Convert to input format
        var inputData = data.Select(d => new SalesInput { Sales = d.Sales }).ToList();
        var dataView = mlContext.Data.LoadFromEnumerable(inputData);
        
        // Window size: for monthly data with yearly seasonality, use 6 (half year)
        // This allows the model to capture the seasonal pattern
        int windowSize = Math.Min(6, inputData.Count / 2);
        
        Console.WriteLine($"Training SSA model...");
        Console.WriteLine($"  Series length: {inputData.Count}");
        Console.WriteLine($"  Window size:   {windowSize}");
        Console.WriteLine($"  Horizon:       {horizon}");
        
        // Configure SSA forecasting pipeline
        var forecastingPipeline = mlContext.Forecasting.ForecastBySsa(
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
        
        // Fit the model
        var model = forecastingPipeline.Fit(dataView);
        
        // Create prediction engine for time series
        var engine = model.CreateTimeSeriesEngine<SalesInput, SalesPrediction>(mlContext);
        
        Console.WriteLine("  Model trained successfully.");
        
        return (model, engine);
    }

    /// <summary>
    /// Evaluates forecast accuracy using MAE, RMSE, and MAPE.
    /// </summary>
    static void EvaluateForecast(float[] actual, float[] predicted, float[] lowerBound, float[] upperBound)
    {
        // Calculate evaluation metrics
        double mae = CalculateMAE(actual, predicted);
        double rmse = CalculateRMSE(actual, predicted);
        double mape = CalculateMAPE(actual, predicted);
        
        Console.WriteLine($"MAE (Mean Absolute Error):     ${mae:N0}");
        Console.WriteLine($"RMSE (Root Mean Squared Error): ${rmse:N0}");
        Console.WriteLine($"MAPE (Mean Abs % Error):        {mape:F1}%");
        
        // Show actual vs predicted
        Console.WriteLine("\nActual vs Predicted:");
        int covered = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            bool inCI = actual[i] >= lowerBound[i] && actual[i] <= upperBound[i];
            if (inCI) covered++;
            string status = inCI ? "✓" : "✗";
            Console.WriteLine($"  Month {i + 1}: Actual=${actual[i]:N0}, Predicted=${predicted[i]:N0}, " +
                            $"CI=[${lowerBound[i]:N0} - ${upperBound[i]:N0}] {status}");
        }
        
        Console.WriteLine($"\n95% CI Coverage: {covered}/{actual.Length} ({100.0 * covered / actual.Length:F0}%)");
    }

    /// <summary>
    /// Mean Absolute Error: average of absolute differences.
    /// </summary>
    static double CalculateMAE(float[] actual, float[] predicted)
    {
        double sum = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            sum += Math.Abs(actual[i] - predicted[i]);
        }
        return sum / actual.Length;
    }

    /// <summary>
    /// Root Mean Squared Error: penalizes large errors more heavily.
    /// </summary>
    static double CalculateRMSE(float[] actual, float[] predicted)
    {
        double sumSquared = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double error = actual[i] - predicted[i];
            sumSquared += error * error;
        }
        return Math.Sqrt(sumSquared / actual.Length);
    }

    /// <summary>
    /// Mean Absolute Percentage Error: scale-independent accuracy metric.
    /// </summary>
    static double CalculateMAPE(float[] actual, float[] predicted)
    {
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
        return count > 0 ? (sum / count) * 100 : 0;
    }

    /// <summary>
    /// Calculates standard deviation of a sequence.
    /// </summary>
    static double StandardDeviation(IEnumerable<float> values)
    {
        var list = values.ToList();
        double avg = list.Average();
        double sumSquares = list.Sum(v => (v - avg) * (v - avg));
        return Math.Sqrt(sumSquares / list.Count);
    }
}
