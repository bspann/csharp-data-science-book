using Microsoft.ML;
using Microsoft.ML.Data;

namespace Chapter09.CustomerChurn;

public class CustomerData
{
    public float Tenure { get; set; }
    public float MonthlyCharges { get; set; }
    public float TotalCharges { get; set; }
    public bool HasTechSupport { get; set; }
    public bool HasOnlineSecurity { get; set; }
    public bool Churn { get; set; }
}

public class ChurnPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== Customer Churn Prediction with ML.NET ===\n");

        var mlContext = new MLContext(seed: 42);

        // Sample training data
        var trainingData = new List<CustomerData>
        {
            new() { Tenure = 1, MonthlyCharges = 70, TotalCharges = 70, HasTechSupport = false, HasOnlineSecurity = false, Churn = true },
            new() { Tenure = 2, MonthlyCharges = 85, TotalCharges = 170, HasTechSupport = false, HasOnlineSecurity = false, Churn = true },
            new() { Tenure = 24, MonthlyCharges = 50, TotalCharges = 1200, HasTechSupport = true, HasOnlineSecurity = true, Churn = false },
            new() { Tenure = 36, MonthlyCharges = 65, TotalCharges = 2340, HasTechSupport = true, HasOnlineSecurity = true, Churn = false },
            new() { Tenure = 48, MonthlyCharges = 55, TotalCharges = 2640, HasTechSupport = true, HasOnlineSecurity = false, Churn = false },
            new() { Tenure = 3, MonthlyCharges = 90, TotalCharges = 270, HasTechSupport = false, HasOnlineSecurity = false, Churn = true },
            new() { Tenure = 12, MonthlyCharges = 45, TotalCharges = 540, HasTechSupport = true, HasOnlineSecurity = true, Churn = false },
            new() { Tenure = 60, MonthlyCharges = 60, TotalCharges = 3600, HasTechSupport = true, HasOnlineSecurity = true, Churn = false },
        };

        var dataView = mlContext.Data.LoadFromEnumerable(trainingData);

        // Build pipeline
        var pipeline = mlContext.Transforms.Concatenate("Features",
                nameof(CustomerData.Tenure),
                nameof(CustomerData.MonthlyCharges),
                nameof(CustomerData.TotalCharges))
            .Append(mlContext.Transforms.Conversion.ConvertType("HasTechSupportFloat", nameof(CustomerData.HasTechSupport), DataKind.Single))
            .Append(mlContext.Transforms.Conversion.ConvertType("HasOnlineSecurityFloat", nameof(CustomerData.HasOnlineSecurity), DataKind.Single))
            .Append(mlContext.Transforms.Concatenate("AllFeatures", "Features", "HasTechSupportFloat", "HasOnlineSecurityFloat"))
            .Append(mlContext.Transforms.NormalizeMinMax("AllFeatures"))
            .Append(mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: nameof(CustomerData.Churn),
                featureColumnName: "AllFeatures"));

        // Train
        Console.WriteLine("Training model...");
        var model = pipeline.Fit(dataView);
        Console.WriteLine("Training complete!\n");

        // Predict
        var predictionEngine = mlContext.Model.CreatePredictionEngine<CustomerData, ChurnPrediction>(model);

        var testCustomers = new[]
        {
            new CustomerData { Tenure = 1, MonthlyCharges = 95, TotalCharges = 95, HasTechSupport = false, HasOnlineSecurity = false },
            new CustomerData { Tenure = 48, MonthlyCharges = 50, TotalCharges = 2400, HasTechSupport = true, HasOnlineSecurity = true },
        };

        Console.WriteLine("=== Predictions ===\n");
        foreach (var customer in testCustomers)
        {
            var prediction = predictionEngine.Predict(customer);
            Console.WriteLine($"Customer: Tenure={customer.Tenure}, Monthly=${customer.MonthlyCharges}, Support={customer.HasTechSupport}");
            Console.WriteLine($"  Will Churn: {prediction.Prediction} (Probability: {prediction.Probability:P1})\n");
        }

        Console.WriteLine("Done!");
    }
}
