using Microsoft.ML;
using Microsoft.ML.Data;

namespace Chapter07.IrisClassification;

// Input data class
public class IrisData
{
    [LoadColumn(0)] public float SepalLength { get; set; }
    [LoadColumn(1)] public float SepalWidth { get; set; }
    [LoadColumn(2)] public float PetalLength { get; set; }
    [LoadColumn(3)] public float PetalWidth { get; set; }
    [LoadColumn(4)] public string Species { get; set; } = string.Empty;
}

// Prediction output class
public class IrisPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedSpecies { get; set; } = string.Empty;
    public float[] Score { get; set; } = Array.Empty<float>();
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== Iris Classification with ML.NET ===\n");

        var mlContext = new MLContext(seed: 42);

        // Create sample data (in real use, load from CSV)
        var trainingData = new List<IrisData>
        {
            new() { SepalLength = 5.1f, SepalWidth = 3.5f, PetalLength = 1.4f, PetalWidth = 0.2f, Species = "setosa" },
            new() { SepalLength = 4.9f, SepalWidth = 3.0f, PetalLength = 1.4f, PetalWidth = 0.2f, Species = "setosa" },
            new() { SepalLength = 4.7f, SepalWidth = 3.2f, PetalLength = 1.3f, PetalWidth = 0.2f, Species = "setosa" },
            new() { SepalLength = 5.0f, SepalWidth = 3.6f, PetalLength = 1.4f, PetalWidth = 0.2f, Species = "setosa" },
            new() { SepalLength = 7.0f, SepalWidth = 3.2f, PetalLength = 4.7f, PetalWidth = 1.4f, Species = "versicolor" },
            new() { SepalLength = 6.4f, SepalWidth = 3.2f, PetalLength = 4.5f, PetalWidth = 1.5f, Species = "versicolor" },
            new() { SepalLength = 6.9f, SepalWidth = 3.1f, PetalLength = 4.9f, PetalWidth = 1.5f, Species = "versicolor" },
            new() { SepalLength = 5.5f, SepalWidth = 2.3f, PetalLength = 4.0f, PetalWidth = 1.3f, Species = "versicolor" },
            new() { SepalLength = 6.3f, SepalWidth = 3.3f, PetalLength = 6.0f, PetalWidth = 2.5f, Species = "virginica" },
            new() { SepalLength = 5.8f, SepalWidth = 2.7f, PetalLength = 5.1f, PetalWidth = 1.9f, Species = "virginica" },
            new() { SepalLength = 7.1f, SepalWidth = 3.0f, PetalLength = 5.9f, PetalWidth = 2.1f, Species = "virginica" },
            new() { SepalLength = 6.5f, SepalWidth = 3.0f, PetalLength = 5.8f, PetalWidth = 2.2f, Species = "virginica" },
        };

        var dataView = mlContext.Data.LoadFromEnumerable(trainingData);

        // Build the pipeline
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Species")
            .Append(mlContext.Transforms.Concatenate("Features",
                nameof(IrisData.SepalLength),
                nameof(IrisData.SepalWidth),
                nameof(IrisData.PetalLength),
                nameof(IrisData.PetalWidth)))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                labelColumnName: "Label",
                featureColumnName: "Features"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

        // Train
        Console.WriteLine("Training model...");
        var model = pipeline.Fit(dataView);
        Console.WriteLine("Training complete!\n");

        // Create prediction engine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

        // Test predictions
        var testSamples = new[]
        {
            new IrisData { SepalLength = 5.1f, SepalWidth = 3.5f, PetalLength = 1.4f, PetalWidth = 0.2f },
            new IrisData { SepalLength = 6.3f, SepalWidth = 2.8f, PetalLength = 5.1f, PetalWidth = 1.5f },
        };

        Console.WriteLine("=== Predictions ===\n");
        foreach (var sample in testSamples)
        {
            var prediction = predictionEngine.Predict(sample);
            Console.WriteLine($"Flower: [{sample.SepalLength}, {sample.SepalWidth}, {sample.PetalLength}, {sample.PetalWidth}]");
            Console.WriteLine($"  Predicted: {prediction.PredictedSpecies}");
            Console.WriteLine($"  Scores: [{string.Join(", ", prediction.Score.Select(s => $"{s:F3}"))}]\n");
        }

        Console.WriteLine("Done!");
    }
}
