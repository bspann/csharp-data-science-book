using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace Chapter12.MovieRecommendations;

public class MovieRating
{
    public float UserId { get; set; }
    public float MovieId { get; set; }
    public float Rating { get; set; }
}

public class MovieRatingPrediction
{
    public float Label { get; set; }
    public float Score { get; set; }
}

class Program
{
    static void Main(string[] args)
    {
        Console.WriteLine("=== Movie Recommendations with ML.NET ===\n");

        var mlContext = new MLContext(seed: 42);

        // Sample ratings data
        var ratings = new List<MovieRating>
        {
            // User 1 likes action movies (1, 2), dislikes romance (3)
            new() { UserId = 1, MovieId = 1, Rating = 5 },
            new() { UserId = 1, MovieId = 2, Rating = 4 },
            new() { UserId = 1, MovieId = 3, Rating = 2 },
            // User 2 likes romance (3, 4), dislikes action (1)
            new() { UserId = 2, MovieId = 1, Rating = 2 },
            new() { UserId = 2, MovieId = 3, Rating = 5 },
            new() { UserId = 2, MovieId = 4, Rating = 4 },
            // User 3 similar to User 1
            new() { UserId = 3, MovieId = 1, Rating = 4 },
            new() { UserId = 3, MovieId = 2, Rating = 5 },
            new() { UserId = 3, MovieId = 5, Rating = 4 },
            // User 4 likes both
            new() { UserId = 4, MovieId = 1, Rating = 4 },
            new() { UserId = 4, MovieId = 3, Rating = 4 },
            new() { UserId = 4, MovieId = 5, Rating = 5 },
        };

        var dataView = mlContext.Data.LoadFromEnumerable(ratings);

        // Configure matrix factorization
        var options = new MatrixFactorizationTrainer.Options
        {
            MatrixColumnIndexColumnName = "UserIdEncoded",
            MatrixRowIndexColumnName = "MovieIdEncoded",
            LabelColumnName = nameof(MovieRating.Rating),
            NumberOfIterations = 20,
            ApproximationRank = 8,
            Quiet = true
        };

        // Train - need to convert to key types first
        Console.WriteLine("Training recommendation model...");
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                outputColumnName: "UserIdEncoded", 
                inputColumnName: nameof(MovieRating.UserId))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(
                outputColumnName: "MovieIdEncoded", 
                inputColumnName: nameof(MovieRating.MovieId)))
            .Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));
        var model = pipeline.Fit(dataView);
        Console.WriteLine("Training complete!\n");

        // Predict
        var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

        Console.WriteLine("=== Predictions ===\n");
        
        // Predict rating for User 1 on Movie 5 (which they haven't rated)
        var testRatings = new[]
        {
            new MovieRating { UserId = 1, MovieId = 5 },  // User 1 on action movie 5
            new MovieRating { UserId = 2, MovieId = 2 },  // User 2 on action movie 2
            new MovieRating { UserId = 3, MovieId = 3 },  // User 3 on romance movie 3
        };

        var movieNames = new Dictionary<float, string>
        {
            { 1, "Die Hard" }, { 2, "Mad Max" }, { 3, "The Notebook" },
            { 4, "Pride & Prejudice" }, { 5, "John Wick" }
        };

        foreach (var test in testRatings)
        {
            var prediction = predictionEngine.Predict(test);
            Console.WriteLine($"User {test.UserId} → {movieNames[test.MovieId]}");
            Console.WriteLine($"  Predicted Rating: {prediction.Score:F1}\n");
        }

        Console.WriteLine("Done!");
    }
}
