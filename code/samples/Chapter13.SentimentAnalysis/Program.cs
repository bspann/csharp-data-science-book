// Chapter 13: Natural Language Processing - Sentiment Analysis
// Product review sentiment classification using ML.NET

using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

namespace Chapter13.SentimentAnalysis;

#region Data Models

/// <summary>
/// Input data for training - product review with sentiment label
/// </summary>
public class ReviewData
{
    public string ReviewText { get; set; } = string.Empty;
    
    [ColumnName("Label")]
    public bool Sentiment { get; set; } // true = positive, false = negative
}

/// <summary>
/// Model prediction output
/// </summary>
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
    
    public string SentimentLabel => Prediction ? "Positive" : "Negative";
    
    public string Confidence => Probability switch
    {
        > 0.9f => "Very High",
        > 0.75f => "High",
        > 0.6f => "Moderate",
        _ => "Low"
    };
}

#endregion

#region Text Preprocessing

/// <summary>
/// Text preprocessing utilities for cleaning and normalizing review text
/// </summary>
public static class TextPreprocessor
{
    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "been", "be",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "it", "its", "this", "that", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them", "their"
    };
    
    private static readonly Dictionary<string, string> Contractions = new()
    {
        { "won't", "will not" }, { "can't", "cannot" }, { "n't", " not" },
        { "'re", " are" }, { "'s", " is" }, { "'d", " would" },
        { "'ll", " will" }, { "'ve", " have" }, { "'m", " am" }
    };
    
    /// <summary>
    /// Cleans and normalizes text for sentiment analysis
    /// </summary>
    public static string CleanText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;
        
        var result = text.ToLowerInvariant();
        
        // Expand contractions - crucial for sentiment!
        // "don't like" becomes "do not like"
        foreach (var (contraction, expansion) in Contractions)
        {
            result = result.Replace(contraction, expansion);
        }
        
        // Remove HTML tags
        result = Regex.Replace(result, @"<[^>]+>", " ");
        
        // Remove URLs
        result = Regex.Replace(result, @"https?://\S+|www\.\S+", " ");
        
        // Handle emphasis markers
        result = result.Replace("!!!", " very_exclaimed ")
                       .Replace("!!", " exclaimed ");
        
        // Normalize repeated characters: "soooo good" -> "so good"
        result = Regex.Replace(result, @"(.)\1{2,}", "$1$1");
        
        // Keep basic punctuation that carries sentiment
        result = Regex.Replace(result, @"[^\w\s\!\?\-]", " ");
        
        // Normalize whitespace
        result = Regex.Replace(result, @"\s+", " ").Trim();
        
        return result;
    }
    
    /// <summary>
    /// Removes common stop words that add little semantic value
    /// </summary>
    public static string RemoveStopWords(string text)
    {
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var filtered = words.Where(w => !StopWords.Contains(w) && w.Length > 1);
        return string.Join(" ", filtered);
    }
}

#endregion

#region Sample Data

/// <summary>
/// Provides embedded sample review data for training and testing
/// </summary>
public static class SampleData
{
    /// <summary>
    /// Sample product reviews for training the sentiment model
    /// </summary>
    public static List<ReviewData> GetTrainingData() =>
    [
        // Positive reviews
        new() { ReviewText = "Absolutely love this product! Best purchase I've made all year.", Sentiment = true },
        new() { ReviewText = "Excellent quality and fast shipping. Highly recommend!", Sentiment = true },
        new() { ReviewText = "This exceeded my expectations. Works perfectly!", Sentiment = true },
        new() { ReviewText = "Great value for the price. Would definitely buy again.", Sentiment = true },
        new() { ReviewText = "Amazing product, works exactly as described. Very satisfied!", Sentiment = true },
        new() { ReviewText = "Perfect! Just what I needed. Five stars!", Sentiment = true },
        new() { ReviewText = "Incredible quality and the customer service was fantastic.", Sentiment = true },
        new() { ReviewText = "So happy with this purchase. It's even better than expected.", Sentiment = true },
        new() { ReviewText = "Wonderful product, great design, and very well made.", Sentiment = true },
        new() { ReviewText = "Outstanding! This is exactly what I was looking for.", Sentiment = true },
        new() { ReviewText = "Love it! Easy to use and works great every time.", Sentiment = true },
        new() { ReviewText = "Fantastic quality for this price point. Very impressed!", Sentiment = true },
        new() { ReviewText = "Super happy with my purchase. Delivery was quick too!", Sentiment = true },
        new() { ReviewText = "This product is amazing. I use it every day now.", Sentiment = true },
        new() { ReviewText = "Best product in its category. Totally worth the money.", Sentiment = true },
        new() { ReviewText = "Very pleased with the quality. Exceeded expectations!", Sentiment = true },
        new() { ReviewText = "Brilliant! Works perfectly and arrived early.", Sentiment = true },
        new() { ReviewText = "Absolutely fantastic. My whole family loves it.", Sentiment = true },
        new() { ReviewText = "Top quality product. Would recommend to anyone.", Sentiment = true },
        new() { ReviewText = "Great purchase! Very well designed and functional.", Sentiment = true },
        new() { ReviewText = "The camera quality is excellent, takes stunning photos.", Sentiment = true },
        new() { ReviewText = "Battery life is amazing, lasts all day easily.", Sentiment = true },
        new() { ReviewText = "Build quality is superb, feels premium and durable.", Sentiment = true },
        new() { ReviewText = "Screen display is gorgeous, colors are vibrant.", Sentiment = true },
        new() { ReviewText = "Performance is blazing fast, no lag at all.", Sentiment = true },
        
        // Negative reviews
        new() { ReviewText = "Terrible product. Complete waste of money!", Sentiment = false },
        new() { ReviewText = "Very disappointed. Broke after just one week of use.", Sentiment = false },
        new() { ReviewText = "Do not buy this! Worst purchase I've ever made.", Sentiment = false },
        new() { ReviewText = "Poor quality. Nothing like what was advertised.", Sentiment = false },
        new() { ReviewText = "Horrible experience. Product arrived damaged.", Sentiment = false },
        new() { ReviewText = "Waste of money. Does not work as described.", Sentiment = false },
        new() { ReviewText = "Very unhappy with this purchase. Returning immediately.", Sentiment = false },
        new() { ReviewText = "Awful product. Cheaply made and broke instantly.", Sentiment = false },
        new() { ReviewText = "Disappointed. Not worth the price at all.", Sentiment = false },
        new() { ReviewText = "Total junk. Save your money and buy elsewhere.", Sentiment = false },
        new() { ReviewText = "Terrible quality. Stopped working after one day.", Sentiment = false },
        new() { ReviewText = "Don't waste your money on this garbage!", Sentiment = false },
        new() { ReviewText = "Worst product ever. Completely useless.", Sentiment = false },
        new() { ReviewText = "Very poor quality. Feels cheap and flimsy.", Sentiment = false },
        new() { ReviewText = "Horrible! Not what I expected at all.", Sentiment = false },
        new() { ReviewText = "Disappointed with quality. Won't buy again.", Sentiment = false },
        new() { ReviewText = "Broke immediately. Returning for refund.", Sentiment = false },
        new() { ReviewText = "Do not recommend. Terrible customer service too.", Sentiment = false },
        new() { ReviewText = "Awful experience from start to finish.", Sentiment = false },
        new() { ReviewText = "Not worth it. Regret this purchase completely.", Sentiment = false },
        new() { ReviewText = "Battery life is terrible, barely lasts two hours.", Sentiment = false },
        new() { ReviewText = "Camera quality is awful, photos are blurry.", Sentiment = false },
        new() { ReviewText = "Build quality is poor, feels cheap and plastic.", Sentiment = false },
        new() { ReviewText = "Screen has dead pixels and colors look washed out.", Sentiment = false },
        new() { ReviewText = "So slow and laggy. Performance is disappointing.", Sentiment = false }
    ];
    
    /// <summary>
    /// Sample reviews for testing predictions
    /// </summary>
    public static string[] GetTestReviews() =>
    [
        "This is an excellent product! I love everything about it.",
        "Terrible experience. Would not recommend to anyone.",
        "Good quality for the price. Pretty satisfied overall.",
        "The product broke after two days. Very disappointed.",
        "Absolutely fantastic! Best purchase this year!",
        "Meh, it's okay. Nothing special but gets the job done.",
        "Worst product I've ever bought. Total waste of money.",
        "Super impressed with the quality. Highly recommend!",
        "Not bad, but not great either. Average product.",
        "I didn't love the battery life, but the camera is amazing!"
    ];
}

#endregion

#region Sentiment Analyzer

/// <summary>
/// Main sentiment analysis class using ML.NET with TF-IDF vectorization
/// </summary>
public class ProductReviewAnalyzer
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private PredictionEngine<ReviewData, SentimentPrediction>? _predictionEngine;
    
    public ProductReviewAnalyzer()
    {
        // Use fixed seed for reproducibility
        _mlContext = new MLContext(seed: 42);
    }
    
    /// <summary>
    /// Trains the sentiment model using TF-IDF vectorization and binary classification
    /// </summary>
    public void Train(IEnumerable<ReviewData> trainingData)
    {
        Console.WriteLine("=== Training Sentiment Analysis Model ===\n");
        
        // Load data into ML.NET
        var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);
        
        // Split into training and validation sets (80/20)
        var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 42);
        
        Console.WriteLine("Building ML pipeline with TF-IDF vectorization...\n");
        
        // Build the text featurization pipeline
        // Using FeaturizeText which combines normalization, tokenization, 
        // stop word removal, and TF-IDF n-gram extraction in one step
        var textOptions = new TextFeaturizingEstimator.Options
        {
            // Word-level features with TF-IDF weighting
            WordFeatureExtractor = new WordBagEstimator.Options
            {
                NgramLength = 2,           // Include bigrams
                UseAllLengths = true,      // Also include unigrams
                Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf
            },
            // Character-level features (helps with typos, morphology)
            CharFeatureExtractor = new WordBagEstimator.Options
            {
                NgramLength = 4,
                UseAllLengths = false,
                Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf
            },
            // Normalization options
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            KeepDiacritics = false,
            KeepPunctuations = false,
            KeepNumbers = true,
            // Include stop words removal
            StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options
            {
                Language = TextFeaturizingEstimator.Language.English
            }
        };
        
        var pipeline = _mlContext.Transforms.Text
            // Step 1-4: FeaturizeText handles normalization, tokenization, 
            // stop words, and TF-IDF n-gram extraction
            .FeaturizeText(
                outputColumnName: "Features",
                options: textOptions,
                inputColumnNames: nameof(ReviewData.ReviewText))
            // Step 5: Binary classification trainer
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features",
                maximumNumberOfIterations: 100));
        
        Console.WriteLine("Training model...\n");
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        // Train the model
        _model = pipeline.Fit(splitData.TrainSet);
        
        stopwatch.Stop();
        Console.WriteLine($"Training completed in {stopwatch.ElapsedMilliseconds}ms\n");
        
        // Evaluate on validation set
        EvaluateModel(splitData.TestSet);
        
        // Create prediction engine for single predictions
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<ReviewData, SentimentPrediction>(_model);
    }
    
    /// <summary>
    /// Evaluates model performance on test data
    /// </summary>
    private void EvaluateModel(IDataView testData)
    {
        Console.WriteLine("=== Model Evaluation ===\n");
        
        var predictions = _model!.Transform(testData);
        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");
        
        Console.WriteLine($"  Accuracy:    {metrics.Accuracy:P2}");
        Console.WriteLine($"  AUC:         {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"  F1 Score:    {metrics.F1Score:P2}");
        Console.WriteLine($"  Precision:   {metrics.PositivePrecision:P2}");
        Console.WriteLine($"  Recall:      {metrics.PositiveRecall:P2}");
        Console.WriteLine();
    }
    
    /// <summary>
    /// Predicts sentiment for a single review
    /// </summary>
    public SentimentPrediction Predict(string reviewText)
    {
        if (_predictionEngine == null)
            throw new InvalidOperationException("Model not trained. Call Train() first.");
        
        // Preprocess the text before prediction
        var cleanedText = TextPreprocessor.CleanText(reviewText);
        
        return _predictionEngine.Predict(new ReviewData { ReviewText = cleanedText });
    }
    
    /// <summary>
    /// Analyzes multiple reviews and displays results
    /// </summary>
    public void AnalyzeReviews(IEnumerable<string> reviews)
    {
        Console.WriteLine("=== Sentiment Predictions ===\n");
        Console.WriteLine($"{"Review",-60} {"Sentiment",-10} {"Confidence",-12} {"Prob",-8}");
        Console.WriteLine(new string('-', 90));
        
        foreach (var review in reviews)
        {
            var prediction = Predict(review);
            var truncatedReview = review.Length > 57 
                ? review[..54] + "..." 
                : review;
            
            Console.WriteLine($"{truncatedReview,-60} {prediction.SentimentLabel,-10} {prediction.Confidence,-12} {prediction.Probability:P1}");
        }
        
        Console.WriteLine();
    }
    
    /// <summary>
    /// Saves the trained model to disk
    /// </summary>
    public void SaveModel(string path)
    {
        if (_model == null)
            throw new InvalidOperationException("No model to save.");
        
        _mlContext.Model.Save(_model, null, path);
        Console.WriteLine($"Model saved to: {path}");
    }
    
    /// <summary>
    /// Loads a previously trained model from disk
    /// </summary>
    public void LoadModel(string path)
    {
        _model = _mlContext.Model.Load(path, out _);
        _predictionEngine = _mlContext.Model.CreatePredictionEngine<ReviewData, SentimentPrediction>(_model);
        Console.WriteLine($"Model loaded from: {path}");
    }
}

#endregion

#region Main Program

public class Program
{
    public static void Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║     Chapter 13: NLP - Product Review Sentiment Analysis      ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝\n");
        
        // Demonstrate text preprocessing
        DemonstrateTextPreprocessing();
        
        // Train and run the sentiment analyzer
        RunSentimentAnalysis();
        
        // Interactive prediction mode
        InteractivePrediction();
    }
    
    static void DemonstrateTextPreprocessing()
    {
        Console.WriteLine("=== Text Preprocessing Demo ===\n");
        
        var sampleText = "I don't LOVE this product!!! It's <b>terrible</b> and sooooo slow. Visit http://example.com for details.";
        
        Console.WriteLine($"Original:    {sampleText}");
        Console.WriteLine($"Cleaned:     {TextPreprocessor.CleanText(sampleText)}");
        Console.WriteLine($"No stops:    {TextPreprocessor.RemoveStopWords(TextPreprocessor.CleanText(sampleText))}");
        Console.WriteLine();
    }
    
    static void RunSentimentAnalysis()
    {
        // Create and train the analyzer
        var analyzer = new ProductReviewAnalyzer();
        
        // Get training data
        var trainingData = SampleData.GetTrainingData();
        Console.WriteLine($"Training with {trainingData.Count} sample reviews...\n");
        
        // Train the model
        analyzer.Train(trainingData);
        
        // Test on sample reviews
        var testReviews = SampleData.GetTestReviews();
        analyzer.AnalyzeReviews(testReviews);
    }
    
    static void InteractivePrediction()
    {
        Console.WriteLine("=== Interactive Mode ===");
        Console.WriteLine("Enter a review to analyze (or 'quit' to exit):\n");
        
        var analyzer = new ProductReviewAnalyzer();
        analyzer.Train(SampleData.GetTrainingData());
        
        while (true)
        {
            Console.Write("\n> ");
            var input = Console.ReadLine();
            
            if (string.IsNullOrWhiteSpace(input) || input.Equals("quit", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("Goodbye!");
                break;
            }
            
            var prediction = analyzer.Predict(input);
            Console.WriteLine($"\n  Sentiment:   {prediction.SentimentLabel}");
            Console.WriteLine($"  Confidence:  {prediction.Confidence} ({prediction.Probability:P1})");
            Console.WriteLine($"  Raw Score:   {prediction.Score:F4}");
        }
    }
}

#endregion
