# Chapter 12: Recommendation Systems

*"People who bought this also bought..."*

You've seen this phrase thousands of times. Whether it's Netflix suggesting your next binge-worthy series, Spotify creating a personalized playlist, or Amazon recommending products you didn't know you needed, recommendation systems have become the invisible architects of our digital experiences. These algorithms don't just suggest—they shape what we watch, listen to, buy, and discover.

For C# developers entering data science, recommendation systems represent a fascinating intersection of practical utility and mathematical elegance. They're also surprisingly accessible. By the end of this chapter, you'll understand the core algorithms behind Netflix's recommendation engine and build your own movie recommender using ML.NET.

## The Business Case for Recommendations

Before diving into algorithms, let's appreciate why companies invest billions in recommendation technology:

- **Netflix** estimates that its recommendation system saves the company $1 billion annually by reducing subscriber churn. When users find content they love, they stay.
- **Amazon** attributes 35% of its revenue to its recommendation engine. Those "frequently bought together" suggestions aren't random—they're calculated to maximize your cart value.
- **Spotify's** Discover Weekly playlist, powered by collaborative filtering, has become a cultural phenomenon, introducing users to artists they'd never find on their own.

The message is clear: good recommendations translate directly to business value. And the techniques we'll explore aren't just for tech giants—they're applicable to any domain where users interact with items, from e-commerce to content platforms to enterprise applications.

## Understanding Collaborative Filtering

Collaborative filtering is the workhorse of recommendation systems. The core insight is beautifully simple: **people who agreed in the past will likely agree in the future**. If you and I both loved *The Shawshank Redemption*, *Pulp Fiction*, and *The Dark Knight*, and I loved *Inception* but you haven't seen it, there's a good chance you'd enjoy *Inception* too.

This approach is called "collaborative" because it leverages the collective wisdom of all users to make predictions. It requires no knowledge of the items themselves—no understanding that *Inception* is a mind-bending sci-fi thriller. All it needs are patterns of user behavior.

### User-Based Collaborative Filtering

User-based collaborative filtering works exactly as our intuition suggests: find users similar to you, then recommend what they liked.

**The Algorithm:**

1. **Build a user-item matrix** where rows are users, columns are items, and cells contain ratings (or implicit feedback like views/purchases).

2. **Compute similarity between users** using metrics like cosine similarity or Pearson correlation.

3. **Find the k most similar users** to the target user (the "neighborhood").

4. **Predict ratings** by taking a weighted average of the neighbors' ratings, where weights are the similarity scores.

Let's visualize this with a simple example. Imagine a user-item matrix for movie ratings:

```
              Inception  Interstellar  The Matrix  Titanic  The Notebook
Alice             5           4            5          2          1
Bob               4           5            4          1          2
Charlie           2           1            3          5          4
Diana             ?           4            5          2          1
```

Diana hasn't rated *Inception*. To predict her rating, we:

1. Calculate similarity between Diana and all other users based on their shared ratings.
2. Alice and Bob have similar taste to Diana (high ratings for sci-fi, low for romance).
3. Weight their *Inception* ratings by similarity and average: Diana would probably rate it around 4.5.

**The Math (Intuitive Version):**

Cosine similarity treats each user's ratings as a vector and measures the angle between them:

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

Two users with identical preferences have a similarity of 1. Completely opposite preferences yield -1. The prediction formula is:

```
predicted_rating(user, item) = Σ(similarity × rating) / Σ(|similarity|)
```

**Pros and Cons:**

User-based filtering is intuitive and can capture complex, unexpected patterns. However, it struggles with scale. With millions of users, computing pairwise similarities becomes computationally expensive. It also suffers from sparsity—most users rate only a tiny fraction of available items, making similarity calculations unreliable.

### Item-Based Collaborative Filtering

Item-based collaborative filtering flips the perspective: instead of finding similar users, find similar items.

**The Insight:**

If users who liked *Inception* also tended to like *Interstellar*, these movies are similar—regardless of genre labels or descriptions. When recommending for a user, we look at items they've rated highly and find similar items they haven't seen.

**The Algorithm:**

1. **Build the same user-item matrix.**

2. **Compute similarity between items** based on how users rated them together.

3. **For a target user**, look at items they rated highly.

4. **Find similar items** to those and recommend the ones the user hasn't seen.

**Why This Often Works Better:**

Item-based filtering has a crucial practical advantage: **item similarities are more stable than user similarities**. A movie's relationship to other movies doesn't change much over time, but a user's preferences might evolve. This means we can precompute item similarities and update them periodically, rather than computing everything on-the-fly.

Amazon pioneered item-based collaborative filtering at scale in the early 2000s, and it remains the backbone of their "Customers who bought this also bought" feature.

```csharp
// Conceptual representation of item similarity calculation
public class ItemSimilarity
{
    public int ItemId1 { get; set; }
    public int ItemId2 { get; set; }
    public float Similarity { get; set; }
}

// Precomputed similarities can be stored and retrieved efficiently
var similarMovies = itemSimilarities
    .Where(s => s.ItemId1 == currentMovieId)
    .OrderByDescending(s => s.Similarity)
    .Take(10)
    .Select(s => s.ItemId2);
```

## Matrix Factorization: The Netflix Prize Revolution

In 2006, Netflix launched a competition that would transform recommendation systems forever. They offered $1 million to anyone who could beat their existing algorithm by 10%. The winning solution, submitted in 2009, relied heavily on **matrix factorization**—a technique that has since become the gold standard.

### The Core Idea: Latent Factors

Matrix factorization rests on an elegant assumption: both users and items can be described by a set of hidden (latent) factors. For movies, these factors might loosely correspond to concepts like:

- How much action vs. drama does the movie have?
- Is it mainstream or indie?
- Is it visually stunning or dialogue-driven?
- Is it a feel-good movie or emotionally heavy?

We don't explicitly define these factors—the algorithm discovers them from the data. Each user gets a vector describing their preferences across these factors, and each item gets a vector describing its characteristics.

**The prediction is simply the dot product of these vectors:**

```
predicted_rating(user, item) = user_vector · item_vector
```

If a user's vector indicates they love action and hate romance, and a movie's vector indicates it's high-action and low-romance, the dot product (predicted rating) will be high.

### Visualizing Latent Factors

Imagine we reduce everything to just two factors for visualization. We might plot movies and users in the same 2D space:

```
Factor 2 (Cerebral ↔ Fun)
    ^
    |  • Inception        • User A
    |  • Interstellar
    |                     
    |--------+-------------> Factor 1 (Serious ↔ Light)
    |        |
    |  • Schindler's List
    |                      • The Hangover
    |                      • User B
```

User A, positioned near *Inception* and *Interstellar*, would receive high predicted ratings for those films. User B, closer to *The Hangover*, has different tastes.

In practice, we use 10-200 factors, not 2. More factors capture more nuance but risk overfitting.

### The Mathematics (Accessible Version)

We start with a sparse ratings matrix **R** (users × items). Most cells are empty because users rate only a fraction of items.

Matrix factorization decomposes **R** into two smaller matrices:

```
R ≈ U × V^T
```

Where:
- **U** is a (users × k) matrix—each row is a user's latent factor vector
- **V** is a (items × k) matrix—each row is an item's latent factor vector
- **k** is the number of latent factors (a hyperparameter)

The algorithm learns **U** and **V** by minimizing the error on known ratings:

```
minimize Σ (actual_rating - predicted_rating)² + λ(regularization)
```

The regularization term (λ) prevents overfitting by penalizing large factor values. This is crucial because we're fitting a model with potentially millions of parameters.

### Training with Stochastic Gradient Descent

The most common optimization approach is stochastic gradient descent (SGD):

1. Initialize user and item factors randomly.
2. For each known rating in the training set:
   - Compute the prediction error.
   - Update the user and item factors to reduce the error.
3. Repeat until convergence.

```csharp
// Conceptual SGD update for matrix factorization
public void UpdateFactors(int userId, int itemId, float actualRating, float learningRate, float regularization)
{
    var prediction = DotProduct(userFactors[userId], itemFactors[itemId]);
    var error = actualRating - prediction;
    
    for (int f = 0; f < numFactors; f++)
    {
        var userFactor = userFactors[userId][f];
        var itemFactor = itemFactors[itemId][f];
        
        // Gradient descent updates
        userFactors[userId][f] += learningRate * (error * itemFactor - regularization * userFactor);
        itemFactors[itemId][f] += learningRate * (error * userFactor - regularization * itemFactor);
    }
}
```

### Alternating Least Squares (ALS)

An alternative to SGD is Alternating Least Squares. The insight is that if we fix the user factors, finding optimal item factors becomes a simple least squares problem (and vice versa).

ALS alternates:
1. Fix item factors, solve for optimal user factors.
2. Fix user factors, solve for optimal item factors.
3. Repeat until convergence.

ALS parallelizes beautifully because each user's factors can be updated independently once item factors are fixed. This makes it popular for distributed systems processing massive datasets.

## ML.NET's Recommendation APIs

ML.NET provides a production-ready implementation of matrix factorization through the `MatrixFactorizationTrainer`. Let's explore how to use it effectively.

### Setting Up the Environment

First, ensure you have the necessary NuGet packages:

```xml
<PackageReference Include="Microsoft.ML" Version="5.0.0" />
<PackageReference Include="Microsoft.ML.Recommender" Version="0.22.0" />
```

### Understanding the Data Model

ML.NET's matrix factorization expects data in a specific format—essentially a list of (user, item, rating) tuples:

```csharp
public class MovieRating
{
    [LoadColumn(0)]
    public float UserId { get; set; }
    
    [LoadColumn(1)]
    public float MovieId { get; set; }
    
    [LoadColumn(2)]
    public float Rating { get; set; }
}

public class MovieRatingPrediction
{
    public float Label { get; set; }
    public float Score { get; set; }
}
```

Note that ML.NET requires user and item IDs to be floats that can be converted to keys. We'll handle this in our preprocessing.

### The MatrixFactorizationTrainer

Here's how to configure and train a matrix factorization model:

```csharp
var mlContext = new MLContext(seed: 42);

// Load data
IDataView trainingData = mlContext.Data.LoadFromTextFile<MovieRating>(
    path: "ratings-train.csv",
    hasHeader: true,
    separatorChar: ',');

// Configure the trainer
var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = nameof(MovieRating.UserId),
    MatrixRowIndexColumnName = nameof(MovieRating.MovieId),
    LabelColumnName = nameof(MovieRating.Rating),
    NumberOfIterations = 100,
    ApproximationRank = 128,  // Number of latent factors
    LearningRate = 0.01,
    Quiet = false
};

// Build the pipeline
var pipeline = mlContext.Recommendation().Trainers.MatrixFactorization(options);

// Train the model
Console.WriteLine("Training the recommendation model...");
var model = pipeline.Fit(trainingData);
```

**Key Parameters Explained:**

- **ApproximationRank**: The number of latent factors (k). Higher values capture more nuance but increase training time and risk overfitting. Start with 50-100 for most applications.

- **NumberOfIterations**: How many passes through the data. More iterations generally improve accuracy until convergence, but with diminishing returns.

- **LearningRate**: Controls the step size in gradient descent. Too high causes instability; too low makes training painfully slow.

- **Lambda** (regularization): Not directly exposed in options but controlled internally. Prevents overfitting by penalizing large factor values.

### Making Predictions

Once trained, generating predictions is straightforward:

```csharp
// Create prediction engine
var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

// Predict rating for a specific user-movie pair
var testInput = new MovieRating { UserId = 6, MovieId = 10 };
var prediction = predictionEngine.Predict(testInput);

Console.WriteLine($"Predicted rating for User {testInput.UserId} on Movie {testInput.MovieId}: {prediction.Score:F2}");
```

### Generating Top-N Recommendations

For practical applications, you want to recommend the top N items for a user:

```csharp
public List<int> GetTopRecommendations(
    PredictionEngine<MovieRating, MovieRatingPrediction> predictionEngine,
    int userId,
    HashSet<int> seenMovies,
    IEnumerable<int> allMovieIds,
    int topN = 10)
{
    var recommendations = new List<(int MovieId, float Score)>();
    
    foreach (var movieId in allMovieIds)
    {
        // Skip movies the user has already seen
        if (seenMovies.Contains(movieId)) continue;
        
        var prediction = predictionEngine.Predict(new MovieRating 
        { 
            UserId = userId, 
            MovieId = movieId 
        });
        
        recommendations.Add((movieId, prediction.Score));
    }
    
    return recommendations
        .OrderByDescending(r => r.Score)
        .Take(topN)
        .Select(r => r.MovieId)
        .ToList();
}
```

## The Cold Start Problem

Every recommendation system faces its nemesis: the cold start problem. How do you recommend items to a new user with no history? How do you recommend a newly added item that no one has rated?

### Types of Cold Start

**New User Cold Start:**
A user signs up for Netflix. They haven't watched anything yet. What do you show them?

**New Item Cold Start:**
A new movie is added to the catalog. No one has rated it. How does it ever get recommended?

**System Cold Start:**
You're launching a new recommendation system with no historical data at all.

### Solutions for New Users

**1. Onboarding Questionnaires**
Ask users about their preferences upfront. Netflix famously asks new users to rate a few movies or select genres they enjoy. This creates an initial profile.

```csharp
public class UserOnboarding
{
    public List<int> FavoriteGenres { get; set; }
    public List<int> RatedMovies { get; set; }
    
    public float[] InitializeUserFactors(float[][] genreFactors)
    {
        // Average the factors of selected genres to create initial user profile
        var initialFactors = new float[genreFactors[0].Length];
        
        foreach (var genreId in FavoriteGenres)
        {
            for (int i = 0; i < initialFactors.Length; i++)
            {
                initialFactors[i] += genreFactors[genreId][i];
            }
        }
        
        // Normalize
        for (int i = 0; i < initialFactors.Length; i++)
        {
            initialFactors[i] /= FavoriteGenres.Count;
        }
        
        return initialFactors;
    }
}
```

**2. Demographic-Based Defaults**
If you know a user's age, location, or other demographics, you can start with recommendations popular among similar demographics.

**3. Popularity-Based Fallback**
When in doubt, recommend popular items. They're popular for a reason. This isn't personalized, but it's better than random.

```csharp
public List<int> GetPopularityBasedRecommendations(int topN = 10)
{
    return ratings
        .GroupBy(r => r.MovieId)
        .Select(g => new 
        { 
            MovieId = g.Key, 
            AverageRating = g.Average(r => r.Rating),
            Count = g.Count()
        })
        .Where(m => m.Count >= 100)  // Minimum rating threshold
        .OrderByDescending(m => m.AverageRating)
        .Take(topN)
        .Select(m => m.MovieId)
        .ToList();
}
```

**4. Hybrid Approaches**
Combine collaborative filtering with content-based features. Even without ratings, you can recommend items similar to what the user explicitly searched for or browsed.

### Solutions for New Items

**1. Content-Based Features**
Use item metadata (genre, director, actors, description) to place new items in the feature space. If a new movie is a Christopher Nolan sci-fi thriller, position it near *Inception* and *Interstellar*.

**2. Exploration-Exploitation Tradeoffs**
Intentionally recommend new items to a subset of users to gather ratings. This is the "multi-armed bandit" approach—balancing showing users what you know they'll like (exploitation) with trying new things to learn (exploration).

**3. Editorial Curation**
For high-profile new items, human curators can seed recommendations. Netflix promotes new releases on the homepage regardless of personalization.

## Evaluation Metrics

How do you know if your recommendation system is any good? Several metrics capture different aspects of quality.

### Root Mean Square Error (RMSE)

RMSE measures prediction accuracy—how close your predicted ratings are to actual ratings.

```
RMSE = √(Σ(actual - predicted)² / n)
```

Lower is better. An RMSE of 0.9 on a 1-5 rating scale means your predictions are typically within about 1 star of actual ratings.

```csharp
public double CalculateRMSE(
    PredictionEngine<MovieRating, MovieRatingPrediction> predictionEngine,
    IEnumerable<MovieRating> testData)
{
    var squaredErrors = new List<double>();
    
    foreach (var rating in testData)
    {
        var prediction = predictionEngine.Predict(rating);
        var error = rating.Rating - prediction.Score;
        squaredErrors.Add(error * error);
    }
    
    return Math.Sqrt(squaredErrors.Average());
}
```

**Limitations of RMSE:**
RMSE treats all predictions equally, but users care more about whether the top recommendations are good. Accurately predicting that a user will rate a terrible movie 1.5 instead of 1.0 doesn't help anyone.

### Precision@K

Precision@K asks: *Of the top K recommendations, how many were actually relevant?*

"Relevant" typically means the user rated the item above some threshold (e.g., 4+ stars) or interacted with it (clicked, purchased, watched).

```
Precision@K = (relevant items in top K) / K
```

```csharp
public double CalculatePrecisionAtK(
    List<int> recommendations,
    HashSet<int> relevantItems,
    int k)
{
    var topK = recommendations.Take(k).ToList();
    var hits = topK.Count(item => relevantItems.Contains(item));
    return (double)hits / k;
}
```

### Recall@K

Recall@K asks: *Of all the relevant items, how many appeared in the top K?*

```
Recall@K = (relevant items in top K) / (total relevant items)
```

```csharp
public double CalculateRecallAtK(
    List<int> recommendations,
    HashSet<int> relevantItems,
    int k)
{
    var topK = recommendations.Take(k).ToList();
    var hits = topK.Count(item => relevantItems.Contains(item));
    return (double)hits / relevantItems.Count;
}
```

### The Precision-Recall Tradeoff

There's an inherent tension between precision and recall:
- **High K**: Better recall (you're showing more items, so you're likely to hit more relevant ones), but worse precision (more noise in the list).
- **Low K**: Better precision (only your most confident recommendations), but worse recall (you might miss relevant items).

In practice, you choose K based on your UI. If you're showing 10 recommendations on a homepage, you care about Precision@10 and Recall@10.

### Evaluating with ML.NET

ML.NET provides built-in evaluation for regression metrics:

```csharp
// Evaluate on test set
var testData = mlContext.Data.LoadFromTextFile<MovieRating>(
    path: "ratings-test.csv",
    hasHeader: true,
    separatorChar: ',');

var predictions = model.Transform(testData);
var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(MovieRating.Rating));

Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F4}");
Console.WriteLine($"MAE: {metrics.MeanAbsoluteError:F4}");
Console.WriteLine($"R²: {metrics.RSquared:F4}");
```

For ranking metrics like Precision@K and Recall@K, you'll need custom evaluation code as shown above.

## Project: Building a Movie Recommendation Engine

Let's put everything together and build a complete movie recommendation system using the MovieLens dataset.

### Project Overview

We'll build a console application that:
1. Loads and preprocesses MovieLens data
2. Trains a matrix factorization model
3. Evaluates model performance
4. Generates personalized recommendations
5. Handles cold start for new users

### Step 1: Project Setup

Create a new console application and add dependencies:

```bash
dotnet new console -n MovieRecommender
cd MovieRecommender
dotnet add package Microsoft.ML
dotnet add package Microsoft.ML.Recommender
dotnet add package CsvHelper
```

### Step 2: Download the MovieLens Dataset

Download the MovieLens 100K or 1M dataset from https://grouplens.org/datasets/movielens/. We'll use the 100K dataset for faster iteration.

### Step 3: Define Data Models

```csharp
using Microsoft.ML.Data;

namespace MovieRecommender.Models
{
    public class MovieRating
    {
        [LoadColumn(0)]
        public float UserId { get; set; }
        
        [LoadColumn(1)]
        public float MovieId { get; set; }
        
        [LoadColumn(2)]
        public float Rating { get; set; }
        
        [LoadColumn(3)]
        public float Timestamp { get; set; }
    }
    
    public class MovieRatingPrediction
    {
        public float Label { get; set; }
        public float Score { get; set; }
    }
    
    public class Movie
    {
        public int MovieId { get; set; }
        public string Title { get; set; }
        public string Genres { get; set; }
    }
}
```

### Step 4: Create the Recommendation Service

```csharp
using Microsoft.ML;
using Microsoft.ML.Trainers;
using MovieRecommender.Models;

namespace MovieRecommender.Services
{
    public class RecommendationService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private PredictionEngine<MovieRating, MovieRatingPrediction> _predictionEngine;
        private Dictionary<int, Movie> _movies;
        private Dictionary<int, HashSet<int>> _userRatedMovies;
        private HashSet<int> _allMovieIds;
        
        public RecommendationService()
        {
            _mlContext = new MLContext(seed: 42);
            _movies = new Dictionary<int, Movie>();
            _userRatedMovies = new Dictionary<int, HashSet<int>>();
            _allMovieIds = new HashSet<int>();
        }
        
        public void LoadMovies(string moviesPath)
        {
            Console.WriteLine("Loading movie metadata...");
            
            using var reader = new StreamReader(moviesPath);
            // Skip header
            reader.ReadLine();
            
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = ParseCsvLine(line);
                
                if (values.Length >= 3)
                {
                    var movie = new Movie
                    {
                        MovieId = int.Parse(values[0]),
                        Title = values[1],
                        Genres = values[2]
                    };
                    _movies[movie.MovieId] = movie;
                    _allMovieIds.Add(movie.MovieId);
                }
            }
            
            Console.WriteLine($"Loaded {_movies.Count} movies.");
        }
        
        public void TrainModel(string ratingsPath, float testFraction = 0.2f)
        {
            Console.WriteLine("Loading ratings data...");
            
            var data = _mlContext.Data.LoadFromTextFile<MovieRating>(
                path: ratingsPath,
                hasHeader: true,
                separatorChar: ',');
            
            // Build user-movie mapping for filtering recommendations
            var ratings = _mlContext.Data.CreateEnumerable<MovieRating>(data, reuseRowObject: false).ToList();
            foreach (var rating in ratings)
            {
                var userId = (int)rating.UserId;
                var movieId = (int)rating.MovieId;
                
                if (!_userRatedMovies.ContainsKey(userId))
                    _userRatedMovies[userId] = new HashSet<int>();
                    
                _userRatedMovies[userId].Add(movieId);
            }
            
            Console.WriteLine($"Loaded {ratings.Count} ratings from {_userRatedMovies.Count} users.");
            
            // Split data
            var split = _mlContext.Data.TrainTestSplit(data, testFraction: testFraction);
            
            // Configure trainer
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = nameof(MovieRating.UserId),
                MatrixRowIndexColumnName = nameof(MovieRating.MovieId),
                LabelColumnName = nameof(MovieRating.Rating),
                NumberOfIterations = 100,
                ApproximationRank = 64,
                LearningRate = 0.1
            };
            
            var pipeline = _mlContext.Recommendation().Trainers.MatrixFactorization(options);
            
            // Train
            Console.WriteLine("Training model (this may take a few minutes)...");
            var watch = System.Diagnostics.Stopwatch.StartNew();
            _model = pipeline.Fit(split.TrainSet);
            watch.Stop();
            Console.WriteLine($"Training completed in {watch.Elapsed.TotalSeconds:F1} seconds.");
            
            // Evaluate
            Console.WriteLine("\nEvaluating model...");
            var predictions = _model.Transform(split.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(MovieRating.Rating));
            
            Console.WriteLine($"  RMSE: {metrics.RootMeanSquaredError:F4}");
            Console.WriteLine($"  MAE:  {metrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"  R²:   {metrics.RSquared:F4}");
            
            // Create prediction engine
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(_model);
        }
        
        public List<(Movie Movie, float PredictedRating)> GetRecommendations(int userId, int topN = 10)
        {
            if (_predictionEngine == null)
                throw new InvalidOperationException("Model not trained. Call TrainModel first.");
            
            var seenMovies = _userRatedMovies.ContainsKey(userId) 
                ? _userRatedMovies[userId] 
                : new HashSet<int>();
            
            var recommendations = new List<(int MovieId, float Score)>();
            
            foreach (var movieId in _allMovieIds)
            {
                if (seenMovies.Contains(movieId)) continue;
                
                var prediction = _predictionEngine.Predict(new MovieRating
                {
                    UserId = userId,
                    MovieId = movieId
                });
                
                // Clamp predictions to valid rating range
                var score = Math.Max(1, Math.Min(5, prediction.Score));
                recommendations.Add((movieId, score));
            }
            
            return recommendations
                .OrderByDescending(r => r.Score)
                .Take(topN)
                .Select(r => (_movies[r.MovieId], r.Score))
                .ToList();
        }
        
        public List<(Movie Movie, float AverageRating)> GetPopularMovies(int topN = 10)
        {
            // Fallback for cold start - return popular movies
            var movieStats = new Dictionary<int, (float Sum, int Count)>();
            
            foreach (var (userId, movies) in _userRatedMovies)
            {
                // We don't have ratings stored here, so this is a simplified version
                // In production, you'd store rating values too
                foreach (var movieId in movies)
                {
                    if (!movieStats.ContainsKey(movieId))
                        movieStats[movieId] = (0, 0);
                    
                    var (sum, count) = movieStats[movieId];
                    movieStats[movieId] = (sum, count + 1);
                }
            }
            
            return movieStats
                .OrderByDescending(m => m.Value.Count)
                .Take(topN)
                .Where(m => _movies.ContainsKey(m.Key))
                .Select(m => (_movies[m.Key], (float)m.Value.Count))
                .ToList();
        }
        
        public List<(Movie Movie, float PredictedRating)> GetRecommendationsForNewUser(
            List<(int MovieId, float Rating)> initialRatings,
            int topN = 10)
        {
            // Simple approach: find users with similar ratings and aggregate their preferences
            // In production, you'd retrain the model incrementally or use a hybrid approach
            
            if (initialRatings.Count == 0)
            {
                Console.WriteLine("No initial ratings provided. Returning popular movies.");
                return GetPopularMovies(topN)
                    .Select(m => (m.Movie, m.AverageRating))
                    .ToList();
            }
            
            // Find similar users based on initial ratings overlap
            var similarUsers = new List<(int UserId, double Similarity)>();
            
            foreach (var (userId, ratedMovies) in _userRatedMovies)
            {
                var overlap = initialRatings.Count(r => ratedMovies.Contains(r.MovieId));
                if (overlap > 0)
                {
                    var similarity = (double)overlap / Math.Max(initialRatings.Count, ratedMovies.Count);
                    similarUsers.Add((userId, similarity));
                }
            }
            
            // Use top 5 similar users to generate recommendations
            var topSimilarUsers = similarUsers
                .OrderByDescending(u => u.Similarity)
                .Take(5)
                .ToList();
            
            if (topSimilarUsers.Count == 0)
            {
                Console.WriteLine("No similar users found. Returning popular movies.");
                return GetPopularMovies(topN)
                    .Select(m => (m.Movie, m.AverageRating))
                    .ToList();
            }
            
            // Aggregate recommendations from similar users
            var movieScores = new Dictionary<int, List<float>>();
            var ratedMovieIds = initialRatings.Select(r => r.MovieId).ToHashSet();
            
            foreach (var (userId, similarity) in topSimilarUsers)
            {
                var userRecs = GetRecommendations(userId, 50);
                foreach (var (movie, score) in userRecs)
                {
                    if (ratedMovieIds.Contains(movie.MovieId)) continue;
                    
                    if (!movieScores.ContainsKey(movie.MovieId))
                        movieScores[movie.MovieId] = new List<float>();
                    
                    movieScores[movie.MovieId].Add(score * (float)similarity);
                }
            }
            
            return movieScores
                .Select(m => (_movies[m.Key], m.Value.Average()))
                .OrderByDescending(m => m.Item2)
                .Take(topN)
                .ToList();
        }
        
        public void SaveModel(string modelPath)
        {
            _mlContext.Model.Save(_model, null, modelPath);
            Console.WriteLine($"Model saved to {modelPath}");
        }
        
        public void LoadModel(string modelPath)
        {
            _model = _mlContext.Model.Load(modelPath, out _);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(_model);
            Console.WriteLine($"Model loaded from {modelPath}");
        }
        
        private string[] ParseCsvLine(string line)
        {
            // Simple CSV parser handling quoted fields
            var result = new List<string>();
            var current = new System.Text.StringBuilder();
            var inQuotes = false;
            
            foreach (var c in line)
            {
                if (c == '"')
                {
                    inQuotes = !inQuotes;
                }
                else if (c == ',' && !inQuotes)
                {
                    result.Add(current.ToString());
                    current.Clear();
                }
                else
                {
                    current.Append(c);
                }
            }
            result.Add(current.ToString());
            
            return result.ToArray();
        }
    }
}
```

### Step 5: Create the Main Application

```csharp
using MovieRecommender.Services;

namespace MovieRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("=== Movie Recommendation Engine ===\n");
            
            var service = new RecommendationService();
            
            // Paths to MovieLens data (adjust as needed)
            var moviesPath = "data/movies.csv";
            var ratingsPath = "data/ratings.csv";
            
            // Load data and train model
            service.LoadMovies(moviesPath);
            service.TrainModel(ratingsPath);
            
            // Interactive mode
            while (true)
            {
                Console.WriteLine("\n--- Options ---");
                Console.WriteLine("1. Get recommendations for existing user");
                Console.WriteLine("2. Get recommendations for new user (cold start)");
                Console.WriteLine("3. Show popular movies");
                Console.WriteLine("4. Save model");
                Console.WriteLine("5. Exit");
                Console.Write("\nChoice: ");
                
                var choice = Console.ReadLine()?.Trim();
                
                switch (choice)
                {
                    case "1":
                        GetExistingUserRecommendations(service);
                        break;
                    case "2":
                        GetNewUserRecommendations(service);
                        break;
                    case "3":
                        ShowPopularMovies(service);
                        break;
                    case "4":
                        service.SaveModel("movie_recommender.zip");
                        break;
                    case "5":
                        return;
                    default:
                        Console.WriteLine("Invalid choice.");
                        break;
                }
            }
        }
        
        static void GetExistingUserRecommendations(RecommendationService service)
        {
            Console.Write("Enter user ID: ");
            if (int.TryParse(Console.ReadLine(), out int userId))
            {
                Console.WriteLine($"\nTop 10 recommendations for User {userId}:");
                Console.WriteLine(new string('-', 60));
                
                var recommendations = service.GetRecommendations(userId, 10);
                
                for (int i = 0; i < recommendations.Count; i++)
                {
                    var (movie, score) = recommendations[i];
                    Console.WriteLine($"{i + 1,2}. {movie.Title,-45} ({score:F2})");
                }
            }
            else
            {
                Console.WriteLine("Invalid user ID.");
            }
        }
        
        static void GetNewUserRecommendations(RecommendationService service)
        {
            Console.WriteLine("\nLet's get to know your taste! Rate some movies (1-5 stars):");
            Console.WriteLine("Enter 'done' when finished.\n");
            
            var initialRatings = new List<(int MovieId, float Rating)>();
            
            // Sample movies to rate (in production, show popular diverse movies)
            var moviesToRate = new (int Id, string Title)[]
            {
                (1, "Toy Story (1995)"),
                (260, "Star Wars: Episode IV (1977)"),
                (480, "Jurassic Park (1993)"),
                (527, "Schindler's List (1993)"),
                (593, "Silence of the Lambs (1991)"),
                (318, "Shawshank Redemption (1994)"),
                (356, "Forrest Gump (1994)"),
                (296, "Pulp Fiction (1994)")
            };
            
            foreach (var (id, title) in moviesToRate)
            {
                Console.Write($"  {title}: ");
                var input = Console.ReadLine()?.Trim().ToLower();
                
                if (input == "done") break;
                if (input == "skip" || string.IsNullOrEmpty(input)) continue;
                
                if (float.TryParse(input, out float rating) && rating >= 1 && rating <= 5)
                {
                    initialRatings.Add((id, rating));
                }
            }
            
            Console.WriteLine($"\nBased on {initialRatings.Count} ratings, here are your recommendations:");
            Console.WriteLine(new string('-', 60));
            
            var recommendations = service.GetRecommendationsForNewUser(initialRatings, 10);
            
            for (int i = 0; i < recommendations.Count; i++)
            {
                var (movie, score) = recommendations[i];
                Console.WriteLine($"{i + 1,2}. {movie.Title,-45} ({score:F2})");
            }
        }
        
        static void ShowPopularMovies(RecommendationService service)
        {
            Console.WriteLine("\nMost popular movies:");
            Console.WriteLine(new string('-', 60));
            
            var popular = service.GetPopularMovies(10);
            
            for (int i = 0; i < popular.Count; i++)
            {
                var (movie, count) = popular[i];
                Console.WriteLine($"{i + 1,2}. {movie.Title,-45} ({count:F0} ratings)");
            }
        }
    }
}
```

### Running the Project

```bash
# Ensure MovieLens data is in the data folder
# ratings.csv format: userId,movieId,rating,timestamp
# movies.csv format: movieId,title,genres

dotnet run
```

### Expected Output

```
=== Movie Recommendation Engine ===

Loading movie metadata...
Loaded 9742 movies.
Loading ratings data...
Loaded 100836 ratings from 610 users.
Training model (this may take a few minutes)...
Training completed in 12.3 seconds.

Evaluating model...
  RMSE: 0.8721
  MAE:  0.6634
  R²:   0.3156

--- Options ---
1. Get recommendations for existing user
2. Get recommendations for new user (cold start)
3. Show popular movies
4. Save model
5. Exit

Choice: 1
Enter user ID: 1

Top 10 recommendations for User 1:
------------------------------------------------------------
 1. Shawshank Redemption, The (1994)              (4.78)
 2. Godfather, The (1972)                         (4.65)
 3. Usual Suspects, The (1995)                    (4.54)
 4. Schindler's List (1993)                       (4.52)
 5. Casablanca (1942)                             (4.48)
 6. One Flew Over the Cuckoo's Nest (1975)        (4.47)
 7. Goodfellas (1990)                             (4.44)
 8. Rear Window (1954)                            (4.41)
 9. Raiders of the Lost Ark (1981)                (4.39)
10. Seven Samurai (1954)                          (4.38)
```

## Advanced Techniques and Considerations

Before discussing scale, let's explore some refinements that can significantly improve recommendation quality.

### Implicit vs. Explicit Feedback

Our examples use explicit feedback—users actively rating items on a 1-5 scale. But in many applications, you only have implicit feedback: views, clicks, purchases, time spent, or skip behavior.

Implicit feedback is:
- **More abundant**: Every user generates it, whether they rate or not
- **Noisier**: A view doesn't necessarily mean a positive preference
- **Non-negative**: You observe positive signals (watched) but absence doesn't mean dislike (maybe they didn't see it)

ML.NET supports implicit feedback through appropriate loss functions:

```csharp
// For implicit feedback, use a different loss function
var options = new MatrixFactorizationTrainer.Options
{
    MatrixColumnIndexColumnName = nameof(MovieRating.UserId),
    MatrixRowIndexColumnName = nameof(MovieRating.MovieId),
    LabelColumnName = nameof(MovieRating.Rating),
    LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass,
    NumberOfIterations = 50,
    ApproximationRank = 32
};
```

### Temporal Dynamics

User preferences change over time. A user who loved romantic comedies five years ago might now prefer documentaries. Similarly, movies can drift in perception—cult classics gain appreciation; once-popular films feel dated.

Sophisticated systems incorporate time:
- **Time-weighted ratings**: Recent ratings count more
- **Temporal factors**: User and item factors that evolve over time
- **Session modeling**: Short-term preferences within a viewing session

```csharp
// Weight recent ratings more heavily
public float GetTimeWeightedRating(float rating, DateTime ratingDate)
{
    var daysAgo = (DateTime.Now - ratingDate).TotalDays;
    var decay = Math.Exp(-daysAgo / 365.0); // Decay over ~1 year
    return (float)(rating * decay);
}
```

### Handling Biases

Users and items have inherent biases that affect ratings:
- **User bias**: Some users rate everything high; others are harsh critics
- **Item bias**: Some movies are universally loved; others polarizing
- **Global bias**: The average rating across all user-item pairs

A more sophisticated prediction accounts for these:

```
prediction(u, i) = global_avg + user_bias[u] + item_bias[i] + user_factors[u] · item_factors[i]
```

This is called **biased matrix factorization** and typically improves RMSE by 5-10%.

```csharp
public class BiasedPrediction
{
    private float _globalMean;
    private Dictionary<int, float> _userBiases;
    private Dictionary<int, float> _itemBiases;
    private float[][] _userFactors;
    private float[][] _itemFactors;
    
    public float Predict(int userId, int itemId)
    {
        var interaction = DotProduct(_userFactors[userId], _itemFactors[itemId]);
        var userBias = _userBiases.GetValueOrDefault(userId, 0f);
        var itemBias = _itemBiases.GetValueOrDefault(itemId, 0f);
        
        return _globalMean + userBias + itemBias + interaction;
    }
}
```

### Hybrid Recommendations

Pure collaborative filtering ignores valuable item metadata. A hybrid system combines:

**Content-based features:**
- Movie: genres, director, actors, plot keywords, release year
- Products: category, brand, description embeddings, price range
- Music: artist, genre, tempo, energy, mood

**Collaborative signals:**
- Rating patterns
- Co-occurrence statistics
- User similarity

Hybrid systems particularly help with cold start—a new Christopher Nolan sci-fi film can leverage content features even without ratings.

```csharp
public class HybridRecommender
{
    private readonly RecommendationService _collaborativeFilter;
    private readonly ContentBasedService _contentFilter;
    
    public List<Movie> GetRecommendations(int userId, int topN = 10)
    {
        // Get candidates from both systems
        var cfRecommendations = _collaborativeFilter.GetRecommendations(userId, topN * 2);
        var cbRecommendations = _contentFilter.GetRecommendations(userId, topN * 2);
        
        // Blend scores (weighted average)
        var combinedScores = new Dictionary<int, float>();
        
        foreach (var (movie, score) in cfRecommendations)
        {
            combinedScores[movie.MovieId] = score * 0.7f;  // 70% collaborative
        }
        
        foreach (var (movie, score) in cbRecommendations)
        {
            if (combinedScores.ContainsKey(movie.MovieId))
                combinedScores[movie.MovieId] += score * 0.3f;  // 30% content-based
            else
                combinedScores[movie.MovieId] = score * 0.3f;
        }
        
        return combinedScores
            .OrderByDescending(kv => kv.Value)
            .Take(topN)
            .Select(kv => GetMovie(kv.Key))
            .ToList();
    }
}
```

### Diversity and Serendipity

A recommendation system optimizing purely for predicted ratings can become a filter bubble—showing users only what they've already seen variations of. Users appreciate:

**Diversity**: Recommendations spanning different genres, styles, and topics within a single list

**Serendipity**: Surprising recommendations that users wouldn't have found themselves but end up loving

Techniques to improve diversity:
- **Maximum Marginal Relevance (MMR)**: Penalize items similar to already-selected recommendations
- **Category constraints**: Ensure top-10 list spans at least 3 genres
- **Exploration bonuses**: Occasionally surface items outside the user's comfort zone

```csharp
public List<Movie> DiversifyRecommendations(List<(Movie Movie, float Score)> candidates, int topN)
{
    var selected = new List<Movie>();
    var selectedGenres = new HashSet<string>();
    
    // Greedy selection with diversity penalty
    while (selected.Count < topN && candidates.Count > 0)
    {
        // Find best candidate considering diversity
        var best = candidates
            .Select(c => new
            {
                Candidate = c,
                DiversityBonus = c.Movie.Genres.Split('|')
                    .Count(g => !selectedGenres.Contains(g)) * 0.1f
            })
            .OrderByDescending(x => x.Candidate.Score + x.DiversityBonus)
            .First();
        
        selected.Add(best.Candidate.Movie);
        foreach (var genre in best.Candidate.Movie.Genres.Split('|'))
            selectedGenres.Add(genre);
            
        candidates.Remove(best.Candidate);
    }
    
    return selected;
}
```

## Scaling Recommendations

Our MovieLens example works well for 100,000 ratings, but what about Netflix's billions? Scaling recommendation systems requires careful architecture:

### Offline vs. Online Computation

**Offline (batch processing):**
- Train models periodically (daily/weekly)
- Precompute recommendations for all users
- Store results in fast key-value stores (Redis, DynamoDB)

**Online (real-time):**
- Update recommendations based on recent activity
- Use lightweight models for instant feedback
- Blend offline recommendations with real-time signals

### Approximate Nearest Neighbors

For item-based filtering at scale, finding similar items requires efficient similarity search. Libraries like FAISS (Facebook AI Similarity Search) and Annoy (Spotify) provide approximate nearest neighbor algorithms that trade perfect accuracy for massive speed improvements.

### Distributed Training

For datasets that don't fit in memory:
- Apache Spark MLlib provides distributed ALS implementation
- Cloud services (AWS SageMaker, Azure ML) offer managed distributed training
- Consider dimensionality reduction techniques to shrink the problem

### Real-World Architecture

```
[User Request]
      ↓
[API Gateway]
      ↓
[Recommendation Service]
      ├── Cache Layer (Redis) ← Precomputed recommendations
      ├── Model Server ← Real-time predictions
      └── Feature Store ← User/item features
      ↓
[Ranking Service] ← Blend multiple signals
      ↓
[Response]
```

## Summary

Recommendation systems are where data science delivers undeniable business value. In this chapter, you've learned:

- **Collaborative filtering** leverages collective user behavior—user-based finds similar users, item-based finds similar items
- **Matrix factorization** discovers latent factors that explain user preferences and item characteristics, enabling compact and accurate models
- **ML.NET's MatrixFactorizationTrainer** provides a production-ready implementation with sensible defaults
- **The cold start problem** requires creative solutions: onboarding questionnaires, popularity fallbacks, and hybrid approaches
- **Evaluation metrics** capture different aspects of quality—RMSE for prediction accuracy, Precision@K and Recall@K for ranking quality

You've also built a complete movie recommendation engine that handles real data, generates personalized suggestions, and addresses cold start scenarios.

The techniques in this chapter scale from hobby projects to production systems serving millions. Whether you're building a product recommendation feature for an e-commerce site, a content discovery system for a media platform, or an internal tool that suggests relevant documents, collaborative filtering and matrix factorization provide a solid foundation.

In the next chapter, we'll explore a different kind of learning—deep learning with neural networks—and see how modern architectures can capture even more complex patterns in data.

---

**Chapter Summary:**
- Collaborative filtering uses crowd wisdom to predict preferences
- User-based filtering finds similar users; item-based finds similar items
- Matrix factorization decomposes ratings into user and item latent factors
- ML.NET's MatrixFactorizationTrainer implements scalable matrix factorization
- Cold start requires fallbacks: popularity, demographics, onboarding
- RMSE measures prediction accuracy; Precision@K and Recall@K measure ranking quality
- Production systems blend offline batch computation with online real-time updates
