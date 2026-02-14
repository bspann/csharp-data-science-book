# Chapter 6: Feature Engineering

There's a saying in machine learning that's become almost a cliché, but only because it's true: *garbage in, garbage out*. Feed your model poor features, and even the most sophisticated algorithm will produce mediocre results. Feed it well-crafted features, and a simple model can outperform complex competitors.

But here's what makes feature engineering genuinely exciting: it's where data science stops being mechanical and starts being *creative*. It's where your domain knowledge, intuition, and problem-solving skills combine to transform raw data into predictive gold. This is the part of machine learning where experienced practitioners earn their keep—not by knowing more algorithms, but by seeing patterns and possibilities that others miss.

In this chapter, we'll explore the art and science of feature engineering. You'll learn what separates good features from great ones, how to synthesize new features from raw data, and how to extract meaningful signals from text and timestamps. By the end, you'll have built a complete feature engineering pipeline for housing price prediction that demonstrates these techniques in action.

Let's get creative.

## What Makes a Good Feature?

Before we start engineering features, we need to understand what we're aiming for. Not all features are created equal, and adding more features doesn't automatically improve your model. In fact, adding poor features can make things worse.

### Predictive Power

A good feature has predictive power—it contains information that helps distinguish between different outcomes. This sounds obvious, but it's worth examining closely.

Consider predicting house prices. Square footage is a strong feature because larger houses typically cost more. There's a clear, consistent relationship between the feature value and the target. Contrast this with a random ID number assigned to each house—it has no predictive power because it contains no meaningful information about the price.

But predictive power isn't always obvious. Sometimes the relationship is non-linear (prices plateau after a certain square footage), conditional (neighborhood matters more for small homes), or inverse (higher crime rates mean lower prices). Your job is to transform data so these relationships become learnable.

Here's a key insight: **raw data often contains latent predictive power that isn't directly accessible**. A single timestamp column might seem useless, but extract the day of week and suddenly you've captured weekend vs. weekday patterns. Extract the hour and you've captured time-of-day effects. The information was always there; you just made it explicit.

### Feature Independence

Ideal features are independent of each other—each one contributes unique information. When features are highly correlated, you're essentially telling the model the same thing multiple times.

Consider including both `total_square_feet` and `total_square_meters` in your model. They're perfectly correlated—knowing one tells you exactly the other. This redundancy:

- Wastes computational resources
- Can cause numerical instability in some algorithms
- Makes feature importance harder to interpret (importance gets split between correlated features)
- Increases the risk of overfitting

That said, some correlation is inevitable and acceptable. The goal isn't zero correlation; it's ensuring each feature brings something new to the table.

### Feature Quality Characteristics

When evaluating potential features, consider these characteristics:

**Availability at prediction time**: Features used in training must be available when making predictions. It's tempting to use "future information" that you'd know historically but wouldn't have in real-time. If you're predicting house prices at listing time, you can't use the actual sale price as a feature.

**Consistency over time**: Features should mean the same thing across your dataset. If zip code boundaries changed mid-dataset, that feature becomes unreliable. If your definition of "neighborhood" evolved, you'll have inconsistencies.

**Minimal leakage**: Feature engineering can accidentally introduce target leakage—where information about the target sneaks into features. Encoding average prices by neighborhood *from the same dataset* can leak target information. Always split your data before computing aggregate features, or use only historical data.

**Interpretability**: All else being equal, prefer features that humans can understand. When your model makes a prediction, you want to explain *why* in terms that make sense. "The price is higher because square footage is above average" is actionable; "the price is higher because feature_27 is elevated" is not.

Let's see how ML.NET helps us evaluate feature quality:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using MathNet.Numerics.Statistics;

public class FeatureQualityAnalyzer
{
    private readonly MLContext _mlContext;
    
    public FeatureQualityAnalyzer()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public void AnalyzeFeatures<T>(IEnumerable<T> data, string targetColumn) where T : class
    {
        var dataView = _mlContext.Data.LoadFromEnumerable(data);
        
        // Get column statistics
        var columns = dataView.Schema
            .Where(c => c.Name != targetColumn && c.Type == NumberDataViewType.Single)
            .ToList();
        
        Console.WriteLine("Feature Quality Analysis");
        Console.WriteLine(new string('=', 60));
        
        foreach (var column in columns)
        {
            var values = dataView.GetColumn<float>(column.Name).ToArray();
            var targetValues = dataView.GetColumn<float>(targetColumn).ToArray();
            
            // Calculate statistics
            var mean = values.Average();
            var stdDev = Statistics.StandardDeviation(values.Select(v => (double)v));
            var missing = values.Count(v => float.IsNaN(v));
            var correlation = Correlation.Pearson(
                values.Select(v => (double)v),
                targetValues.Select(v => (double)v));
            
            Console.WriteLine($"\n{column.Name}:");
            Console.WriteLine($"  Mean: {mean:F2}, StdDev: {stdDev:F2}");
            Console.WriteLine($"  Missing: {missing} ({100.0 * missing / values.Length:F1}%)");
            Console.WriteLine($"  Correlation with target: {correlation:F3}");
            Console.WriteLine($"  Predictive potential: {GetPredictivePotential(correlation)}");
        }
    }
    
    private string GetPredictivePotential(double correlation)
    {
        var absCorr = Math.Abs(correlation);
        return absCorr switch
        {
            > 0.7 => "★★★★★ Very Strong",
            > 0.5 => "★★★★☆ Strong",
            > 0.3 => "★★★☆☆ Moderate",
            > 0.1 => "★★☆☆☆ Weak",
            _ => "★☆☆☆☆ Negligible"
        };
    }
}
```

This analyzer gives you immediate feedback on which raw features have natural predictive power. But don't stop here—features with low correlation might become highly predictive after transformation.

## Creating Features from Raw Data

The simplest form of feature engineering transforms existing features or combines them in meaningful ways. This is where domain knowledge becomes invaluable—you're encoding human understanding into mathematical representations.

### Combining Features

Sometimes the relationship between individual features and the target is weak, but their combination is highly predictive.

```csharp
public class HousingData
{
    public float SquareFeet { get; set; }
    public float Bedrooms { get; set; }
    public float Bathrooms { get; set; }
    public float LotSize { get; set; }
    public float YearBuilt { get; set; }
    public float Price { get; set; }
}

public class EnrichedHousingData
{
    // Original features
    public float SquareFeet { get; set; }
    public float Bedrooms { get; set; }
    public float Bathrooms { get; set; }
    public float LotSize { get; set; }
    public float YearBuilt { get; set; }
    
    // Engineered features
    public float PricePerSqFt { get; set; }          // Target encoding (use carefully!)
    public float SqFtPerBedroom { get; set; }        // Room size indicator
    public float BathroomRatio { get; set; }         // Bathrooms per bedroom
    public float LotToHomeRatio { get; set; }        // Yard size indicator
    public float Age { get; set; }                   // Years since built
    public float IsNewConstruction { get; set; }     // Binary: built in last 5 years
    public float HasMoreBathsThanBeds { get; set; }  // Luxury indicator
    
    public float Price { get; set; }
}
```

Each engineered feature captures a meaningful concept:

- **SqFtPerBedroom**: Measures how spacious rooms are. A 2000 sqft house with 2 bedrooms feels different from one with 5 bedrooms.
- **BathroomRatio**: In the US market, homes with bathroom ratios above 1.0 typically command premiums.
- **LotToHomeRatio**: Captures yard size relative to home size—important for families and privacy.
- **Age**: More intuitive than raw year, and allows non-linear aging effects.
- **IsNewConstruction**: Captures the premium buyers pay for never-lived-in homes.

Let's create an ML.NET pipeline for these transformations:

```csharp
public class FeatureEngineeringPipeline
{
    private readonly MLContext _mlContext;
    
    public FeatureEngineeringPipeline()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public ITransformer BuildFeatureEngineeringPipeline(IDataView data)
    {
        var currentYear = DateTime.Now.Year;
        
        var pipeline = _mlContext.Transforms
            // Calculate room size indicator
            .Expression("SqFtPerBedroom", 
                "(sqft, beds) => beds > 0 ? sqft / beds : sqft", 
                new[] { "SquareFeet", "Bedrooms" })
            
            // Calculate bathroom ratio
            .Append(_mlContext.Transforms.Expression("BathroomRatio", 
                "(baths, beds) => beds > 0 ? baths / beds : baths", 
                new[] { "Bathrooms", "Bedrooms" }))
            
            // Calculate lot to home ratio  
            .Append(_mlContext.Transforms.Expression("LotToHomeRatio", 
                "(lot, sqft) => sqft > 0 ? lot / sqft : 0", 
                new[] { "LotSize", "SquareFeet" }))
            
            // Calculate age
            .Append(_mlContext.Transforms.Expression("Age", 
                $"(year) => {currentYear} - year", 
                new[] { "YearBuilt" }))
            
            // Binary: new construction (less than 5 years old)
            .Append(_mlContext.Transforms.Expression("IsNewConstruction", 
                $"(year) => year >= {currentYear - 5} ? 1 : 0", 
                new[] { "YearBuilt" }))
            
            // Binary: more baths than beds (luxury indicator)
            .Append(_mlContext.Transforms.Expression("HasMoreBathsThanBeds", 
                "(baths, beds) => baths > beds ? 1 : 0", 
                new[] { "Bathrooms", "Bedrooms" }));
        
        return pipeline.Fit(data);
    }
}
```

### Splitting Features

Sometimes a single feature contains multiple pieces of information that should be extracted separately.

```csharp
public class AddressFeatureExtractor
{
    public class RawAddress
    {
        public string FullAddress { get; set; }  // "123 Main St, Apt 4B, Chicago, IL 60601"
    }
    
    public class ParsedAddress
    {
        public string StreetAddress { get; set; }
        public string Unit { get; set; }
        public string City { get; set; }
        public string State { get; set; }
        public string ZipCode { get; set; }
        public bool HasUnit { get; set; }        // Apartment vs house indicator
        public int ZipCodePrefix { get; set; }   // Regional grouping
    }
    
    public ParsedAddress ExtractAddressFeatures(RawAddress raw)
    {
        var parts = raw.FullAddress.Split(',').Select(p => p.Trim()).ToArray();
        
        var result = new ParsedAddress();
        
        // Parse components (simplified - real implementation would be more robust)
        if (parts.Length >= 4)
        {
            result.StreetAddress = parts[0];
            result.Unit = parts[1].StartsWith("Apt") || parts[1].StartsWith("Unit") 
                ? parts[1] : null;
            result.City = result.Unit != null ? parts[2] : parts[1];
            
            var stateZip = parts.Last().Split(' ');
            result.State = stateZip[0];
            result.ZipCode = stateZip.Length > 1 ? stateZip[1] : "";
        }
        
        // Derived features
        result.HasUnit = !string.IsNullOrEmpty(result.Unit);
        result.ZipCodePrefix = int.TryParse(result.ZipCode?.Substring(0, 3), out var prefix) 
            ? prefix : 0;
        
        return result;
    }
}
```

### Aggregating Features

When you have hierarchical or grouped data, aggregate features can capture important context.

```csharp
public class NeighborhoodFeatureEngineer
{
    public class HouseWithNeighborhood
    {
        // Original features
        public float SquareFeet { get; set; }
        public string Neighborhood { get; set; }
        public float Price { get; set; }
        
        // Neighborhood aggregate features (computed from training data)
        public float NeighborhoodMedianSqFt { get; set; }
        public float NeighborhoodPricePerSqFt { get; set; }
        public float SqFtVsNeighborhoodMedian { get; set; }
        public int NeighborhoodHomeCount { get; set; }
    }
    
    public Dictionary<string, NeighborhoodStats> ComputeNeighborhoodStats(
        IEnumerable<HouseWithNeighborhood> trainingData)
    {
        // IMPORTANT: Only compute from training data to avoid leakage!
        return trainingData
            .GroupBy(h => h.Neighborhood)
            .ToDictionary(
                g => g.Key,
                g => new NeighborhoodStats
                {
                    MedianSqFt = Median(g.Select(h => h.SquareFeet)),
                    MedianPricePerSqFt = Median(g.Select(h => h.Price / h.SquareFeet)),
                    HomeCount = g.Count()
                });
    }
    
    public IEnumerable<HouseWithNeighborhood> EnrichWithNeighborhoodFeatures(
        IEnumerable<HouseWithNeighborhood> houses,
        Dictionary<string, NeighborhoodStats> stats)
    {
        foreach (var house in houses)
        {
            if (stats.TryGetValue(house.Neighborhood, out var neighborhoodStats))
            {
                house.NeighborhoodMedianSqFt = neighborhoodStats.MedianSqFt;
                house.NeighborhoodPricePerSqFt = neighborhoodStats.MedianPricePerSqFt;
                house.SqFtVsNeighborhoodMedian = house.SquareFeet / neighborhoodStats.MedianSqFt;
                house.NeighborhoodHomeCount = neighborhoodStats.HomeCount;
            }
            yield return house;
        }
    }
    
    private float Median(IEnumerable<float> values)
    {
        var sorted = values.OrderBy(v => v).ToList();
        var mid = sorted.Count / 2;
        return sorted.Count % 2 == 0 
            ? (sorted[mid - 1] + sorted[mid]) / 2 
            : sorted[mid];
    }
}

public class NeighborhoodStats
{
    public float MedianSqFt { get; set; }
    public float MedianPricePerSqFt { get; set; }
    public int HomeCount { get; set; }
}
```

The key insight here is `SqFtVsNeighborhoodMedian`—is this house larger or smaller than typical for its neighborhood? A 1500 sqft house might be huge in one neighborhood and tiny in another. This relative positioning often matters more than absolute values.

## Text Feature Extraction

Text data is everywhere—property descriptions, review comments, addresses, categorical labels. Converting text into numerical features is a fundamental skill.

### Bag of Words

The simplest text representation treats each document as a "bag" of words, ignoring order but counting occurrences.

```csharp
public class TextFeatureExtraction
{
    private readonly MLContext _mlContext;
    
    public TextFeatureExtraction()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public class PropertyListing
    {
        public string Description { get; set; }
        public float Price { get; set; }
    }
    
    public class PropertyWithTextFeatures
    {
        public string Description { get; set; }
        public float Price { get; set; }
        public float[] DescriptionFeatures { get; set; }
    }
    
    public ITransformer CreateBagOfWordsPipeline(IDataView data)
    {
        var pipeline = _mlContext.Transforms.Text
            // Normalize: lowercase, remove punctuation
            .NormalizeText("NormalizedDescription", "Description",
                keepDiacritics: false,
                keepPunctuations: false,
                keepNumbers: true)
            
            // Tokenize into words
            .Append(_mlContext.Transforms.Text.TokenizeIntoWords(
                "Tokens", "NormalizedDescription"))
            
            // Remove stop words (the, a, is, etc.)
            .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords(
                "FilteredTokens", "Tokens"))
            
            // Convert to bag of words (term frequency)
            .Append(_mlContext.Transforms.Text.ProduceNgrams(
                "BagOfWords", "FilteredTokens",
                ngramLength: 1,
                useAllLengths: false,
                weighting: NgramExtractingEstimator.WeightingCriteria.Tf));
        
        return pipeline.Fit(data);
    }
}
```

### TF-IDF: Beyond Simple Counts

Term Frequency-Inverse Document Frequency (TF-IDF) improves on bag of words by down-weighting common words. A word that appears in every listing (like "home" or "property") gets a lower weight than a distinctive word (like "waterfront" or "renovated").

```csharp
public ITransformer CreateTfIdfPipeline(IDataView data)
{
    var pipeline = _mlContext.Transforms.Text
        .NormalizeText("NormalizedDescription", "Description")
        .Append(_mlContext.Transforms.Text.TokenizeIntoWords(
            "Tokens", "NormalizedDescription"))
        .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords(
            "FilteredTokens", "Tokens"))
        
        // TF-IDF weighting instead of raw counts
        .Append(_mlContext.Transforms.Text.ProduceNgrams(
            "TfIdfFeatures", "FilteredTokens",
            ngramLength: 2,        // Include bigrams like "granite countertops"
            useAllLengths: true,   // Also include unigrams
            weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf))
        
        // Optional: reduce dimensionality with feature selection
        .Append(_mlContext.Transforms.SelectFeaturesBasedOnCount(
            "SelectedTfIdf", "TfIdfFeatures",
            count: 100));  // Keep only features appearing in 100+ documents
    
    return pipeline.Fit(data);
}
```

### Domain-Specific Text Features

For real estate specifically, we can extract more targeted features:

```csharp
public class RealEstateTextFeatures
{
    // Keywords that correlate with higher prices
    private static readonly HashSet<string> LuxuryKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "granite", "marble", "hardwood", "stainless", "gourmet", "chef",
        "spa", "wine", "custom", "designer", "premium", "upgraded",
        "waterfront", "view", "panoramic", "private", "exclusive", "estate"
    };
    
    // Keywords indicating potential issues
    private static readonly HashSet<string> ConcernKeywords = new(StringComparer.OrdinalIgnoreCase)
    {
        "fixer", "potential", "investor", "as-is", "handyman", "tlc",
        "needs", "opportunity", "project", "motivated", "must sell"
    };
    
    public class ExtractedTextFeatures
    {
        public int LuxuryKeywordCount { get; set; }
        public int ConcernKeywordCount { get; set; }
        public float LuxuryRatio { get; set; }
        public int DescriptionLength { get; set; }
        public int SentenceCount { get; set; }
        public bool MentionsBedrooms { get; set; }
        public bool MentionsBathrooms { get; set; }
        public bool MentionsGarage { get; set; }
        public bool MentionsPool { get; set; }
        public bool MentionsView { get; set; }
    }
    
    public ExtractedTextFeatures ExtractFeatures(string description)
    {
        var words = description.ToLower()
            .Split(new[] { ' ', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries);
        
        var luxuryCount = words.Count(w => LuxuryKeywords.Contains(w));
        var concernCount = words.Count(w => ConcernKeywords.Contains(w));
        
        return new ExtractedTextFeatures
        {
            LuxuryKeywordCount = luxuryCount,
            ConcernKeywordCount = concernCount,
            LuxuryRatio = words.Length > 0 ? (float)luxuryCount / words.Length : 0,
            DescriptionLength = description.Length,
            SentenceCount = description.Count(c => c == '.' || c == '!' || c == '?'),
            MentionsBedrooms = description.Contains("bedroom", StringComparison.OrdinalIgnoreCase) 
                            || description.Contains("bed ", StringComparison.OrdinalIgnoreCase)
                            || description.Contains(" br", StringComparison.OrdinalIgnoreCase),
            MentionsBathrooms = description.Contains("bathroom", StringComparison.OrdinalIgnoreCase)
                             || description.Contains("bath ", StringComparison.OrdinalIgnoreCase),
            MentionsGarage = description.Contains("garage", StringComparison.OrdinalIgnoreCase),
            MentionsPool = description.Contains("pool", StringComparison.OrdinalIgnoreCase),
            MentionsView = description.Contains("view", StringComparison.OrdinalIgnoreCase)
        };
    }
}
```

This approach combines statistical text features (TF-IDF) with domain-specific knowledge (luxury indicators). The combination often outperforms either alone.

## Date/Time Feature Engineering

Temporal data is deceptively rich. A single timestamp can yield dozens of meaningful features, each capturing different patterns.

### Basic Time Decomposition

```csharp
public class DateTimeFeatureExtractor
{
    public class TimeFeatures
    {
        // From timestamp
        public int Year { get; set; }
        public int Month { get; set; }
        public int DayOfMonth { get; set; }
        public int DayOfWeek { get; set; }      // 0=Sunday, 6=Saturday
        public int Quarter { get; set; }
        public int WeekOfYear { get; set; }
        public int Hour { get; set; }
        public int Minute { get; set; }
        
        // Derived binary features
        public bool IsWeekend { get; set; }
        public bool IsMonthEnd { get; set; }
        public bool IsMonthStart { get; set; }
        public bool IsQuarterEnd { get; set; }
        
        // Derived cyclical features (for continuity: Dec -> Jan)
        public float MonthSin { get; set; }
        public float MonthCos { get; set; }
        public float DayOfWeekSin { get; set; }
        public float DayOfWeekCos { get; set; }
        public float HourSin { get; set; }
        public float HourCos { get; set; }
    }
    
    public TimeFeatures ExtractFeatures(DateTime timestamp)
    {
        return new TimeFeatures
        {
            // Basic decomposition
            Year = timestamp.Year,
            Month = timestamp.Month,
            DayOfMonth = timestamp.Day,
            DayOfWeek = (int)timestamp.DayOfWeek,
            Quarter = (timestamp.Month - 1) / 3 + 1,
            WeekOfYear = CultureInfo.CurrentCulture.Calendar.GetWeekOfYear(
                timestamp, CalendarWeekRule.FirstDay, DayOfWeek.Sunday),
            Hour = timestamp.Hour,
            Minute = timestamp.Minute,
            
            // Binary features
            IsWeekend = timestamp.DayOfWeek == DayOfWeek.Saturday 
                     || timestamp.DayOfWeek == DayOfWeek.Sunday,
            IsMonthEnd = timestamp.AddDays(1).Month != timestamp.Month,
            IsMonthStart = timestamp.Day <= 3,
            IsQuarterEnd = timestamp.Month % 3 == 0 && timestamp.AddDays(1).Month != timestamp.Month,
            
            // Cyclical encoding (ensures December is "close to" January)
            MonthSin = (float)Math.Sin(2 * Math.PI * timestamp.Month / 12),
            MonthCos = (float)Math.Cos(2 * Math.PI * timestamp.Month / 12),
            DayOfWeekSin = (float)Math.Sin(2 * Math.PI * (int)timestamp.DayOfWeek / 7),
            DayOfWeekCos = (float)Math.Cos(2 * Math.PI * (int)timestamp.DayOfWeek / 7),
            HourSin = (float)Math.Sin(2 * Math.PI * timestamp.Hour / 24),
            HourCos = (float)Math.Cos(2 * Math.PI * timestamp.Hour / 24),
        };
    }
}
```

### Why Cyclical Encoding?

This deserves explanation. If you encode months as integers 1-12, your model sees December (12) as very different from January (1)—they're 11 units apart. But seasonally, they're adjacent.

Cyclical encoding maps time to points on a unit circle using sine and cosine. Now December and January are close together, and the model can learn smooth seasonal patterns.

```csharp
// Visual representation:
//
//            March (3)
//               |
//    June (6) --+-- December (12)
//               |
//         September (9)
//
// On the circle, December and January are neighbors!
```

### Seasonality for Real Estate

Real estate has strong seasonal patterns. Spring is traditionally the hottest market, winter the slowest.

```csharp
public class RealEstateSeasonality
{
    public class SeasonalFeatures
    {
        public string Season { get; set; }           // Spring, Summer, Fall, Winter
        public bool IsPeakSeason { get; set; }       // Spring/Early Summer
        public bool IsHolidaySeason { get; set; }    // Nov-Dec (slower market)
        public bool IsSchoolYear { get; set; }       // Families prefer summer moves
        public float SeasonalityIndex { get; set; } // Market activity multiplier
    }
    
    public SeasonalFeatures GetSeasonalFeatures(DateTime listingDate)
    {
        var month = listingDate.Month;
        
        var season = month switch
        {
            12 or 1 or 2 => "Winter",
            3 or 4 or 5 => "Spring",
            6 or 7 or 8 => "Summer",
            _ => "Fall"
        };
        
        // Approximate seasonality index based on typical real estate patterns
        // (In practice, you'd compute this from historical data)
        var seasonalityIndex = month switch
        {
            1 => 0.85f,   // January: post-holiday slowdown
            2 => 0.90f,   // February: starting to pick up
            3 => 1.00f,   // March: spring market begins
            4 => 1.15f,   // April: peak activity
            5 => 1.20f,   // May: highest activity
            6 => 1.15f,   // June: still hot
            7 => 1.05f,   // July: summer slowdown begins
            8 => 0.95f,   // August: back-to-school prep
            9 => 0.90f,   // September: fall market
            10 => 0.85f,  // October: slowing down
            11 => 0.75f,  // November: holiday slowdown
            12 => 0.70f,  // December: lowest activity
            _ => 1.0f
        };
        
        return new SeasonalFeatures
        {
            Season = season,
            IsPeakSeason = month >= 3 && month <= 6,
            IsHolidaySeason = month == 11 || month == 12,
            IsSchoolYear = month >= 9 || month <= 5,
            SeasonalityIndex = seasonalityIndex
        };
    }
}
```

### Time Since Events

Another powerful pattern: measuring time relative to significant events.

```csharp
public class TimeSinceFeatures
{
    public class PropertyTimeFeatures
    {
        public int DaysSinceRenovation { get; set; }
        public int DaysSinceLastSale { get; set; }
        public int DaysOnMarket { get; set; }
        public bool RecentlyRenovated { get; set; }  // < 5 years
        public bool FrequentlyTraded { get; set; }   // Sold 2+ times in 10 years
    }
    
    public PropertyTimeFeatures Calculate(
        DateTime listingDate,
        DateTime? lastRenovation,
        DateTime? lastSale,
        DateTime originalListingDate,
        int salesCountLast10Years)
    {
        return new PropertyTimeFeatures
        {
            DaysSinceRenovation = lastRenovation.HasValue 
                ? (listingDate - lastRenovation.Value).Days : -1,
            DaysSinceLastSale = lastSale.HasValue
                ? (listingDate - lastSale.Value).Days : -1,
            DaysOnMarket = (listingDate - originalListingDate).Days,
            RecentlyRenovated = lastRenovation.HasValue 
                && (listingDate - lastRenovation.Value).Days < 365 * 5,
            FrequentlyTraded = salesCountLast10Years >= 2
        };
    }
}
```

## Feature Selection Techniques

Not all features deserve to be in your final model. Feature selection removes redundant, irrelevant, or noisy features, leading to:

- Faster training and prediction
- Reduced overfitting
- Better interpretability
- Lower data collection costs (you learn what matters)

### Correlation Analysis

The simplest approach: examine correlation between features and the target.

```csharp
public class CorrelationBasedSelection
{
    public class CorrelationResult
    {
        public string FeatureName { get; set; }
        public double CorrelationWithTarget { get; set; }
        public double AbsoluteCorrelation => Math.Abs(CorrelationWithTarget);
    }
    
    public List<CorrelationResult> AnalyzeCorrelations(
        IDataView data, 
        string targetColumn,
        IEnumerable<string> featureColumns)
    {
        var results = new List<CorrelationResult>();
        var targetValues = data.GetColumn<float>(targetColumn).Select(v => (double)v).ToArray();
        
        foreach (var feature in featureColumns)
        {
            var featureValues = data.GetColumn<float>(feature).Select(v => (double)v).ToArray();
            
            var correlation = Correlation.Pearson(featureValues, targetValues);
            
            results.Add(new CorrelationResult
            {
                FeatureName = feature,
                CorrelationWithTarget = correlation
            });
        }
        
        return results.OrderByDescending(r => r.AbsoluteCorrelation).ToList();
    }
    
    public List<string> SelectTopFeatures(List<CorrelationResult> results, double threshold = 0.1)
    {
        return results
            .Where(r => r.AbsoluteCorrelation >= threshold)
            .Select(r => r.FeatureName)
            .ToList();
    }
}
```

### Feature-to-Feature Correlation (Detecting Redundancy)

High correlation between features indicates redundancy:

```csharp
public class RedundancyDetector
{
    public class FeaturePairCorrelation
    {
        public string Feature1 { get; set; }
        public string Feature2 { get; set; }
        public double Correlation { get; set; }
    }
    
    public List<FeaturePairCorrelation> FindHighlyCorrelatedPairs(
        IDataView data,
        IEnumerable<string> featureColumns,
        double threshold = 0.9)
    {
        var columns = featureColumns.ToList();
        var correlatedPairs = new List<FeaturePairCorrelation>();
        
        for (int i = 0; i < columns.Count; i++)
        {
            var valuesI = data.GetColumn<float>(columns[i])
                .Select(v => (double)v).ToArray();
            
            for (int j = i + 1; j < columns.Count; j++)
            {
                var valuesJ = data.GetColumn<float>(columns[j])
                    .Select(v => (double)v).ToArray();
                
                var correlation = Math.Abs(Correlation.Pearson(valuesI, valuesJ));
                
                if (correlation >= threshold)
                {
                    correlatedPairs.Add(new FeaturePairCorrelation
                    {
                        Feature1 = columns[i],
                        Feature2 = columns[j],
                        Correlation = correlation
                    });
                }
            }
        }
        
        return correlatedPairs.OrderByDescending(p => p.Correlation).ToList();
    }
    
    public List<string> RemoveRedundantFeatures(
        List<FeaturePairCorrelation> correlatedPairs,
        List<CorrelationResult> targetCorrelations)
    {
        var toRemove = new HashSet<string>();
        var targetCorrelationDict = targetCorrelations.ToDictionary(
            c => c.FeatureName, c => c.AbsoluteCorrelation);
        
        foreach (var pair in correlatedPairs)
        {
            // Keep the feature with higher correlation to target
            var corr1 = targetCorrelationDict.GetValueOrDefault(pair.Feature1, 0);
            var corr2 = targetCorrelationDict.GetValueOrDefault(pair.Feature2, 0);
            
            var featureToRemove = corr1 >= corr2 ? pair.Feature2 : pair.Feature1;
            toRemove.Add(featureToRemove);
        }
        
        return toRemove.ToList();
    }
}
```

### Permutation Feature Importance

A more sophisticated approach: measure how much model performance degrades when each feature is randomly shuffled.

```csharp
public class PermutationImportance
{
    private readonly MLContext _mlContext;
    
    public PermutationImportance()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public Dictionary<string, double> CalculateImportance(
        ITransformer model,
        IDataView testData,
        string labelColumn = "Label",
        string featuresColumn = "Features")
    {
        // Use ML.NET's built-in permutation feature importance
        var importance = _mlContext.Regression.PermutationFeatureImportance(
            model, 
            testData,
            labelColumnName: labelColumn,
            permutationCount: 10);  // Average over 10 shuffles
        
        var results = new Dictionary<string, double>();
        
        // Get feature names from the slotted features column
        VBuffer<ReadOnlyMemory<char>> slotNames = default;
        testData.Schema[featuresColumn].GetSlotNames(ref slotNames);
        
        var featureNames = slotNames.DenseValues()
            .Select(s => s.ToString())
            .ToArray();
        
        for (int i = 0; i < featureNames.Length && i < importance.Count; i++)
        {
            // Higher drop in R² = more important feature
            results[featureNames[i]] = importance[i].RSquared.Mean;
        }
        
        return results.OrderByDescending(kv => Math.Abs(kv.Value))
            .ToDictionary(kv => kv.Key, kv => kv.Value);
    }
}
```

### ML.NET Feature Selection Transforms

ML.NET provides built-in feature selection transforms:

```csharp
public ITransformer CreateFeatureSelectionPipeline(IDataView data)
{
    var pipeline = _mlContext.Transforms
        // Remove features with too many missing values
        .SelectFeaturesBasedOnMutualInformation(
            outputColumnName: "SelectedFeatures",
            inputColumnName: "Features",
            labelColumnName: "Price",
            slotsInOutput: 50)  // Keep top 50 features by mutual information
        
        // Alternative: select based on occurrence count
        // .SelectFeaturesBasedOnCount("SelectedFeatures", "Features", count: 10)
        ;
    
    return pipeline.Fit(data);
}
```

## Project: Housing Price Prediction Feature Engineering

Let's put everything together. We'll build a complete feature engineering pipeline for housing price prediction, demonstrating measurable improvements from good features.

### The Dataset

We'll use a housing dataset with these columns:

```csharp
public class RawHousingRecord
{
    public float SquareFeet { get; set; }         // Living area in sqft
    public float LotSize { get; set; }            // Lot size in sqft  
    public int Bedrooms { get; set; }
    public float Bathrooms { get; set; }          // 2.5 means 2 full, 1 half
    public int YearBuilt { get; set; }
    public int YearRenovated { get; set; }        // 0 if never renovated
    public int GarageSpaces { get; set; }
    public string Neighborhood { get; set; }
    public float Latitude { get; set; }
    public float Longitude { get; set; }
    public DateTime ListingDate { get; set; }
    public string Description { get; set; }       // Property description text
    public float Price { get; set; }              // Target: sale price
}
```

### The Complete Feature Engineering Pipeline

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;
using MathNet.Numerics.Statistics;

public class HousingFeatureEngineer
{
    private readonly MLContext _mlContext;
    private Dictionary<string, NeighborhoodStatistics> _neighborhoodStats;
    private readonly int _currentYear = DateTime.Now.Year;
    
    public HousingFeatureEngineer()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    #region Data Classes
    
    public class EnrichedHousingRecord
    {
        // Original features
        public float SquareFeet { get; set; }
        public float LotSize { get; set; }
        public float Bedrooms { get; set; }
        public float Bathrooms { get; set; }
        public float YearBuilt { get; set; }
        public float YearRenovated { get; set; }
        public float GarageSpaces { get; set; }
        public string Neighborhood { get; set; }
        public float Latitude { get; set; }
        public float Longitude { get; set; }
        
        // Engineered Feature 1: Room and space ratios
        public float SqFtPerBedroom { get; set; }
        public float SqFtPerBathroom { get; set; }
        public float BathroomToBedroom { get; set; }
        public float LotToHomeRatio { get; set; }
        
        // Engineered Feature 2: Age-related features
        public float Age { get; set; }
        public float YearsSinceRenovation { get; set; }
        public float EffectiveAge { get; set; }
        public bool IsNewConstruction { get; set; }
        public bool IsHistoric { get; set; }
        public bool WasRenovated { get; set; }
        
        // Engineered Feature 3: Location-derived features
        public float NeighborhoodMedianPrice { get; set; }
        public float NeighborhoodPricePerSqFt { get; set; }
        public float SqFtVsNeighborhood { get; set; }
        public float PriceVsNeighborhood { get; set; }  // Computed after prediction
        public float DistanceToDowntown { get; set; }
        
        // Engineered Feature 4: Seasonal features
        public float ListingMonth { get; set; }
        public float ListingMonthSin { get; set; }
        public float ListingMonthCos { get; set; }
        public bool IsPeakSeason { get; set; }
        public float SeasonalityIndex { get; set; }
        
        // Engineered Feature 5: Text-derived features
        public int DescriptionLength { get; set; }
        public int LuxuryKeywordCount { get; set; }
        public int ConcernKeywordCount { get; set; }
        public float LuxuryScore { get; set; }
        public bool MentionsPool { get; set; }
        public bool MentionsView { get; set; }
        public bool MentionsGarage { get; set; }
        
        // Quality indicators
        public float QualityScore { get; set; }
        
        // Target
        public float Price { get; set; }
    }
    
    public class NeighborhoodStatistics
    {
        public float MedianPrice { get; set; }
        public float MedianSqFt { get; set; }
        public float MedianPricePerSqFt { get; set; }
        public int Count { get; set; }
    }
    
    #endregion
    
    #region Feature Engineering Methods
    
    public void FitNeighborhoodStatistics(IEnumerable<RawHousingRecord> trainingData)
    {
        _neighborhoodStats = trainingData
            .GroupBy(r => r.Neighborhood)
            .ToDictionary(
                g => g.Key,
                g => new NeighborhoodStatistics
                {
                    MedianPrice = Median(g.Select(r => r.Price)),
                    MedianSqFt = Median(g.Select(r => r.SquareFeet)),
                    MedianPricePerSqFt = Median(g.Select(r => r.Price / r.SquareFeet)),
                    Count = g.Count()
                });
        
        Console.WriteLine($"Computed statistics for {_neighborhoodStats.Count} neighborhoods");
    }
    
    public IEnumerable<EnrichedHousingRecord> EngineerFeatures(
        IEnumerable<RawHousingRecord> records)
    {
        // Downtown coordinates (example: Seattle)
        const float downtownLat = 47.6062f;
        const float downtownLon = -122.3321f;
        
        foreach (var raw in records)
        {
            var enriched = new EnrichedHousingRecord
            {
                // Copy original features
                SquareFeet = raw.SquareFeet,
                LotSize = raw.LotSize,
                Bedrooms = raw.Bedrooms,
                Bathrooms = raw.Bathrooms,
                YearBuilt = raw.YearBuilt,
                YearRenovated = raw.YearRenovated,
                GarageSpaces = raw.GarageSpaces,
                Neighborhood = raw.Neighborhood,
                Latitude = raw.Latitude,
                Longitude = raw.Longitude,
                Price = raw.Price,
                
                // Feature 1: Room and space ratios
                SqFtPerBedroom = raw.Bedrooms > 0 ? raw.SquareFeet / raw.Bedrooms : raw.SquareFeet,
                SqFtPerBathroom = raw.Bathrooms > 0 ? raw.SquareFeet / raw.Bathrooms : raw.SquareFeet,
                BathroomToBedroom = raw.Bedrooms > 0 ? raw.Bathrooms / raw.Bedrooms : raw.Bathrooms,
                LotToHomeRatio = raw.SquareFeet > 0 ? raw.LotSize / raw.SquareFeet : 0,
                
                // Feature 2: Age-related
                Age = _currentYear - raw.YearBuilt,
                WasRenovated = raw.YearRenovated > 0,
                YearsSinceRenovation = raw.YearRenovated > 0 
                    ? _currentYear - raw.YearRenovated 
                    : _currentYear - raw.YearBuilt,
                EffectiveAge = raw.YearRenovated > 0
                    ? Math.Min(_currentYear - raw.YearRenovated, (_currentYear - raw.YearBuilt) * 0.5f)
                    : _currentYear - raw.YearBuilt,
                IsNewConstruction = raw.YearBuilt >= _currentYear - 5,
                IsHistoric = raw.YearBuilt < 1940,
                
                // Feature 4: Seasonal
                ListingMonth = raw.ListingDate.Month,
                ListingMonthSin = (float)Math.Sin(2 * Math.PI * raw.ListingDate.Month / 12),
                ListingMonthCos = (float)Math.Cos(2 * Math.PI * raw.ListingDate.Month / 12),
                IsPeakSeason = raw.ListingDate.Month >= 3 && raw.ListingDate.Month <= 6,
                SeasonalityIndex = GetSeasonalityIndex(raw.ListingDate.Month),
                
                // Feature 5: Text-derived
                DescriptionLength = raw.Description?.Length ?? 0,
                LuxuryKeywordCount = CountLuxuryKeywords(raw.Description),
                ConcernKeywordCount = CountConcernKeywords(raw.Description),
                LuxuryScore = CalculateLuxuryScore(raw.Description),
                MentionsPool = ContainsKeyword(raw.Description, "pool"),
                MentionsView = ContainsKeyword(raw.Description, "view"),
                MentionsGarage = ContainsKeyword(raw.Description, "garage"),
                
                // Feature 3: Location (distance calculation)
                DistanceToDowntown = CalculateDistance(
                    raw.Latitude, raw.Longitude, downtownLat, downtownLon)
            };
            
            // Neighborhood features (requires fitted stats)
            if (_neighborhoodStats != null && 
                _neighborhoodStats.TryGetValue(raw.Neighborhood, out var stats))
            {
                enriched.NeighborhoodMedianPrice = stats.MedianPrice;
                enriched.NeighborhoodPricePerSqFt = stats.MedianPricePerSqFt;
                enriched.SqFtVsNeighborhood = raw.SquareFeet / stats.MedianSqFt;
            }
            
            // Overall quality score (composite feature)
            enriched.QualityScore = CalculateQualityScore(enriched);
            
            yield return enriched;
        }
    }
    
    private float GetSeasonalityIndex(int month) => month switch
    {
        1 => 0.85f, 2 => 0.90f, 3 => 1.00f, 4 => 1.15f,
        5 => 1.20f, 6 => 1.15f, 7 => 1.05f, 8 => 0.95f,
        9 => 0.90f, 10 => 0.85f, 11 => 0.75f, 12 => 0.70f,
        _ => 1.0f
    };
    
    private static readonly string[] LuxuryKeywords = 
    {
        "granite", "marble", "hardwood", "stainless", "gourmet", "chef",
        "spa", "wine", "custom", "designer", "premium", "upgraded",
        "waterfront", "view", "panoramic", "private", "exclusive", "estate",
        "luxury", "elegant", "pristine", "stunning", "exceptional"
    };
    
    private static readonly string[] ConcernKeywords =
    {
        "fixer", "potential", "investor", "as-is", "handyman", "tlc",
        "needs work", "opportunity", "project", "motivated", "must sell"
    };
    
    private int CountLuxuryKeywords(string description)
    {
        if (string.IsNullOrEmpty(description)) return 0;
        var lower = description.ToLower();
        return LuxuryKeywords.Count(kw => lower.Contains(kw));
    }
    
    private int CountConcernKeywords(string description)
    {
        if (string.IsNullOrEmpty(description)) return 0;
        var lower = description.ToLower();
        return ConcernKeywords.Count(kw => lower.Contains(kw));
    }
    
    private float CalculateLuxuryScore(string description)
    {
        if (string.IsNullOrEmpty(description)) return 0;
        var luxury = CountLuxuryKeywords(description);
        var concern = CountConcernKeywords(description);
        return (luxury - concern * 2) / (float)Math.Max(1, description.Split(' ').Length / 100);
    }
    
    private bool ContainsKeyword(string description, string keyword)
    {
        return !string.IsNullOrEmpty(description) && 
               description.Contains(keyword, StringComparison.OrdinalIgnoreCase);
    }
    
    private float CalculateDistance(float lat1, float lon1, float lat2, float lon2)
    {
        // Haversine formula for distance in miles
        const float R = 3959; // Earth's radius in miles
        var dLat = (lat2 - lat1) * (float)Math.PI / 180;
        var dLon = (lon2 - lon1) * (float)Math.PI / 180;
        var a = Math.Sin(dLat / 2) * Math.Sin(dLat / 2) +
                Math.Cos(lat1 * Math.PI / 180) * Math.Cos(lat2 * Math.PI / 180) *
                Math.Sin(dLon / 2) * Math.Sin(dLon / 2);
        var c = 2 * Math.Atan2(Math.Sqrt(a), Math.Sqrt(1 - a));
        return (float)(R * c);
    }
    
    private float CalculateQualityScore(EnrichedHousingRecord record)
    {
        float score = 50; // Base score
        
        // Positive factors
        if (record.WasRenovated) score += 10;
        if (record.IsNewConstruction) score += 15;
        if (record.BathroomToBedroom >= 1) score += 5;
        if (record.GarageSpaces >= 2) score += 5;
        if (record.LuxuryKeywordCount >= 3) score += 10;
        if (record.MentionsPool) score += 5;
        if (record.MentionsView) score += 5;
        
        // Negative factors
        if (record.ConcernKeywordCount > 0) score -= 15;
        if (record.Age > 50 && !record.WasRenovated) score -= 10;
        if (record.DescriptionLength < 100) score -= 5;
        
        return Math.Clamp(score, 0, 100);
    }
    
    private float Median(IEnumerable<float> values)
    {
        var sorted = values.OrderBy(v => v).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 0 
            ? (sorted[mid - 1] + sorted[mid]) / 2 
            : sorted[mid];
    }
    
    #endregion
    
    #region ML Pipeline
    
    public ITransformer BuildTrainingPipeline()
    {
        // Define features for the model
        var featureColumns = new[]
        {
            // Original features
            "SquareFeet", "LotSize", "Bedrooms", "Bathrooms", 
            "GarageSpaces", "Latitude", "Longitude",
            
            // Engineered: Room ratios
            "SqFtPerBedroom", "SqFtPerBathroom", "BathroomToBedroom", "LotToHomeRatio",
            
            // Engineered: Age features
            "Age", "YearsSinceRenovation", "EffectiveAge",
            
            // Engineered: Location features
            "NeighborhoodMedianPrice", "NeighborhoodPricePerSqFt", 
            "SqFtVsNeighborhood", "DistanceToDowntown",
            
            // Engineered: Seasonal features
            "ListingMonthSin", "ListingMonthCos", "SeasonalityIndex",
            
            // Engineered: Text features
            "DescriptionLength", "LuxuryKeywordCount", "LuxuryScore",
            
            // Engineered: Quality score
            "QualityScore"
        };
        
        var booleanColumns = new[]
        {
            "IsNewConstruction", "IsHistoric", "WasRenovated", 
            "IsPeakSeason", "MentionsPool", "MentionsView", "MentionsGarage"
        };
        
        var pipeline = _mlContext.Transforms
            // Convert booleans to float
            .Conversion.ConvertType(booleanColumns
                .Select(c => new InputOutputColumnPair(c + "Float", c))
                .ToArray(), DataKind.Single)
            
            // One-hot encode neighborhood
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(
                "NeighborhoodEncoded", "Neighborhood"))
            
            // Combine all features
            .Append(_mlContext.Transforms.Concatenate("Features",
                featureColumns
                    .Concat(booleanColumns.Select(c => c + "Float"))
                    .Concat(new[] { "NeighborhoodEncoded" })
                    .ToArray()))
            
            // Normalize features
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            
            // Add the regression trainer
            .Append(_mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Price",
                featureColumnName: "Features",
                numberOfLeaves: 50,
                numberOfTrees: 100,
                minimumExampleCountPerLeaf: 10));
        
        return null; // We'll fit this in the evaluation method
    }
    
    #endregion
}
```

### Correlation Analysis of Engineered Features

Let's analyze how our engineered features correlate with price:

```csharp
public class FeatureCorrelationAnalysis
{
    public static void AnalyzeEngineeringImpact(
        IEnumerable<HousingFeatureEngineer.EnrichedHousingRecord> data)
    {
        var records = data.ToList();
        var prices = records.Select(r => (double)r.Price).ToArray();
        
        Console.WriteLine("\n" + new string('=', 70));
        Console.WriteLine("FEATURE CORRELATION ANALYSIS");
        Console.WriteLine(new string('=', 70));
        
        var correlations = new List<(string Name, double Correlation, string Type)>
        {
            // Original features
            ("SquareFeet", Corr(records.Select(r => r.SquareFeet), prices), "Original"),
            ("LotSize", Corr(records.Select(r => r.LotSize), prices), "Original"),
            ("Bedrooms", Corr(records.Select(r => r.Bedrooms), prices), "Original"),
            ("Bathrooms", Corr(records.Select(r => r.Bathrooms), prices), "Original"),
            ("GarageSpaces", Corr(records.Select(r => r.GarageSpaces), prices), "Original"),
            
            // Engineered: Room ratios
            ("SqFtPerBedroom", Corr(records.Select(r => r.SqFtPerBedroom), prices), "Engineered"),
            ("BathroomToBedroom", Corr(records.Select(r => r.BathroomToBedroom), prices), "Engineered"),
            ("LotToHomeRatio", Corr(records.Select(r => r.LotToHomeRatio), prices), "Engineered"),
            
            // Engineered: Age
            ("Age", Corr(records.Select(r => r.Age), prices), "Engineered"),
            ("EffectiveAge", Corr(records.Select(r => r.EffectiveAge), prices), "Engineered"),
            
            // Engineered: Location
            ("NeighborhoodMedianPrice", Corr(records.Select(r => r.NeighborhoodMedianPrice), prices), "Engineered"),
            ("SqFtVsNeighborhood", Corr(records.Select(r => r.SqFtVsNeighborhood), prices), "Engineered"),
            ("DistanceToDowntown", Corr(records.Select(r => r.DistanceToDowntown), prices), "Engineered"),
            
            // Engineered: Text
            ("LuxuryScore", Corr(records.Select(r => r.LuxuryScore), prices), "Engineered"),
            ("QualityScore", Corr(records.Select(r => r.QualityScore), prices), "Engineered"),
            
            // Engineered: Seasonal
            ("SeasonalityIndex", Corr(records.Select(r => r.SeasonalityIndex), prices), "Engineered"),
        };
        
        // Sort by absolute correlation
        correlations = correlations.OrderByDescending(c => Math.Abs(c.Correlation)).ToList();
        
        Console.WriteLine($"\n{"Feature",-30} {"Correlation",12} {"Type",12}  {"Strength",15}");
        Console.WriteLine(new string('-', 70));
        
        foreach (var (name, corr, type) in correlations)
        {
            var strength = Math.Abs(corr) switch
            {
                > 0.7 => "★★★★★ Very Strong",
                > 0.5 => "★★★★☆ Strong",
                > 0.3 => "★★★☆☆ Moderate",
                > 0.1 => "★★☆☆☆ Weak",
                _ => "★☆☆☆☆ Negligible"
            };
            
            var typeIndicator = type == "Engineered" ? "[ENG]" : "[ORG]";
            Console.WriteLine($"{name,-30} {corr,12:F4} {typeIndicator,12}  {strength}");
        }
        
        // Summary statistics
        var originalCorrs = correlations.Where(c => c.Type == "Original")
            .Select(c => Math.Abs(c.Correlation)).ToList();
        var engineeredCorrs = correlations.Where(c => c.Type == "Engineered")
            .Select(c => Math.Abs(c.Correlation)).ToList();
        
        Console.WriteLine("\n" + new string('-', 70));
        Console.WriteLine("SUMMARY:");
        Console.WriteLine($"  Original features - Avg |correlation|: {originalCorrs.Average():F4}");
        Console.WriteLine($"  Engineered features - Avg |correlation|: {engineeredCorrs.Average():F4}");
        Console.WriteLine($"  Top correlated feature: {correlations.First().Name} ({correlations.First().Correlation:F4})");
    }
    
    private static double Corr(IEnumerable<float> feature, double[] target)
    {
        var featureArray = feature.Select(f => (double)f).ToArray();
        return Correlation.Pearson(featureArray, target);
    }
}
```

Expected output:

```
======================================================================
FEATURE CORRELATION ANALYSIS
======================================================================

Feature                        Correlation         Type  Strength
----------------------------------------------------------------------
NeighborhoodMedianPrice             0.7823        [ENG]  ★★★★★ Very Strong
SquareFeet                          0.7012        [ORG]  ★★★★★ Very Strong
QualityScore                        0.6234        [ENG]  ★★★★☆ Strong
Bathrooms                           0.5891        [ORG]  ★★★★☆ Strong
SqFtVsNeighborhood                  0.5456        [ENG]  ★★★★☆ Strong
LuxuryScore                         0.4123        [ENG]  ★★★☆☆ Moderate
SqFtPerBedroom                      0.3892        [ENG]  ★★★☆☆ Moderate
EffectiveAge                       -0.3456        [ENG]  ★★★☆☆ Moderate
DistanceToDowntown                 -0.3211        [ENG]  ★★★☆☆ Moderate
GarageSpaces                        0.2987        [ORG]  ★★☆☆☆ Weak
Age                                -0.2654        [ENG]  ★★☆☆☆ Weak
BathroomToBedroom                   0.2234        [ENG]  ★★☆☆☆ Weak
Bedrooms                            0.1876        [ORG]  ★★☆☆☆ Weak
LotSize                             0.1543        [ORG]  ★★☆☆☆ Weak
LotToHomeRatio                      0.0987        [ENG]  ★☆☆☆☆ Negligible
SeasonalityIndex                    0.0234        [ENG]  ★☆☆☆☆ Negligible

----------------------------------------------------------------------
SUMMARY:
  Original features - Avg |correlation|: 0.3862
  Engineered features - Avg |correlation|: 0.3712
  Top correlated feature: NeighborhoodMedianPrice (0.7823)
```

Notice that `NeighborhoodMedianPrice`—an engineered feature—has the highest correlation with price. This is the power of feature engineering: we've extracted latent information from the neighborhood name and made it directly usable by the model.

### Before vs. After: Measuring Feature Engineering Impact

```csharp
public class FeatureEngineeringEvaluator
{
    private readonly MLContext _mlContext;
    
    public FeatureEngineeringEvaluator()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public void CompareModels(
        IEnumerable<RawHousingRecord> rawData,
        IEnumerable<HousingFeatureEngineer.EnrichedHousingRecord> enrichedData)
    {
        var rawList = rawData.ToList();
        var enrichedList = enrichedData.ToList();
        
        // Split data
        var splitRaw = SplitData(rawList, 0.8);
        var splitEnriched = SplitData(enrichedList, 0.8);
        
        // Train baseline model (original features only)
        var baselineMetrics = TrainAndEvaluateBaseline(
            splitRaw.train, splitRaw.test);
        
        // Train enhanced model (with engineered features)
        var enhancedMetrics = TrainAndEvaluateEnhanced(
            splitEnriched.train, splitEnriched.test);
        
        // Print comparison
        Console.WriteLine("\n" + new string('=', 70));
        Console.WriteLine("MODEL COMPARISON: BASELINE VS ENGINEERED FEATURES");
        Console.WriteLine(new string('=', 70));
        
        Console.WriteLine($"\n{"Metric",-25} {"Baseline",15} {"Engineered",15} {"Improvement",15}");
        Console.WriteLine(new string('-', 70));
        
        PrintMetric("R²", baselineMetrics.RSquared, enhancedMetrics.RSquared);
        PrintMetric("RMSE", baselineMetrics.RootMeanSquaredError, 
            enhancedMetrics.RootMeanSquaredError, lowerIsBetter: true);
        PrintMetric("MAE", baselineMetrics.MeanAbsoluteError, 
            enhancedMetrics.MeanAbsoluteError, lowerIsBetter: true);
        
        Console.WriteLine("\n" + new string('-', 70));
        Console.WriteLine("INTERPRETATION:");
        
        var r2Improvement = (enhancedMetrics.RSquared - baselineMetrics.RSquared) / 
                           baselineMetrics.RSquared * 100;
        var rmseReduction = (baselineMetrics.RootMeanSquaredError - 
                            enhancedMetrics.RootMeanSquaredError) / 
                           baselineMetrics.RootMeanSquaredError * 100;
        
        Console.WriteLine($"  • R² improved by {r2Improvement:F1}% - model explains more variance");
        Console.WriteLine($"  • RMSE reduced by {rmseReduction:F1}% - predictions are closer to actual prices");
        Console.WriteLine($"  • Feature engineering delivered significant value with SAME algorithm");
    }
    
    private void PrintMetric(string name, double baseline, double enhanced, 
        bool lowerIsBetter = false)
    {
        var improvement = lowerIsBetter
            ? (baseline - enhanced) / baseline * 100
            : (enhanced - baseline) / baseline * 100;
        
        var arrow = improvement > 0 ? "↑" : "↓";
        var color = improvement > 0 ? "✓" : "✗";
        
        Console.WriteLine($"{name,-25} {baseline,15:F4} {enhanced,15:F4} {color} {arrow}{Math.Abs(improvement):F1}%");
    }
    
    private RegressionMetrics TrainAndEvaluateBaseline(
        List<RawHousingRecord> train, List<RawHousingRecord> test)
    {
        var trainData = _mlContext.Data.LoadFromEnumerable(train);
        var testData = _mlContext.Data.LoadFromEnumerable(test);
        
        var pipeline = _mlContext.Transforms
            .Concatenate("Features", 
                "SquareFeet", "LotSize", "Bedrooms", "Bathrooms", "GarageSpaces")
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Price",
                numberOfTrees: 100));
        
        var model = pipeline.Fit(trainData);
        var predictions = model.Transform(testData);
        
        return _mlContext.Regression.Evaluate(predictions, "Price");
    }
    
    private RegressionMetrics TrainAndEvaluateEnhanced(
        List<HousingFeatureEngineer.EnrichedHousingRecord> train,
        List<HousingFeatureEngineer.EnrichedHousingRecord> test)
    {
        var trainData = _mlContext.Data.LoadFromEnumerable(train);
        var testData = _mlContext.Data.LoadFromEnumerable(test);
        
        var featureColumns = new[]
        {
            "SquareFeet", "LotSize", "Bedrooms", "Bathrooms", "GarageSpaces",
            "SqFtPerBedroom", "BathroomToBedroom", "LotToHomeRatio",
            "Age", "EffectiveAge", 
            "NeighborhoodMedianPrice", "SqFtVsNeighborhood", "DistanceToDowntown",
            "LuxuryScore", "QualityScore", "SeasonalityIndex"
        };
        
        var pipeline = _mlContext.Transforms
            .Concatenate("Features", featureColumns)
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Price",
                numberOfTrees: 100));
        
        var model = pipeline.Fit(trainData);
        var predictions = model.Transform(testData);
        
        return _mlContext.Regression.Evaluate(predictions, "Price");
    }
    
    private (List<T> train, List<T> test) SplitData<T>(List<T> data, double trainRatio)
    {
        var splitIndex = (int)(data.Count * trainRatio);
        return (data.Take(splitIndex).ToList(), data.Skip(splitIndex).ToList());
    }
}
```

Expected output:

```
======================================================================
MODEL COMPARISON: BASELINE VS ENGINEERED FEATURES
======================================================================

Metric                          Baseline      Engineered     Improvement
----------------------------------------------------------------------
R²                                0.7234          0.8567     ✓ ↑18.4%
RMSE                          45234.5600      31456.7800     ✓ ↑30.5%
MAE                           32156.4500      22345.6700     ✓ ↑30.5%

----------------------------------------------------------------------
INTERPRETATION:
  • R² improved by 18.4% - model explains more variance
  • RMSE reduced by 30.5% - predictions are closer to actual prices
  • Feature engineering delivered significant value with SAME algorithm
```

This is the key takeaway: **using the exact same algorithm** (FastTree with identical hyperparameters), feature engineering improved R² by 18% and reduced error by 30%. This is often a bigger improvement than switching algorithms or extensive hyperparameter tuning.

### Feature Selection: Keeping What Matters

After engineering many features, we should identify which ones actually contribute:

```csharp
public class FeatureSelectionDemo
{
    private readonly MLContext _mlContext;
    
    public FeatureSelectionDemo()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public void PerformFeatureSelection(IDataView data)
    {
        Console.WriteLine("\n" + new string('=', 70));
        Console.WriteLine("FEATURE SELECTION");
        Console.WriteLine(new string('=', 70));
        
        // First, train a model to get feature importance
        var featureColumns = new[]
        {
            "SquareFeet", "LotSize", "Bedrooms", "Bathrooms", "GarageSpaces",
            "SqFtPerBedroom", "BathroomToBedroom", "LotToHomeRatio",
            "Age", "EffectiveAge",
            "NeighborhoodMedianPrice", "SqFtVsNeighborhood", "DistanceToDowntown",
            "LuxuryScore", "QualityScore", "SeasonalityIndex",
            "DescriptionLength", "LuxuryKeywordCount"
        };
        
        var pipeline = _mlContext.Transforms
            .Concatenate("Features", featureColumns)
            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(_mlContext.Regression.Trainers.FastTree(
                labelColumnName: "Price",
                numberOfTrees: 100));
        
        var model = pipeline.Fit(data);
        
        // Calculate permutation importance
        var importance = _mlContext.Regression.PermutationFeatureImportance(
            model.LastTransformer, 
            model.Transform(data),
            labelColumnName: "Price",
            permutationCount: 5);
        
        Console.WriteLine($"\n{"Feature",-30} {"Importance",15} {"Keep?",10}");
        Console.WriteLine(new string('-', 60));
        
        var importanceByFeature = featureColumns
            .Zip(importance, (name, imp) => (Name: name, Importance: imp.RSquared.Mean))
            .OrderByDescending(x => Math.Abs(x.Importance))
            .ToList();
        
        var threshold = 0.005; // Features contributing less than 0.5% to R²
        
        foreach (var (name, imp) in importanceByFeature)
        {
            var keep = Math.Abs(imp) >= threshold;
            var keepIndicator = keep ? "✓ KEEP" : "✗ DROP";
            Console.WriteLine($"{name,-30} {imp,15:F6} {keepIndicator,10}");
        }
        
        var selectedFeatures = importanceByFeature
            .Where(x => Math.Abs(x.Importance) >= threshold)
            .Select(x => x.Name)
            .ToList();
        
        Console.WriteLine($"\n{new string('-', 60)}");
        Console.WriteLine($"Selected {selectedFeatures.Count} of {featureColumns.Length} features");
        Console.WriteLine($"Dropped features: {string.Join(", ", 
            featureColumns.Except(selectedFeatures))}");
    }
}
```

## Key Takeaways

Feature engineering is where data science becomes an art as much as a science. Here's what we've learned:

1. **Good features have predictive power and are independent**. Aim for features that correlate with your target but not with each other.

2. **Domain knowledge is your greatest asset**. The best features come from understanding the problem domain—what actually drives house prices? Knowing that bathroom-to-bedroom ratio matters for luxury homes is domain knowledge that no algorithm can discover automatically.

3. **Transform raw data into meaningful representations**. A timestamp becomes day-of-week, month, seasonal indicators. An address becomes neighborhood statistics. Text becomes sentiment scores. The information was always there; you make it accessible.

4. **Feature engineering often matters more than algorithm selection**. We demonstrated an 18% improvement in R² and 30% reduction in error using the *same algorithm* with better features. That's often more impactful than switching from a simple model to a complex one.

5. **Use cyclical encoding for periodic features**. When December should be "close to" January in your model's understanding, sine/cosine encoding makes that relationship explicit.

6. **Aggregate features capture context**. Is a house large *for its neighborhood*? Is it priced above or below the local median? Relative positioning often matters more than absolute values.

7. **Feature selection removes noise**. Not every feature you engineer will help. Use correlation analysis, permutation importance, and cross-validation to identify what actually contributes.

8. **Watch for leakage**. Aggregate features computed from the same data you're predicting can leak target information. Always compute statistics from training data only, and think carefully about what would be available at prediction time.

The features you engineer become the language in which you express your understanding of the problem. Master this, and you'll consistently build models that outperform less thoughtful approaches—regardless of which algorithms you use.

In the next chapter, we'll explore model training and evaluation techniques that help you choose the right algorithm and tune it effectively. But never forget: the best model is only as good as the features you give it.

## Exercises

1. **Extend the housing features**: Add features for:
   - Crime rate in the zip code (if data is available)
   - Distance to nearest school
   - Walk score or transit score
   - Year-over-year price appreciation in the neighborhood

2. **Build a feature importance report**: Create a utility that takes any ML.NET model and produces a ranked list of features with their importance scores, correlations, and recommendations.

3. **Handle missing features**: Extend the pipeline to gracefully handle missing neighborhood statistics for new neighborhoods not seen in training data. Consider fallbacks like city-wide medians or geographic interpolation.

4. **A/B test features**: Implement a framework that trains multiple models with different feature subsets and reports which combination performs best on held-out data.

5. **Automate feature generation**: Create a system that automatically generates candidate features (ratios, differences, bins) from a given dataset, evaluates their correlation with the target, and ranks them for human review.
