// Chapter 6: Feature Engineering for Housing Price Prediction
// Demonstrates creating powerful features that improve model accuracy

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.Analysis;
using System.Text.RegularExpressions;

Console.WriteLine("=".PadRight(70, '='));
Console.WriteLine("Chapter 6: Feature Engineering - Housing Price Prediction");
Console.WriteLine("=".PadRight(70, '='));

// ============================================================================
// SAMPLE DATA - Embedded housing dataset
// ============================================================================

var housingData = new List<HousingRecord>
{
    // Format: Id, Rooms, Bedrooms, Population, Households, MedianIncome, HouseAge, 
    //         ListingDate, Description, MedianPrice
    new(1, 6.0f, 2.0f, 1500, 400, 4.5f, 25, "2024-03-15", "Charming 2BR home with modern kitchen and hardwood floors", 285000),
    new(2, 8.0f, 4.0f, 2200, 550, 6.2f, 10, "2024-06-20", "Spacious family home with pool and 3-car garage, newly renovated", 520000),
    new(3, 5.0f, 2.0f, 1800, 480, 3.8f, 45, "2024-01-10", "Cozy starter home near schools", 195000),
    new(4, 7.0f, 3.0f, 1200, 300, 5.5f, 15, "2024-07-04", "Modern 3BR with ocean view and smart home features", 425000),
    new(5, 4.0f, 1.0f, 900, 250, 2.9f, 60, "2023-11-22", "Studio condo downtown", 145000),
    new(6, 9.0f, 4.0f, 2500, 600, 7.1f, 5, "2024-08-30", "Luxury estate with wine cellar, home theater, and guest house", 780000),
    new(7, 6.0f, 3.0f, 1600, 420, 4.2f, 30, "2024-02-28", "Well-maintained 3BR with large backyard and updated bathrooms", 310000),
    new(8, 5.0f, 2.0f, 1400, 380, 3.5f, 40, "2024-04-12", "Affordable 2BR fixer-upper with potential", 175000),
    new(9, 7.0f, 3.0f, 1900, 500, 5.8f, 8, "2024-09-05", "New construction with granite counters and stainless appliances", 465000),
    new(10, 8.0f, 4.0f, 2100, 520, 6.5f, 12, "2024-05-18", "Executive home with home office, gym, and solar panels installed", 545000),
    new(11, 4.0f, 1.0f, 800, 220, 3.1f, 55, "2023-12-01", "Compact 1BR in quiet neighborhood", 155000),
    new(12, 6.0f, 2.0f, 1300, 350, 4.8f, 20, "2024-03-25", "Updated 2BR with new roof and HVAC system", 295000),
    new(13, 10.0f, 5.0f, 2800, 680, 8.2f, 3, "2024-10-01", "Brand new mansion with infinity pool, 5BR suites, and smart automation", 950000),
    new(14, 5.0f, 2.0f, 1100, 290, 3.9f, 35, "2024-01-30", "Classic 2BR bungalow with original character", 225000),
    new(15, 7.0f, 3.0f, 1700, 440, 5.2f, 18, "2024-06-08", "Renovated 3BR with open floor plan and deck", 385000),
    new(16, 6.0f, 2.0f, 1450, 390, 4.3f, 28, "2024-04-20", "Move-in ready 2BR with garage", 275000),
    new(17, 8.0f, 3.0f, 2000, 510, 5.9f, 14, "2024-07-15", "Elegant 3BR with fireplace and chef's kitchen", 475000),
    new(18, 5.0f, 2.0f, 1250, 330, 3.6f, 42, "2024-02-14", "Cozy cottage style home needs TLC", 185000),
    new(19, 9.0f, 4.0f, 2400, 580, 7.5f, 6, "2024-08-22", "Modern farmhouse with barn and acreage, premium finishes throughout", 695000),
    new(20, 6.0f, 3.0f, 1550, 410, 4.6f, 22, "2024-05-03", "Family-friendly 3BR near parks and shopping", 325000),
};

Console.WriteLine($"\n📊 Loaded {housingData.Count} housing records\n");

// ============================================================================
// PART 1: RATIO FEATURES
// ============================================================================

Console.WriteLine("-".PadRight(70, '-'));
Console.WriteLine("PART 1: Creating Ratio Features");
Console.WriteLine("-".PadRight(70, '-'));

Console.WriteLine("\nRatio features capture relationships between variables that raw");
Console.WriteLine("features miss. They're often more predictive than the originals.\n");

foreach (var house in housingData)
{
    // Rooms per household - indicates home size relative to area density
    house.RoomsPerHousehold = house.TotalRooms / house.Households;
    
    // Bedrooms per room - indicates how much space is sleeping vs living
    house.BedroomsPerRoom = house.TotalBedrooms / house.TotalRooms;
    
    // Population per household - indicates crowding/family size
    house.PopulationPerHousehold = (float)house.Population / house.Households;
    
    // Income per room - proxy for affluence of the area
    house.IncomePerRoom = house.MedianIncome / house.TotalRooms;
}

Console.WriteLine("Sample ratio features computed:\n");
Console.WriteLine($"{"ID",-4} {"Rooms/HH",10} {"BR/Room",10} {"Pop/HH",10} {"Inc/Room",10}");
Console.WriteLine(new string('-', 46));
foreach (var h in housingData.Take(5))
{
    Console.WriteLine($"{h.Id,-4} {h.RoomsPerHousehold,10:F3} {h.BedroomsPerRoom,10:F3} {h.PopulationPerHousehold,10:F3} {h.IncomePerRoom,10:F3}");
}

// ============================================================================
// PART 2: BINNING CONTINUOUS VARIABLES
// ============================================================================

Console.WriteLine("\n" + "-".PadRight(70, '-'));
Console.WriteLine("PART 2: Binning Continuous Variables");
Console.WriteLine("-".PadRight(70, '-'));

Console.WriteLine("\nBinning converts continuous values into categories, which can");
Console.WriteLine("capture non-linear relationships and reduce noise.\n");

foreach (var house in housingData)
{
    // Age group binning
    house.AgeGroup = house.HouseAge switch
    {
        <= 5 => "New (0-5)",
        <= 15 => "Recent (6-15)",
        <= 30 => "Established (16-30)",
        <= 50 => "Mature (31-50)",
        _ => "Vintage (50+)"
    };
    
    // Price category binning
    house.PriceCategory = house.MedianPrice switch
    {
        < 200000 => "Budget",
        < 350000 => "Mid-Range",
        < 500000 => "Premium",
        < 750000 => "Luxury",
        _ => "Ultra-Luxury"
    };
    
    // Income bracket
    house.IncomeBracket = house.MedianIncome switch
    {
        < 3.5f => "Low",
        < 5.0f => "Middle",
        < 6.5f => "Upper-Middle",
        _ => "High"
    };
}

Console.WriteLine("Age and price category distribution:\n");
var ageGroups = housingData.GroupBy(h => h.AgeGroup).OrderBy(g => g.Key);
Console.WriteLine("Age Groups:");
foreach (var group in ageGroups)
{
    var bar = new string('█', group.Count() * 2);
    Console.WriteLine($"  {group.Key,-20} {bar} ({group.Count()})");
}

Console.WriteLine("\nPrice Categories:");
var priceCategories = housingData.GroupBy(h => h.PriceCategory)
    .OrderBy(g => g.First().MedianPrice);
foreach (var group in priceCategories)
{
    var bar = new string('█', group.Count() * 2);
    Console.WriteLine($"  {group.Key,-15} {bar} ({group.Count()})");
}

// ============================================================================
// PART 3: DATE/TIME FEATURE EXTRACTION
// ============================================================================

Console.WriteLine("\n" + "-".PadRight(70, '-'));
Console.WriteLine("PART 3: Date/Time Feature Extraction");
Console.WriteLine("-".PadRight(70, '-'));

Console.WriteLine("\nTemporal features can reveal seasonal patterns, market trends,");
Console.WriteLine("and cyclical behaviors that affect pricing.\n");

foreach (var house in housingData)
{
    var listingDate = DateTime.Parse(house.ListingDate);
    
    // Extract temporal components
    house.ListingMonth = listingDate.Month;
    house.ListingQuarter = (listingDate.Month - 1) / 3 + 1;
    house.ListingDayOfWeek = (int)listingDate.DayOfWeek;
    house.IsWeekendListing = listingDate.DayOfWeek is DayOfWeek.Saturday or DayOfWeek.Sunday;
    
    // Seasonal indicator
    house.Season = listingDate.Month switch
    {
        >= 3 and <= 5 => "Spring",
        >= 6 and <= 8 => "Summer",
        >= 9 and <= 11 => "Fall",
        _ => "Winter"
    };
    
    // Days since reference date (for trend analysis)
    var referenceDate = new DateTime(2023, 1, 1);
    house.DaysSinceReference = (listingDate - referenceDate).Days;
}

Console.WriteLine("Temporal features extracted:\n");
Console.WriteLine($"{"ID",-4} {"Date",-12} {"Month",6} {"Qtr",4} {"Season",-8} {"Weekend",8}");
Console.WriteLine(new string('-', 50));
foreach (var h in housingData.Take(6))
{
    Console.WriteLine($"{h.Id,-4} {h.ListingDate,-12} {h.ListingMonth,6} {h.ListingQuarter,4} {h.Season,-8} {h.IsWeekendListing,8}");
}

Console.WriteLine("\nSeasonal listing distribution:");
var seasons = housingData.GroupBy(h => h.Season);
foreach (var season in seasons.OrderBy(s => s.Key))
{
    var avgPrice = season.Average(h => h.MedianPrice);
    Console.WriteLine($"  {season.Key,-8}: {season.Count(),2} listings, avg price ${avgPrice:N0}");
}

// ============================================================================
// PART 4: TEXT FEATURE EXTRACTION
// ============================================================================

Console.WriteLine("\n" + "-".PadRight(70, '-'));
Console.WriteLine("PART 4: Text Feature Extraction");
Console.WriteLine("-".PadRight(70, '-'));

Console.WriteLine("\nText descriptions contain valuable signals. We extract structured");
Console.WriteLine("features from unstructured text to capture property attributes.\n");

// Define keyword sets for feature extraction
var luxuryKeywords = new[] { "luxury", "estate", "mansion", "premium", "elegant", "executive", "infinity", "wine cellar", "theater", "smart" };
var amenityKeywords = new[] { "pool", "garage", "fireplace", "deck", "patio", "gym", "office", "solar" };
var conditionKeywords = new[] { "new", "renovated", "updated", "modern", "brand new" };
var issueKeywords = new[] { "fixer", "tlc", "potential", "needs" };

foreach (var house in housingData)
{
    var descLower = house.Description.ToLower();
    
    // Word count (longer descriptions often mean more features)
    house.DescriptionWordCount = house.Description.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
    
    // Keyword presence features
    house.HasLuxuryKeywords = luxuryKeywords.Any(k => descLower.Contains(k));
    house.HasAmenityKeywords = amenityKeywords.Any(k => descLower.Contains(k));
    house.HasConditionKeywords = conditionKeywords.Any(k => descLower.Contains(k));
    house.HasIssueKeywords = issueKeywords.Any(k => descLower.Contains(k));
    
    // Count of luxury keywords (intensity measure)
    house.LuxuryScore = luxuryKeywords.Count(k => descLower.Contains(k));
    house.AmenityCount = amenityKeywords.Count(k => descLower.Contains(k));
    
    // Sentiment proxy: exclamation marks and superlatives
    house.HasSuperlatives = Regex.IsMatch(descLower, @"\b(best|amazing|stunning|gorgeous|perfect|beautiful)\b");
}

Console.WriteLine("Text features extracted:\n");
Console.WriteLine($"{"ID",-4} {"Words",6} {"Luxury",7} {"Amenity",8} {"Issues",7} {"LuxScore",9}");
Console.WriteLine(new string('-', 48));
foreach (var h in housingData.Take(8))
{
    Console.WriteLine($"{h.Id,-4} {h.DescriptionWordCount,6} {h.HasLuxuryKeywords,7} {h.HasAmenityKeywords,8} {h.HasIssueKeywords,7} {h.LuxuryScore,9}");
}

Console.WriteLine("\nPrice comparison by text features:");
var withLuxury = housingData.Where(h => h.HasLuxuryKeywords).ToList();
var withoutLuxury = housingData.Where(h => !h.HasLuxuryKeywords).ToList();
Console.WriteLine($"  With luxury keywords:    ${withLuxury.Average(h => h.MedianPrice):N0} avg ({withLuxury.Count} homes)");
Console.WriteLine($"  Without luxury keywords: ${withoutLuxury.Average(h => h.MedianPrice):N0} avg ({withoutLuxury.Count} homes)");

var withIssues = housingData.Where(h => h.HasIssueKeywords).ToList();
var withoutIssues = housingData.Where(h => !h.HasIssueKeywords).ToList();
Console.WriteLine($"  With issue keywords:     ${withIssues.Average(h => h.MedianPrice):N0} avg ({withIssues.Count} homes)");
Console.WriteLine($"  Without issue keywords:  ${withoutIssues.Average(h => h.MedianPrice):N0} avg ({withoutIssues.Count} homes)");

// ============================================================================
// PART 5: FEATURE SELECTION VIA CORRELATION
// ============================================================================

Console.WriteLine("\n" + "-".PadRight(70, '-'));
Console.WriteLine("PART 5: Feature Selection via Correlation Analysis");
Console.WriteLine("-".PadRight(70, '-'));

Console.WriteLine("\nNot all features are equally useful. Correlation analysis helps");
Console.WriteLine("identify which features have the strongest relationship with price.\n");

// Calculate correlations with price
var correlations = new Dictionary<string, double>
{
    ["MedianIncome"] = CalculateCorrelation(housingData.Select(h => (double)h.MedianIncome), housingData.Select(h => (double)h.MedianPrice)),
    ["TotalRooms"] = CalculateCorrelation(housingData.Select(h => (double)h.TotalRooms), housingData.Select(h => (double)h.MedianPrice)),
    ["TotalBedrooms"] = CalculateCorrelation(housingData.Select(h => (double)h.TotalBedrooms), housingData.Select(h => (double)h.MedianPrice)),
    ["HouseAge"] = CalculateCorrelation(housingData.Select(h => (double)h.HouseAge), housingData.Select(h => (double)h.MedianPrice)),
    ["Population"] = CalculateCorrelation(housingData.Select(h => (double)h.Population), housingData.Select(h => (double)h.MedianPrice)),
    ["Households"] = CalculateCorrelation(housingData.Select(h => (double)h.Households), housingData.Select(h => (double)h.MedianPrice)),
    // Engineered features
    ["RoomsPerHousehold"] = CalculateCorrelation(housingData.Select(h => (double)h.RoomsPerHousehold), housingData.Select(h => (double)h.MedianPrice)),
    ["BedroomsPerRoom"] = CalculateCorrelation(housingData.Select(h => (double)h.BedroomsPerRoom), housingData.Select(h => (double)h.MedianPrice)),
    ["PopPerHousehold"] = CalculateCorrelation(housingData.Select(h => (double)h.PopulationPerHousehold), housingData.Select(h => (double)h.MedianPrice)),
    ["IncomePerRoom"] = CalculateCorrelation(housingData.Select(h => (double)h.IncomePerRoom), housingData.Select(h => (double)h.MedianPrice)),
    ["LuxuryScore"] = CalculateCorrelation(housingData.Select(h => (double)h.LuxuryScore), housingData.Select(h => (double)h.MedianPrice)),
    ["AmenityCount"] = CalculateCorrelation(housingData.Select(h => (double)h.AmenityCount), housingData.Select(h => (double)h.MedianPrice)),
    ["DescWordCount"] = CalculateCorrelation(housingData.Select(h => (double)h.DescriptionWordCount), housingData.Select(h => (double)h.MedianPrice)),
};

Console.WriteLine("Feature correlations with MedianPrice:\n");
Console.WriteLine($"{"Feature",-20} {"Correlation",12} {"Strength",-15}");
Console.WriteLine(new string('-', 50));

foreach (var (feature, corr) in correlations.OrderByDescending(c => Math.Abs(c.Value)))
{
    var strength = Math.Abs(corr) switch
    {
        >= 0.8 => "Very Strong",
        >= 0.6 => "Strong",
        >= 0.4 => "Moderate",
        >= 0.2 => "Weak",
        _ => "Very Weak"
    };
    
    var bar = new string(corr >= 0 ? '+' : '-', (int)(Math.Abs(corr) * 20));
    Console.WriteLine($"{feature,-20} {corr,12:F4} {strength,-15} {bar}");
}

Console.WriteLine("\n✨ Notice how engineered features often have stronger correlations!");

// ============================================================================
// PART 6: MODEL COMPARISON - BEFORE vs AFTER FEATURE ENGINEERING
// ============================================================================

Console.WriteLine("\n" + "-".PadRight(70, '-'));
Console.WriteLine("PART 6: Model Comparison - Before vs After Feature Engineering");
Console.WriteLine("-".PadRight(70, '-'));

Console.WriteLine("\nThe ultimate test: do engineered features improve model accuracy?");
Console.WriteLine("We'll train two models and compare their performance.\n");

var mlContext = new MLContext(seed: 42);

// Prepare data for ML.NET
var mlData = housingData.Select(h => new HousingMLData
{
    // Original features
    TotalRooms = h.TotalRooms,
    TotalBedrooms = h.TotalBedrooms,
    Population = h.Population,
    Households = h.Households,
    MedianIncome = h.MedianIncome,
    HouseAge = h.HouseAge,
    
    // Engineered features
    RoomsPerHousehold = h.RoomsPerHousehold,
    BedroomsPerRoom = h.BedroomsPerRoom,
    PopulationPerHousehold = h.PopulationPerHousehold,
    IncomePerRoom = h.IncomePerRoom,
    LuxuryScore = h.LuxuryScore,
    AmenityCount = h.AmenityCount,
    DescriptionWordCount = h.DescriptionWordCount,
    HasLuxuryKeywords = h.HasLuxuryKeywords ? 1f : 0f,
    HasIssueKeywords = h.HasIssueKeywords ? 1f : 0f,
    ListingQuarter = h.ListingQuarter,
    
    // Target
    Price = h.MedianPrice
}).ToList();

var dataView = mlContext.Data.LoadFromEnumerable(mlData);

// Split data
var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.25);

// MODEL 1: Original features only
Console.WriteLine("Training Model 1 (Original Features Only)...");
var originalFeatures = new[] { "TotalRooms", "TotalBedrooms", "Population", "Households", "MedianIncome", "HouseAge" };

var pipeline1 = mlContext.Transforms.Concatenate("Features", originalFeatures)
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

var model1 = pipeline1.Fit(split.TrainSet);
var predictions1 = model1.Transform(split.TestSet);
var metrics1 = mlContext.Regression.Evaluate(predictions1, labelColumnName: "Price");

// MODEL 2: With engineered features
Console.WriteLine("Training Model 2 (With Engineered Features)...");
var engineeredFeatures = new[] 
{ 
    "TotalRooms", "TotalBedrooms", "Population", "Households", "MedianIncome", "HouseAge",
    "RoomsPerHousehold", "BedroomsPerRoom", "PopulationPerHousehold", "IncomePerRoom",
    "LuxuryScore", "AmenityCount", "DescriptionWordCount", "HasLuxuryKeywords", "HasIssueKeywords",
    "ListingQuarter"
};

var pipeline2 = mlContext.Transforms.Concatenate("Features", engineeredFeatures)
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", maximumNumberOfIterations: 100));

var model2 = pipeline2.Fit(split.TrainSet);
var predictions2 = model2.Transform(split.TestSet);
var metrics2 = mlContext.Regression.Evaluate(predictions2, labelColumnName: "Price");

// Compare results
Console.WriteLine("\n" + "=".PadRight(70, '='));
Console.WriteLine("MODEL COMPARISON RESULTS");
Console.WriteLine("=".PadRight(70, '='));

Console.WriteLine($"\n{"Metric",-25} {"Original",15} {"Engineered",15} {"Improvement",15}");
Console.WriteLine(new string('-', 72));

var r2Improvement = ((metrics2.RSquared - metrics1.RSquared) / Math.Abs(metrics1.RSquared)) * 100;
var rmseImprovement = ((metrics1.RootMeanSquaredError - metrics2.RootMeanSquaredError) / metrics1.RootMeanSquaredError) * 100;
var maeImprovement = ((metrics1.MeanAbsoluteError - metrics2.MeanAbsoluteError) / metrics1.MeanAbsoluteError) * 100;

Console.WriteLine($"{"R² Score",-25} {metrics1.RSquared,15:F4} {metrics2.RSquared,15:F4} {r2Improvement,14:F1}%");
Console.WriteLine($"{"RMSE",-25} {metrics1.RootMeanSquaredError,15:F2} {metrics2.RootMeanSquaredError,15:F2} {rmseImprovement,14:F1}%");
Console.WriteLine($"{"MAE",-25} {metrics1.MeanAbsoluteError,15:F2} {metrics2.MeanAbsoluteError,15:F2} {maeImprovement,14:F1}%");
Console.WriteLine($"{"Feature Count",-25} {originalFeatures.Length,15} {engineeredFeatures.Length,15}");

Console.WriteLine("\n" + "=".PadRight(70, '='));
Console.WriteLine("KEY TAKEAWAYS");
Console.WriteLine("=".PadRight(70, '='));

Console.WriteLine(@"
  📈 Feature engineering can significantly improve model performance
  
  🔧 Ratio features often capture relationships better than raw values
  
  📊 Binning helps models handle non-linear patterns
  
  📅 Temporal features reveal seasonal and trend patterns
  
  📝 Text mining extracts structured signals from descriptions
  
  🎯 Correlation analysis helps select the most predictive features
  
  ⚡ Even simple engineered features can boost accuracy substantially
");

Console.WriteLine("Run complete! ✅\n");

// ============================================================================
// HELPER METHODS
// ============================================================================

static double CalculateCorrelation(IEnumerable<double> x, IEnumerable<double> y)
{
    var xList = x.ToList();
    var yList = y.ToList();
    
    var n = xList.Count;
    var sumX = xList.Sum();
    var sumY = yList.Sum();
    var sumXY = xList.Zip(yList, (a, b) => a * b).Sum();
    var sumX2 = xList.Sum(a => a * a);
    var sumY2 = yList.Sum(b => b * b);
    
    var numerator = n * sumXY - sumX * sumY;
    var denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator == 0 ? 0 : numerator / denominator;
}

// ============================================================================
// DATA CLASSES
// ============================================================================

class HousingRecord
{
    public int Id { get; set; }
    public float TotalRooms { get; set; }
    public float TotalBedrooms { get; set; }
    public int Population { get; set; }
    public int Households { get; set; }
    public float MedianIncome { get; set; }
    public int HouseAge { get; set; }
    public string ListingDate { get; set; } = "";
    public string Description { get; set; } = "";
    public float MedianPrice { get; set; }
    
    // Ratio features
    public float RoomsPerHousehold { get; set; }
    public float BedroomsPerRoom { get; set; }
    public float PopulationPerHousehold { get; set; }
    public float IncomePerRoom { get; set; }
    
    // Binned features
    public string AgeGroup { get; set; } = "";
    public string PriceCategory { get; set; } = "";
    public string IncomeBracket { get; set; } = "";
    
    // Temporal features
    public int ListingMonth { get; set; }
    public int ListingQuarter { get; set; }
    public int ListingDayOfWeek { get; set; }
    public bool IsWeekendListing { get; set; }
    public string Season { get; set; } = "";
    public int DaysSinceReference { get; set; }
    
    // Text features
    public int DescriptionWordCount { get; set; }
    public bool HasLuxuryKeywords { get; set; }
    public bool HasAmenityKeywords { get; set; }
    public bool HasConditionKeywords { get; set; }
    public bool HasIssueKeywords { get; set; }
    public int LuxuryScore { get; set; }
    public int AmenityCount { get; set; }
    public bool HasSuperlatives { get; set; }
    
    public HousingRecord(int id, float rooms, float bedrooms, int pop, int hh, 
                         float income, int age, string date, string desc, float price)
    {
        Id = id;
        TotalRooms = rooms;
        TotalBedrooms = bedrooms;
        Population = pop;
        Households = hh;
        MedianIncome = income;
        HouseAge = age;
        ListingDate = date;
        Description = desc;
        MedianPrice = price;
    }
}

class HousingMLData
{
    // Original features
    public float TotalRooms { get; set; }
    public float TotalBedrooms { get; set; }
    public float Population { get; set; }
    public float Households { get; set; }
    public float MedianIncome { get; set; }
    public float HouseAge { get; set; }
    
    // Engineered features
    public float RoomsPerHousehold { get; set; }
    public float BedroomsPerRoom { get; set; }
    public float PopulationPerHousehold { get; set; }
    public float IncomePerRoom { get; set; }
    public float LuxuryScore { get; set; }
    public float AmenityCount { get; set; }
    public float DescriptionWordCount { get; set; }
    public float HasLuxuryKeywords { get; set; }
    public float HasIssueKeywords { get; set; }
    public float ListingQuarter { get; set; }
    
    // Target
    public float Price { get; set; }
}
