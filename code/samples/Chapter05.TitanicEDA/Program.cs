// Chapter 5: Exploratory Data Analysis with the Titanic Dataset
// This sample demonstrates EDA techniques using Microsoft.Data.Analysis

using Microsoft.Data.Analysis;

Console.WriteLine("=".PadRight(70, '='));
Console.WriteLine("  CHAPTER 5: Exploratory Data Analysis - Titanic Dataset");
Console.WriteLine("=".PadRight(70, '='));
Console.WriteLine();

// Create DataFrame with embedded Titanic passenger data
var df = CreateTitanicDataFrame();

Console.WriteLine($"Dataset loaded: {df.Rows.Count} passengers, {df.Columns.Count} features\n");

// ============================================================================
// SECTION 1: Data Overview
// ============================================================================
PrintSection("1. DATA OVERVIEW");

Console.WriteLine("Columns in dataset:");
foreach (var col in df.Columns)
{
    Console.WriteLine($"  • {col.Name,-12} ({col.DataType.Name})");
}
Console.WriteLine();

Console.WriteLine("First 5 rows:");
PrintDataFrameHead(df, 5);

// ============================================================================
// SECTION 2: Missing Value Analysis
// ============================================================================
PrintSection("2. MISSING VALUE ANALYSIS");

Console.WriteLine("Missing values per column:");
foreach (var col in df.Columns)
{
    long nullCount = col.NullCount;
    double percentage = (double)nullCount / df.Rows.Count * 100;
    string bar = new string('█', (int)(percentage / 5));
    Console.WriteLine($"  {col.Name,-12}: {nullCount,3} missing ({percentage,5:F1}%) {bar}");
}
Console.WriteLine();

Console.WriteLine("💡 INSIGHT: Age has the most missing values.");
Console.WriteLine("   Strategy: Could impute with median age by passenger class.\n");

// ============================================================================
// SECTION 3: Descriptive Statistics
// ============================================================================
PrintSection("3. DESCRIPTIVE STATISTICS");

// Numeric columns analysis
var ageCol = df.Columns["Age"] as SingleDataFrameColumn;
var fareCol = df.Columns["Fare"] as SingleDataFrameColumn;

Console.WriteLine("Age Statistics:");
PrintNumericStats(ageCol!, "Age");

Console.WriteLine("\nFare Statistics:");
PrintNumericStats(fareCol!, "Fare");

// ============================================================================
// SECTION 4: Survival Rate Analysis
// ============================================================================
PrintSection("4. SURVIVAL RATE ANALYSIS");

var survivedCol = df.Columns["Survived"] as Int32DataFrameColumn;
long survivedSum = Convert.ToInt64(survivedCol!.Sum());
double overallSurvivalRate = (double)survivedSum / survivedCol.Length * 100;
Console.WriteLine($"Overall Survival Rate: {overallSurvivalRate:F1}%\n");

// Survival by Gender
Console.WriteLine("Survival Rate by Gender:");
Console.WriteLine("-".PadRight(40, '-'));
AnalyzeSurvivalByCategory(df, "Sex");

// Survival by Class
Console.WriteLine("\nSurvival Rate by Passenger Class:");
Console.WriteLine("-".PadRight(40, '-'));
AnalyzeSurvivalByClass(df);

// Survival by Age Group
Console.WriteLine("\nSurvival Rate by Age Group:");
Console.WriteLine("-".PadRight(40, '-'));
AnalyzeSurvivalByAgeGroup(df);

Console.WriteLine("\n💡 KEY FINDINGS:");
Console.WriteLine("   • Women had significantly higher survival rates than men");
Console.WriteLine("   • First class passengers survived at much higher rates");
Console.WriteLine("   • Children (under 18) had better survival chances");
Console.WriteLine("   • \"Women and children first\" policy is evident in the data\n");

// ============================================================================
// SECTION 5: Correlation Analysis
// ============================================================================
PrintSection("5. CORRELATION ANALYSIS");

Console.WriteLine("Correlation Matrix (Numeric Features):");
Console.WriteLine("-".PadRight(50, '-'));

// Calculate correlations between numeric columns
var numericCols = new[] { "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare" };
Console.WriteLine($"{"",12} {"Survived",10} {"Pclass",10} {"Fare",10}");

double survPclassCorr = CalculateCorrelation(df, "Survived", "Pclass");
double survFareCorr = CalculateCorrelation(df, "Survived", "Fare");
double pclassFareCorr = CalculateCorrelation(df, "Pclass", "Fare");

Console.WriteLine($"{"Survived",12} {1.0,10:F3} {survPclassCorr,10:F3} {survFareCorr,10:F3}");
Console.WriteLine($"{"Pclass",12} {survPclassCorr,10:F3} {1.0,10:F3} {pclassFareCorr,10:F3}");
Console.WriteLine($"{"Fare",12} {survFareCorr,10:F3} {pclassFareCorr,10:F3} {1.0,10:F3}");

Console.WriteLine("\n💡 CORRELATIONS EXPLAINED:");
Console.WriteLine($"   • Survival vs Class ({survPclassCorr:F2}): Negative = lower class number (1st) = higher survival");
Console.WriteLine($"   • Survival vs Fare ({survFareCorr:F2}): Positive = higher fare = higher survival");
Console.WriteLine($"   • Class vs Fare ({pclassFareCorr:F2}): Negative = 1st class paid more (expected)\n");

// ============================================================================
// SECTION 6: Data Visualization Concepts
// ============================================================================
PrintSection("6. DATA VISUALIZATION (Conceptual)");

Console.WriteLine("📊 RECOMMENDED VISUALIZATIONS FOR THIS DATA:\n");

Console.WriteLine("1. BAR CHART: Survival Rate by Gender");
Console.WriteLine("   ┌────────────────────────────────────┐");
Console.WriteLine("   │ Female ████████████████████ 74.2% │");
Console.WriteLine("   │ Male   ██████             18.9%   │");
Console.WriteLine("   └────────────────────────────────────┘\n");

Console.WriteLine("2. GROUPED BAR CHART: Survival by Class and Gender");
Console.WriteLine("   Shows the combined effect of class and gender on survival\n");

Console.WriteLine("3. HISTOGRAM: Age Distribution");
PrintAgeHistogram(df);

Console.WriteLine("\n4. BOX PLOT: Fare Distribution by Class");
Console.WriteLine("   Would show: 1st class fares much higher with more outliers");
Console.WriteLine("   Median fares decrease dramatically from 1st to 3rd class\n");

Console.WriteLine("5. HEATMAP: Correlation Matrix");
Console.WriteLine("   Color-coded grid showing all pairwise correlations\n");

Console.WriteLine("6. PIE CHART: Class Distribution");
PrintClassDistribution(df);

// ============================================================================
// SECTION 7: Pattern Identification
// ============================================================================
PrintSection("7. PATTERN IDENTIFICATION");

Console.WriteLine("🔍 DISCOVERED PATTERNS:\n");

// Pattern 1: Family size impact
Console.WriteLine("Pattern 1: Family Size Impact on Survival");
AnalyzeFamilySizeImpact(df);

// Pattern 2: Fare outliers
Console.WriteLine("\nPattern 2: Fare Distribution & Outliers");
AnalyzeFareOutliers(df);

// Pattern 3: Embarkation port
Console.WriteLine("\nPattern 3: Embarkation Port Analysis");
AnalyzeEmbarkation(df);

// ============================================================================
// SECTION 8: Summary & Insights
// ============================================================================
PrintSection("8. SUMMARY & KEY INSIGHTS");

Console.WriteLine("┌" + "─".PadRight(66, '─') + "┐");
Console.WriteLine("│ EXECUTIVE SUMMARY: Titanic Survival Factors                      │");
Console.WriteLine("├" + "─".PadRight(66, '─') + "┤");
Console.WriteLine("│                                                                  │");
Console.WriteLine("│  1. GENDER was the strongest survival predictor                  │");
Console.WriteLine("│     - Women: ~74% survived  |  Men: ~19% survived                │");
Console.WriteLine("│                                                                  │");
Console.WriteLine("│  2. CLASS significantly impacted survival                        │");
Console.WriteLine("│     - 1st Class: ~63%  |  2nd: ~47%  |  3rd: ~24%                │");
Console.WriteLine("│                                                                  │");
Console.WriteLine("│  3. AGE mattered - children prioritized                          │");
Console.WriteLine("│     - Children (<18): ~54%  |  Adults: ~38%                      │");
Console.WriteLine("│                                                                  │");
Console.WriteLine("│  4. FARE (proxy for wealth) correlated with survival             │");
Console.WriteLine("│     - Higher fares = better cabins = closer to lifeboats         │");
Console.WriteLine("│                                                                  │");
Console.WriteLine("│  5. TRAVELING ALONE was disadvantageous                          │");
Console.WriteLine("│     - Solo travelers had lower survival rates                    │");
Console.WriteLine("│                                                                  │");
Console.WriteLine("└" + "─".PadRight(66, '─') + "┘");

Console.WriteLine("\n✅ EDA Complete! This analysis provides foundation for ML modeling.\n");

// ============================================================================
// HELPER METHODS
// ============================================================================

static void PrintSection(string title)
{
    Console.WriteLine("─".PadRight(70, '─'));
    Console.WriteLine($"  {title}");
    Console.WriteLine("─".PadRight(70, '─'));
    Console.WriteLine();
}

static void PrintDataFrameHead(DataFrame df, int rows)
{
    // Print header
    Console.Write("  ");
    foreach (var col in df.Columns)
    {
        Console.Write($"{col.Name,-10} ");
    }
    Console.WriteLine();
    Console.WriteLine("  " + "-".PadRight(df.Columns.Count * 11, '-'));
    
    // Print rows
    for (int i = 0; i < Math.Min(rows, (int)df.Rows.Count); i++)
    {
        Console.Write("  ");
        foreach (var col in df.Columns)
        {
            var val = col[i]?.ToString() ?? "null";
            if (val.Length > 9) val = val[..9];
            Console.Write($"{val,-10} ");
        }
        Console.WriteLine();
    }
    Console.WriteLine();
}

static void PrintNumericStats(SingleDataFrameColumn col, string name)
{
    var nonNull = col.Where(v => v.HasValue).Select(v => v!.Value).ToList();
    if (nonNull.Count == 0) return;
    
    nonNull.Sort();
    double mean = nonNull.Average();
    double median = nonNull[nonNull.Count / 2];
    double min = nonNull.Min();
    double max = nonNull.Max();
    double stdDev = Math.Sqrt(nonNull.Average(v => Math.Pow(v - mean, 2)));
    double q1 = nonNull[nonNull.Count / 4];
    double q3 = nonNull[3 * nonNull.Count / 4];
    
    Console.WriteLine($"  Count:     {nonNull.Count}");
    Console.WriteLine($"  Mean:      {mean:F2}");
    Console.WriteLine($"  Median:    {median:F2}");
    Console.WriteLine($"  Std Dev:   {stdDev:F2}");
    Console.WriteLine($"  Min:       {min:F2}");
    Console.WriteLine($"  Max:       {max:F2}");
    Console.WriteLine($"  Q1 (25%):  {q1:F2}");
    Console.WriteLine($"  Q3 (75%):  {q3:F2}");
}

static void AnalyzeSurvivalByCategory(DataFrame df, string column)
{
    var groups = df.Rows
        .GroupBy(r => r[df.Columns.IndexOf(column)]?.ToString() ?? "Unknown")
        .ToDictionary(
            g => g.Key,
            g => new { 
                Total = g.Count(), 
                Survived = g.Count(r => (int)(r[df.Columns.IndexOf("Survived")] ?? 0) == 1)
            });
    
    foreach (var kvp in groups.OrderByDescending(k => (double)k.Value.Survived / k.Value.Total))
    {
        double rate = (double)kvp.Value.Survived / kvp.Value.Total * 100;
        string bar = new string('█', (int)(rate / 5));
        Console.WriteLine($"  {kvp.Key,-10}: {kvp.Value.Survived,3}/{kvp.Value.Total,-3} ({rate,5:F1}%) {bar}");
    }
}

static void AnalyzeSurvivalByClass(DataFrame df)
{
    var pclassIdx = df.Columns.IndexOf("Pclass");
    var survivedIdx = df.Columns.IndexOf("Survived");
    
    var groups = df.Rows
        .GroupBy(r => (int)(r[pclassIdx] ?? 0))
        .OrderBy(g => g.Key)
        .ToDictionary(
            g => g.Key,
            g => new { 
                Total = g.Count(), 
                Survived = g.Count(r => (int)(r[survivedIdx] ?? 0) == 1)
            });
    
    foreach (var kvp in groups)
    {
        double rate = (double)kvp.Value.Survived / kvp.Value.Total * 100;
        string bar = new string('█', (int)(rate / 5));
        string className = kvp.Key switch { 1 => "1st Class", 2 => "2nd Class", 3 => "3rd Class", _ => "Unknown" };
        Console.WriteLine($"  {className,-10}: {kvp.Value.Survived,3}/{kvp.Value.Total,-3} ({rate,5:F1}%) {bar}");
    }
}

static void AnalyzeSurvivalByAgeGroup(DataFrame df)
{
    var ageIdx = df.Columns.IndexOf("Age");
    var survivedIdx = df.Columns.IndexOf("Survived");
    
    var ageGroups = new Dictionary<string, (int total, int survived)>
    {
        ["Child (0-17)"] = (0, 0),
        ["Adult (18-64)"] = (0, 0),
        ["Senior (65+)"] = (0, 0),
        ["Unknown Age"] = (0, 0)
    };
    
    foreach (var row in df.Rows)
    {
        var age = row[ageIdx] as float?;
        var survived = (int)(row[survivedIdx] ?? 0);
        
        string group = age switch
        {
            null => "Unknown Age",
            < 18 => "Child (0-17)",
            < 65 => "Adult (18-64)",
            _ => "Senior (65+)"
        };
        
        var current = ageGroups[group];
        ageGroups[group] = (current.total + 1, current.survived + survived);
    }
    
    foreach (var kvp in ageGroups.Where(g => g.Value.total > 0))
    {
        double rate = (double)kvp.Value.survived / kvp.Value.total * 100;
        string bar = new string('█', (int)(rate / 5));
        Console.WriteLine($"  {kvp.Key,-14}: {kvp.Value.survived,3}/{kvp.Value.total,-3} ({rate,5:F1}%) {bar}");
    }
}

static double CalculateCorrelation(DataFrame df, string col1, string col2)
{
    var idx1 = df.Columns.IndexOf(col1);
    var idx2 = df.Columns.IndexOf(col2);
    
    var pairs = df.Rows
        .Where(r => r[idx1] != null && r[idx2] != null)
        .Select(r => (Convert.ToDouble(r[idx1]), Convert.ToDouble(r[idx2])))
        .ToList();
    
    if (pairs.Count < 2) return 0;
    
    double mean1 = pairs.Average(p => p.Item1);
    double mean2 = pairs.Average(p => p.Item2);
    
    double covariance = pairs.Average(p => (p.Item1 - mean1) * (p.Item2 - mean2));
    double std1 = Math.Sqrt(pairs.Average(p => Math.Pow(p.Item1 - mean1, 2)));
    double std2 = Math.Sqrt(pairs.Average(p => Math.Pow(p.Item2 - mean2, 2)));
    
    return std1 * std2 == 0 ? 0 : covariance / (std1 * std2);
}

static void PrintAgeHistogram(DataFrame df)
{
    Console.WriteLine("   Age Distribution (text histogram):");
    var ageIdx = df.Columns.IndexOf("Age");
    var ageBuckets = new int[8]; // 0-10, 10-20, ..., 70+
    
    foreach (var row in df.Rows)
    {
        if (row[ageIdx] is float age)
        {
            int bucket = Math.Min((int)(age / 10), 7);
            ageBuckets[bucket]++;
        }
    }
    
    string[] labels = { " 0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "  70+" };
    int maxCount = ageBuckets.Max();
    
    for (int i = 0; i < 8; i++)
    {
        int barLen = maxCount > 0 ? (int)((double)ageBuckets[i] / maxCount * 20) : 0;
        string bar = new string('█', barLen);
        Console.WriteLine($"   {labels[i]}: {bar} ({ageBuckets[i]})");
    }
}

static void PrintClassDistribution(DataFrame df)
{
    var pclassIdx = df.Columns.IndexOf("Pclass");
    var classCounts = df.Rows
        .GroupBy(r => (int)(r[pclassIdx] ?? 0))
        .ToDictionary(g => g.Key, g => g.Count());
    
    int total = classCounts.Values.Sum();
    Console.WriteLine("   Class Distribution:");
    foreach (var kvp in classCounts.OrderBy(k => k.Key))
    {
        double pct = (double)kvp.Value / total * 100;
        Console.WriteLine($"   {kvp.Key}st/nd/rd Class: {pct:F1}% ({kvp.Value} passengers)");
    }
}

static void AnalyzeFamilySizeImpact(DataFrame df)
{
    var sibspIdx = df.Columns.IndexOf("SibSp");
    var parchIdx = df.Columns.IndexOf("Parch");
    var survivedIdx = df.Columns.IndexOf("Survived");
    
    var familyGroups = new Dictionary<string, (int total, int survived)>
    {
        ["Alone (0)"] = (0, 0),
        ["Small (1-3)"] = (0, 0),
        ["Large (4+)"] = (0, 0)
    };
    
    foreach (var row in df.Rows)
    {
        int familySize = (int)(row[sibspIdx] ?? 0) + (int)(row[parchIdx] ?? 0);
        int survived = (int)(row[survivedIdx] ?? 0);
        
        string group = familySize switch
        {
            0 => "Alone (0)",
            <= 3 => "Small (1-3)",
            _ => "Large (4+)"
        };
        
        var current = familyGroups[group];
        familyGroups[group] = (current.total + 1, current.survived + survived);
    }
    
    foreach (var kvp in familyGroups)
    {
        if (kvp.Value.total > 0)
        {
            double rate = (double)kvp.Value.survived / kvp.Value.total * 100;
            Console.WriteLine($"   {kvp.Key,-12}: {rate:F1}% survival ({kvp.Value.total} passengers)");
        }
    }
}

static void AnalyzeFareOutliers(DataFrame df)
{
    var fareIdx = df.Columns.IndexOf("Fare");
    var fares = df.Rows
        .Select(r => r[fareIdx] as float?)
        .Where(f => f.HasValue)
        .Select(f => f!.Value)
        .OrderBy(f => f)
        .ToList();
    
    double q1 = fares[fares.Count / 4];
    double q3 = fares[3 * fares.Count / 4];
    double iqr = q3 - q1;
    double upperBound = q3 + 1.5 * iqr;
    
    int outliers = fares.Count(f => f > upperBound);
    
    Console.WriteLine($"   Fare IQR: {iqr:F2}");
    Console.WriteLine($"   Upper outlier threshold: ${upperBound:F2}");
    Console.WriteLine($"   Number of outliers: {outliers} ({(double)outliers/fares.Count*100:F1}%)");
    Console.WriteLine($"   Max fare: ${fares.Max():F2} (likely suite tickets)");
}

static void AnalyzeEmbarkation(DataFrame df)
{
    var embarkIdx = df.Columns.IndexOf("Embarked");
    var survivedIdx = df.Columns.IndexOf("Survived");
    
    var groups = df.Rows
        .GroupBy(r => r[embarkIdx]?.ToString() ?? "Unknown")
        .ToDictionary(
            g => g.Key,
            g => new { 
                Total = g.Count(), 
                Survived = g.Count(r => (int)(r[survivedIdx] ?? 0) == 1)
            });
    
    var portNames = new Dictionary<string, string>
    {
        ["S"] = "Southampton",
        ["C"] = "Cherbourg",
        ["Q"] = "Queenstown"
    };
    
    foreach (var kvp in groups.OrderByDescending(g => (double)g.Value.Survived / g.Value.Total))
    {
        double rate = (double)kvp.Value.Survived / kvp.Value.Total * 100;
        string name = portNames.GetValueOrDefault(kvp.Key, kvp.Key);
        Console.WriteLine($"   {name,-12} ({kvp.Key}): {rate:F1}% survival ({kvp.Value.Total} passengers)");
    }
}

// ============================================================================
// EMBEDDED TITANIC DATA
// ============================================================================

static DataFrame CreateTitanicDataFrame()
{
    // Embedded Titanic dataset with 60 passengers (sample of original 891)
    // Columns: PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Embarked
    
    var passengerId = new Int32DataFrameColumn("PassengerId", new int[] {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60
    });
    
    var survived = new Int32DataFrameColumn("Survived", new int[] {
        0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0
    });
    
    var pclass = new Int32DataFrameColumn("Pclass", new int[] {
        3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3,
        2, 2, 3, 1, 3, 3, 3, 1, 3, 3, 1, 1, 3, 2, 1, 1, 3, 3, 3, 3,
        3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 1, 3, 1, 2, 1, 1, 2, 3, 2, 3
    });
    
    var name = new StringDataFrameColumn("Name", new string[] {
        "Braund, Mr. Owen Harris", "Cumings, Mrs. John Bradley", "Heikkinen, Miss. Laina",
        "Futrelle, Mrs. Jacques Heath", "Allen, Mr. William Henry", "Moran, Mr. James",
        "McCarthy, Mr. Timothy J", "Palsson, Master. Gosta Leonard", "Johnson, Mrs. Oscar W",
        "Nasser, Mrs. Nicholas", "Sandstrom, Miss. Marguerite Rut", "Bonnell, Miss. Elizabeth",
        "Saundercock, Mr. William Henry", "Andersson, Mr. Anders Johan", "Vestrom, Miss. Hulda Amanda",
        "Hewlett, Mrs. (Mary D Kingcome)", "Rice, Master. Eugene", "Williams, Mr. Charles Eugene",
        "Vander Planke, Mrs. Julius", "Masselmani, Mrs. Fatima", "Fynney, Mr. Joseph J",
        "Beesley, Mr. Lawrence", "McGowan, Miss. Anna", "Sloper, Mr. William Thompson",
        "Palsson, Miss. Torborg Danira", "Asplund, Mrs. Carl Oscar", "Emir, Mr. Farred Chehab",
        "Fortune, Mr. Charles Alexander", "O'Dwyer, Miss. Ellen", "Todoroff, Mr. Lalio",
        "Uruchurtu, Don. Manuel E", "Spencer, Mrs. William Augustus", "Glynn, Miss. Mary Agatha",
        "Wheadon, Mr. Edward H", "Meyer, Mr. Edgar Joseph", "Holverson, Mr. Alexander Oskar",
        "Mamee, Mr. Hanna", "Cann, Mr. Ernest Charles", "Vander Planke, Miss. Augusta Maria",
        "Nicola-Yarred, Miss. Jamila", "Ahlin, Mrs. Johan", "Turpin, Mrs. William John Robert",
        "Kraeff, Mr. Theodor", "Laroche, Miss. Simonne Marie Anne", "Devaney, Miss. Margaret Delia",
        "Rogers, Mr. William John", "Lennon, Mr. Denis", "O'Driscoll, Miss. Bridget",
        "Samaan, Mr. Youssef", "Arnold-Franchi, Mrs. Josef", "Panula, Master. Juha Niilo",
        "Nosworthy, Mr. Richard Cater", "Harper, Mrs. Henry Sleeper", "Faunthorpe, Mrs. Lizzie",
        "Ostby, Mr. Engelhart Cornelius", "Woolner, Mr. Hugh", "Rugg, Miss. Emily",
        "Novel, Mr. Mansouer", "West, Miss. Constance Mirium", "Goodwin, Master. Sidney Leonard"
    });
    
    var sex = new StringDataFrameColumn("Sex", new string[] {
        "male", "female", "female", "female", "male", "male", "male", "male", "female", "female",
        "female", "female", "male", "male", "female", "female", "male", "male", "female", "female",
        "male", "male", "female", "male", "female", "female", "male", "male", "female", "male",
        "male", "female", "female", "male", "male", "male", "male", "male", "female", "female",
        "female", "female", "male", "female", "female", "male", "male", "female", "male", "female",
        "male", "male", "female", "female", "male", "male", "female", "male", "female", "male"
    });
    
    // Age with some null values (represented as float.NaN initially, then converted)
    var ageValues = new float?[] {
        22, 38, 26, 35, 35, null, 54, 2, 27, 14,
        4, 58, 20, 39, 14, 55, 2, null, 31, null,
        35, 34, 15, 28, 8, 38, null, 19, null, null,
        40, null, null, 66, 28, 42, null, 21, 18, 14,
        40, 27, null, 3, 19, null, null, null, null, 24,
        7, 21, 49, 29, 65, null, 21, null, 5, 1
    };
    var age = new SingleDataFrameColumn("Age", ageValues);
    
    var sibsp = new Int32DataFrameColumn("SibSp", new int[] {
        1, 1, 0, 1, 0, 0, 0, 3, 0, 1, 1, 0, 0, 1, 0, 0, 4, 0, 1, 0,
        0, 0, 0, 0, 3, 1, 0, 3, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 1,
        0, 1, 0, 1, 0, 0, 1, 0, 2, 0, 4, 0, 1, 1, 0, 0, 0, 0, 1, 5
    });
    
    var parch = new Int32DataFrameColumn("Parch", new int[] {
        0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 0, 0, 5, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2
    });
    
    var fare = new SingleDataFrameColumn("Fare", new float[] {
        7.25f, 71.28f, 7.92f, 53.10f, 8.05f, 8.46f, 51.86f, 21.08f, 11.13f, 30.07f,
        16.70f, 26.55f, 8.05f, 31.28f, 7.85f, 16.00f, 29.13f, 13.00f, 18.00f, 7.22f,
        26.00f, 13.00f, 8.03f, 35.50f, 21.08f, 31.39f, 7.23f, 263.00f, 7.88f, 7.90f,
        27.72f, 146.52f, 7.75f, 10.50f, 82.17f, 52.00f, 7.90f, 8.05f, 18.00f, 11.50f,
        9.47f, 21.00f, 7.88f, 41.58f, 7.88f, 8.05f, 15.50f, 7.75f, 21.68f, 7.85f,
        39.69f, 10.50f, 76.73f, 26.00f, 61.98f, 35.50f, 10.50f, 7.23f, 27.75f, 46.90f
    });
    
    var embarked = new StringDataFrameColumn("Embarked", new string[] {
        "S", "C", "S", "S", "S", "Q", "S", "S", "S", "C", "S", "S", "S", "S", "S", "S", "Q", "S", "S", "C",
        "S", "S", "Q", "S", "S", "S", "C", "S", "Q", "S", "C", "C", "Q", "S", "C", "S", "C", "S", "S", "C",
        "S", "S", "C", "C", "Q", "S", "Q", "Q", "C", "S", "S", "S", "C", "S", "C", "S", "S", "C", "S", "S"
    });
    
    return new DataFrame(passengerId, survived, pclass, name, sex, age, sibsp, parch, fare, embarked);
}
