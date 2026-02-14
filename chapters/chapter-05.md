# Chapter 5: Exploratory Data Analysis (EDA)

You've loaded the data. The DataFrame is sitting in memory, columns aligned, types inferred. Now comes the question every data scientist faces: *What do I actually have here?*

Exploratory Data Analysis—EDA—is the detective work of data science. Before you build models, before you make predictions, before you deploy anything to production, you need to understand your data. What patterns exist? Where are the problems? Which features matter? What stories hide in the numbers?

For developers, EDA might feel uncomfortably unstructured. There's no compiler error telling you when you've explored "enough." No unit test to confirm your analysis is correct. It's more art than engineering, which makes many of us uncomfortable.

But here's the reframe: EDA is debugging for data. You're stepping through the dataset, inspecting variables, looking for unexpected behavior. You've done this a thousand times with code—now you're doing it with information.

Let's build your statistical intuition and your C# toolkit for exploration.

## Statistical Fundamentals (No PhD Required)

Before we touch code, let's establish the vocabulary. Statistics has a reputation for being intimidating, but most of what you need for practical data science comes down to a few core concepts.

### Measures of Central Tendency: Where's the Middle?

When someone asks "what's typical in this data?", they're asking about central tendency. There are three common answers:

**Mean** (average): Add everything up, divide by count. You know this one. In code: `values.Average()`. The mean is sensitive to outliers—one billionaire walks into a bar and suddenly the "average" wealth in the room is $50 million.

**Median**: Sort the values, pick the middle one. Half the data is above, half below. The median is robust to outliers—that billionaire doesn't affect it. When someone says "median home price," they're trying to give you a sense of what a typical home actually costs, not let Manhattan penthouses skew the picture.

**Mode**: The most frequently occurring value. Useful for categorical data (what's the most common product category?) or discrete values (what's the most common household size?).

```csharp
// Mean is built into LINQ
double mean = values.Average();

// Median requires a bit more work
double Median(IEnumerable<double> source)
{
    var sorted = source.OrderBy(x => x).ToArray();
    int mid = sorted.Length / 2;
    return sorted.Length % 2 == 0 
        ? (sorted[mid - 1] + sorted[mid]) / 2.0 
        : sorted[mid];
}

// Mode: most frequent value
T Mode<T>(IEnumerable<T> source) where T : notnull
{
    return source
        .GroupBy(x => x)
        .OrderByDescending(g => g.Count())
        .First()
        .Key;
}
```

**When to use which?** Use mean when your data is roughly symmetric and outliers aren't a concern. Use median when you have skewed data or outliers. Use mode for categorical data or when you need the most common value.

**Real-world example**: You're analyzing customer order values. The mean is $150, but the median is $45. What's happening? A small number of enterprise customers placing $10,000+ orders are pulling up the mean. For most purposes—marketing, pricing decisions, customer segmentation—the median tells you what a "typical" customer spends. The mean tells you something different: the average revenue per order, which matters for financial projections.

### Measures of Spread: How Scattered Is the Data?

Knowing the center isn't enough. A dataset where everyone earns $50,000 looks very different from one where half earn $20,000 and half earn $80,000—even though both have the same mean.

**Range**: Max minus min. Simple but crude—one outlier and it's useless.

**Variance**: The average of squared distances from the mean. If you're thinking "why squared?", it's because we want distances in both directions to count as spread, and squaring makes negatives positive while also penalizing larger deviations more heavily.

**Standard Deviation**: The square root of variance. It's in the same units as your original data, making it interpretable. If your data is in dollars, standard deviation is also in dollars.

```csharp
public static double StandardDeviation(IEnumerable<double> values)
{
    var list = values.ToArray();
    if (list.Length <= 1) return 0;
    
    double mean = list.Average();
    double sumSquaredDiffs = list.Sum(v => Math.Pow(v - mean, 2));
    
    // Using N-1 for sample standard deviation (Bessel's correction)
    return Math.Sqrt(sumSquaredDiffs / (list.Length - 1));
}
```

**The 68-95-99.7 rule**: For normally distributed data, about 68% falls within one standard deviation of the mean, 95% within two, and 99.7% within three. This gives you a quick sense of what's "normal" and what's an outlier.

### Distributions: The Shape of Data

A distribution describes how values are spread across the possible range. Think of it as a histogram in your mind.

**Normal Distribution** (Gaussian): The classic bell curve. Heights, test scores, measurement errors—many natural phenomena follow this shape. It's symmetric around the mean.

**Skewed Distributions**: When the tail stretches more in one direction. Income is right-skewed (long tail of high earners). Most people earn moderate amounts, but a few earn astronomical sums.

**Bimodal/Multimodal**: Multiple peaks. This often indicates you're looking at a mixture of populations. If you see a bimodal distribution of response times, maybe you have two distinct user behaviors mixed together.

**Uniform Distribution**: Every value is equally likely. Think dice rolls or random number generators.

Understanding distribution shape helps you choose the right statistical tools and alerts you to data quality issues. If you expect normally distributed data and see something heavily skewed, that's information worth investigating.

**Developer intuition**: Think of distributions like frequency analysis in logging. If you're tracking response times and see a bimodal distribution, you probably have two different code paths—maybe cache hits versus cache misses. A right-skewed distribution with a long tail suggests occasional expensive operations (database scans, cold starts, network timeouts). The shape of your data tells a story before you calculate a single statistic.

### Percentiles and Quartiles

Percentiles tell you what percentage of data falls below a value. The median is the 50th percentile. The 90th percentile means 90% of values are below this point.

**Quartiles** divide data into four equal parts:
- Q1 (25th percentile): One quarter of data is below this
- Q2 (50th percentile): The median
- Q3 (75th percentile): Three quarters of data is below this

**Interquartile Range (IQR)**: Q3 minus Q1. This captures the middle 50% of your data and is useful for outlier detection. Values below Q1 - 1.5×IQR or above Q3 + 1.5×IQR are often considered outliers.

```csharp
public static (double Q1, double Median, double Q3, double IQR) Quartiles(double[] values)
{
    var sorted = values.OrderBy(x => x).ToArray();
    int n = sorted.Length;
    
    double Percentile(double p)
    {
        double index = (n - 1) * p;
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);
        if (lower == upper) return sorted[lower];
        return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
    }
    
    double q1 = Percentile(0.25);
    double median = Percentile(0.50);
    double q3 = Percentile(0.75);
    
    return (q1, median, q3, q3 - q1);
}
```

## Descriptive Statistics with Microsoft.Data.Analysis

Now let's put these concepts into practice with the DataFrame API. Microsoft.Data.Analysis provides a pandas-like experience for .NET developers, and it includes built-in methods for common statistical operations.

```csharp
using Microsoft.Data.Analysis;

// Load a CSV file
var df = DataFrame.LoadCsv("data/employees.csv");

// Basic descriptive statistics for numeric columns
DataFrame description = df.Description();
Console.WriteLine(description);
```

The `Description()` method returns a new DataFrame containing count, mean, std, min, max, and percentiles for each numeric column—similar to pandas' `describe()`.

For more targeted analysis:

```csharp
// Single column statistics
DoubleDataFrameColumn salary = (DoubleDataFrameColumn)df["Salary"];

Console.WriteLine($"Count: {salary.Length}");
Console.WriteLine($"Nulls: {salary.NullCount}");
Console.WriteLine($"Mean: {salary.Mean():C}");
Console.WriteLine($"Median: {salary.Median():C}");
Console.WriteLine($"Std Dev: {salary.StandardDeviation():C}");
Console.WriteLine($"Min: {salary.Min():C}");
Console.WriteLine($"Max: {salary.Max():C}");
```

### Group-Based Statistics

Real insight often comes from comparing groups:

```csharp
// Average salary by department
var byDepartment = df.GroupBy("Department")
    .Mean("Salary");

foreach (var row in byDepartment.Rows)
{
    Console.WriteLine($"{row["Department"]}: {row["Salary"]:C}");
}

// Multiple aggregations
var summary = df.GroupBy("Department")
    .Aggregate(
        ("Salary", AggregateFunction.Mean),
        ("Salary", AggregateFunction.Max),
        ("YearsExperience", AggregateFunction.Mean)
    );
```

### Value Counts and Categorical Analysis

For categorical columns, value counts reveal the distribution:

```csharp
// How many employees in each department?
var deptCounts = df["Department"].ValueCounts();
foreach (var pair in deptCounts)
{
    double percentage = 100.0 * pair.Value / df.Rows.Count;
    Console.WriteLine($"{pair.Key}: {pair.Value} ({percentage:F1}%)");
}

// Cross-tabulation: department by job level
var crosstab = df.GroupBy(new[] { "Department", "JobLevel" })
    .Count();
```

### Handling Missing Data

EDA must account for missing values—they're almost always present in real data:

```csharp
// Check for missing values across all columns
foreach (var col in df.Columns)
{
    long nullCount = col.NullCount;
    if (nullCount > 0)
    {
        double pct = 100.0 * nullCount / col.Length;
        Console.WriteLine($"{col.Name}: {nullCount} missing ({pct:F1}%)");
    }
}

// Filter to complete cases only
var completeCases = df.DropNulls();
Console.WriteLine($"Complete cases: {completeCases.Rows.Count} of {df.Rows.Count}");
```

## Data Visualization with ScottPlot and Plotly.NET

Numbers tell part of the story. Visualizations tell the rest. A well-crafted chart can reveal patterns that would take pages of statistics to describe—and some patterns you'd never find in the numbers at all.

### ScottPlot: Fast, Simple .NET Charts

ScottPlot is a plotting library designed for .NET developers who want quick visualizations without the complexity of full-featured charting frameworks.

```csharp
using ScottPlot;

// Create a histogram of salaries
var plt = new Plot(600, 400);
double[] salaries = df["Salary"].Cast<double>().ToArray();

plt.AddHistogram(salaries, binCount: 20);
plt.Title("Salary Distribution");
plt.XLabel("Salary ($)");
plt.YLabel("Frequency");
plt.SaveFig("salary_histogram.png");
```

[FIGURE: Histogram showing salary distribution with 20 bins. The distribution appears roughly normal with a slight right skew, most salaries clustering between $50,000 and $90,000]

**Interpreting the histogram**: The shape tells you immediately what no single number can. Is the distribution normal? Skewed? Bimodal? Are there outliers on the high end? A histogram is often your first visualization in any EDA.

**Box plots** visualize the five-number summary (min, Q1, median, Q3, max) and outliers:

```csharp
var plt = new Plot(600, 400);

// Box plot comparing salaries across departments
string[] departments = df["Department"].Cast<string>().Distinct().ToArray();
List<double[]> salaryGroups = new();

foreach (var dept in departments)
{
    var deptSalaries = df.Filter(df["Department"].ElementwiseEquals(dept))
        ["Salary"].Cast<double>().ToArray();
    salaryGroups.Add(deptSalaries);
}

plt.AddBoxPlots(salaryGroups.ToArray());
plt.XTicks(Enumerable.Range(0, departments.Length).Select(i => (double)i).ToArray(), 
           departments);
plt.Title("Salary Distribution by Department");
plt.YLabel("Salary ($)");
plt.SaveFig("salary_boxplot.png");
```

[FIGURE: Side-by-side box plots comparing salary distributions across Engineering, Sales, Marketing, and HR departments. Engineering shows the highest median and largest spread, while HR has the lowest median but few outliers]

**Interpreting box plots**: The box shows the middle 50% of data. The line inside is the median. Whiskers extend to show the rest of the distribution. Dots beyond the whiskers are outliers. Comparing boxes across categories immediately shows which groups have higher values, more variation, or more outliers.

**What to look for in box plots**:
- **Position**: Higher boxes mean higher values for that group
- **Box size**: Larger boxes indicate more variability within the group
- **Whisker length**: Long whiskers suggest extended tails (potential outliers)
- **Outlier dots**: Individual points beyond whiskers are statistical outliers
- **Median line position**: If it's not centered in the box, the distribution is skewed

### Plotly.NET: Interactive Visualizations

When static images aren't enough, Plotly.NET creates interactive HTML charts that users can zoom, pan, and hover for details:

```csharp
using Plotly.NET;
using Plotly.NET.LayoutObjects;

// Scatter plot: Experience vs Salary
var experience = df["YearsExperience"].Cast<double>();
var salaries = df["Salary"].Cast<double>();
var departments = df["Department"].Cast<string>();

var chart = Chart2D.Chart.Scatter<double, double, string>(
    x: experience,
    y: salaries,
    mode: StyleParam.Mode.Markers,
    MultiText: departments
)
.WithXAxisStyle(Title.init("Years of Experience"))
.WithYAxisStyle(Title.init("Salary ($)"))
.WithTitle("Experience vs Salary");

chart.SaveHtml("experience_salary_scatter.html");
```

[FIGURE: Interactive scatter plot with experience on x-axis and salary on y-axis. Points are colored by department. A clear positive correlation is visible—more experience generally means higher salary—but with significant variance. Hovering over any point shows the department and exact values]

**Interpreting scatter plots**: Scatter plots reveal relationships between two continuous variables. Look for:
- **Direction**: Positive (upward slope) or negative (downward) correlation?
- **Strength**: Tight cluster or diffuse cloud?
- **Linearity**: Does a straight line describe the relationship, or is it curved?
- **Outliers**: Points far from the main cluster
- **Clusters**: Distinct groups that might indicate subpopulations

**Bar charts** for categorical comparisons:

```csharp
var deptCounts = df["Department"].ValueCounts();

var chart = Chart.Bar<string, int, string>(
    keys: deptCounts.Keys,
    values: deptCounts.Values
)
.WithTitle("Employees by Department")
.WithXAxisStyle(Title.init("Department"))
.WithYAxisStyle(Title.init("Count"));

chart.SaveHtml("department_counts.html");
```

[FIGURE: Bar chart showing employee counts per department. Engineering has the most employees at around 150, followed by Sales at 95, Marketing at 60, and HR at 35]

## Correlation Analysis and Feature Relationships

Correlation quantifies the strength of the relationship between two variables. It's a number between -1 and +1:

- **+1**: Perfect positive correlation (as one increases, so does the other)
- **0**: No linear relationship
- **-1**: Perfect negative correlation (as one increases, the other decreases)

### Calculating Correlation in C#

```csharp
public static double PearsonCorrelation(double[] x, double[] y)
{
    if (x.Length != y.Length)
        throw new ArgumentException("Arrays must have the same length");
    
    int n = x.Length;
    double sumX = x.Sum();
    double sumY = y.Sum();
    double sumXY = x.Zip(y, (a, b) => a * b).Sum();
    double sumX2 = x.Sum(a => a * a);
    double sumY2 = y.Sum(b => b * b);
    
    double numerator = n * sumXY - sumX * sumY;
    double denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return denominator == 0 ? 0 : numerator / denominator;
}
```

### Building a Correlation Matrix

For datasets with multiple numeric features, a correlation matrix shows all pairwise correlations:

```csharp
public static double[,] CorrelationMatrix(DataFrame df, string[] columns)
{
    int n = columns.Length;
    var matrix = new double[n, n];
    var data = columns.Select(c => df[c].Cast<double>().ToArray()).ToArray();
    
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix[i, j] = PearsonCorrelation(data[i], data[j]);
        }
    }
    
    return matrix;
}

// Calculate and visualize
string[] numericCols = { "Age", "Salary", "YearsExperience", "PerformanceScore" };
var corrMatrix = CorrelationMatrix(df, numericCols);

// Print formatted matrix
Console.WriteLine("Correlation Matrix:");
Console.Write("".PadRight(20));
foreach (var col in numericCols)
    Console.Write(col.PadRight(20));
Console.WriteLine();

for (int i = 0; i < numericCols.Length; i++)
{
    Console.Write(numericCols[i].PadRight(20));
    for (int j = 0; j < numericCols.Length; j++)
    {
        Console.Write($"{corrMatrix[i, j]:F3}".PadRight(20));
    }
    Console.WriteLine();
}
```

### Visualizing Correlations with a Heatmap

```csharp
using Plotly.NET;

// Create correlation heatmap
var heatmap = Chart.Heatmap<double, string, string>(
    zData: corrMatrix,
    X: numericCols,
    Y: numericCols,
    Colorscale: StyleParam.Colorscale.RdBu,
    ShowScale: true
)
.WithTitle("Feature Correlation Matrix");

heatmap.SaveHtml("correlation_heatmap.html");
```

[FIGURE: Heatmap showing pairwise correlations between Age, Salary, YearsExperience, and PerformanceScore. Bright red indicates strong positive correlation (e.g., Age and YearsExperience at 0.85). Blue would indicate negative correlation. The diagonal is all 1.0 (each variable correlates perfectly with itself)]

**Interpreting correlations**:
- **0.7 to 1.0**: Strong correlation—these variables move together
- **0.4 to 0.7**: Moderate correlation—noticeable relationship
- **0.1 to 0.4**: Weak correlation—might be noise
- **0 to 0.1**: No meaningful correlation

**Caution**: Correlation does not imply causation. Ice cream sales and drowning deaths are correlated, but ice cream doesn't cause drowning—summer causes both.

## Identifying Patterns and Anomalies

EDA isn't just about describing what's normal—it's about finding what's unusual. Anomalies can indicate data quality issues, fraud, or genuinely interesting phenomena.

### Statistical Outlier Detection

The IQR method provides a simple rule:

```csharp
public static (double lower, double upper) OutlierBounds(double[] values)
{
    var (q1, _, q3, iqr) = Quartiles(values);
    return (q1 - 1.5 * iqr, q3 + 1.5 * iqr);
}

public static IEnumerable<(int index, double value)> FindOutliers(double[] values)
{
    var (lower, upper) = OutlierBounds(values);
    
    for (int i = 0; i < values.Length; i++)
    {
        if (values[i] < lower || values[i] > upper)
            yield return (i, values[i]);
    }
}

// Find salary outliers
var salaries = df["Salary"].Cast<double>().ToArray();
var outliers = FindOutliers(salaries).ToList();

Console.WriteLine($"Found {outliers.Count} outliers:");
foreach (var (idx, value) in outliers.Take(10))
{
    Console.WriteLine($"  Row {idx}: ${value:N0}");
}
```

### Z-Score Method

For normally distributed data, z-scores measure how many standard deviations a value is from the mean:

```csharp
public static double[] ZScores(double[] values)
{
    double mean = values.Average();
    double std = StandardDeviation(values);
    
    return values.Select(v => (v - mean) / std).ToArray();
}

// Values with |z| > 3 are often considered outliers
var zScores = ZScores(salaries);
var extremeOutliers = salaries
    .Zip(zScores, (val, z) => (val, z))
    .Where(pair => Math.Abs(pair.z) > 3)
    .ToList();
```

### Pattern Detection Through Grouping

Sometimes patterns emerge only when you segment the data:

```csharp
// Do different tenure groups have different performance patterns?
var tenureGroups = df.Rows
    .Select(row => new 
    {
        TenureGroup = (int)row["YearsExperience"] switch
        {
            < 2 => "Junior",
            < 5 => "Mid",
            < 10 => "Senior",
            _ => "Expert"
        },
        Performance = (double)row["PerformanceScore"]
    })
    .GroupBy(x => x.TenureGroup)
    .Select(g => new
    {
        Group = g.Key,
        AvgPerformance = g.Average(x => x.Performance),
        StdPerformance = StandardDeviation(g.Select(x => x.Performance))
    });

foreach (var group in tenureGroups)
{
    Console.WriteLine($"{group.Group}: {group.AvgPerformance:F2} ± {group.StdPerformance:F2}");
}
```

### Temporal Patterns

If your data has a time dimension, look for trends and seasonality:

```csharp
// Monthly hiring trends
var hiresByMonth = df.Rows
    .Select(row => DateTime.Parse(row["HireDate"].ToString()))
    .GroupBy(d => new { d.Year, d.Month })
    .Select(g => new { g.Key.Year, g.Key.Month, Count = g.Count() })
    .OrderBy(x => x.Year).ThenBy(x => x.Month);

var dates = hiresByMonth.Select(x => new DateTime(x.Year, x.Month, 1)).ToArray();
var counts = hiresByMonth.Select(x => (double)x.Count).ToArray();

var plt = new Plot(800, 400);
plt.AddScatter(dates.Select(d => d.ToOADate()).ToArray(), counts);
plt.XAxis.DateTimeFormat(true);
plt.Title("Hiring Trends Over Time");
plt.SaveFig("hiring_trends.png");
```

[FIGURE: Line chart showing monthly hiring counts from 2020 to 2025. Clear seasonality visible with peaks in January and September, troughs in December. Overall upward trend suggests company growth]

## Project: Complete EDA on the Titanic Dataset

Let's put everything together with a real dataset. The Titanic dataset is a classic—it contains information about passengers aboard the ill-fated ship, and the analysis question is sobering: who survived?

### Loading and Initial Inspection

```csharp
using Microsoft.Data.Analysis;
using ScottPlot;
using Plotly.NET;

// Load the dataset
var df = DataFrame.LoadCsv("titanic.csv");

// Basic info
Console.WriteLine($"Dataset Shape: {df.Rows.Count} rows × {df.Columns.Count} columns");
Console.WriteLine("\nColumns:");
foreach (var col in df.Columns)
{
    Console.WriteLine($"  {col.Name,-15} {col.DataType.Name,-10} " +
                      $"(nulls: {col.NullCount})");
}

// First few rows
Console.WriteLine("\nFirst 5 rows:");
Console.WriteLine(df.Head(5));
```

Output:
```
Dataset Shape: 891 rows × 12 columns

Columns:
  PassengerId     Int32      (nulls: 0)
  Survived        Int32      (nulls: 0)
  Pclass          Int32      (nulls: 0)
  Name            String     (nulls: 0)
  Sex             String     (nulls: 0)
  Age             Single     (nulls: 177)
  SibSp           Int32      (nulls: 0)
  Parch           Int32      (nulls: 0)
  Ticket          String     (nulls: 0)
  Fare            Single     (nulls: 0)
  Cabin           String     (nulls: 687)
  Embarked        String     (nulls: 2)
```

Already we see important information: Age has 177 missing values (about 20%), and Cabin is mostly missing. These will affect our analysis.

### Survival Overview

```csharp
// Overall survival rate
var survivalCounts = df["Survived"].ValueCounts();
int survived = (int)survivalCounts[1];
int died = (int)survivalCounts[0];
double survivalRate = 100.0 * survived / df.Rows.Count;

Console.WriteLine($"\nSurvival Statistics:");
Console.WriteLine($"  Survived: {survived} ({survivalRate:F1}%)");
Console.WriteLine($"  Died: {died} ({100 - survivalRate:F1}%)");

// Visualization
var chart = Chart.Pie<int, string>(
    values: new[] { survived, died },
    labels: new[] { "Survived", "Did Not Survive" },
    Marker: Marker.init(
        Colors: new[] { Color.fromHex("#2ecc71"), Color.fromHex("#e74c3c") }
    )
)
.WithTitle("Titanic Survival Rate");

chart.SaveHtml("survival_pie.html");
```

Output:
```
Survival Statistics:
  Survived: 342 (38.4%)
  Died: 549 (61.6%)
```

[FIGURE: Pie chart showing Titanic survival distribution. 38.4% survived (green), 61.6% did not survive (red). The visualization immediately conveys the tragedy—less than 4 in 10 passengers survived]

### Survival by Passenger Class

The Titanic famously had different classes of accommodation. Did class affect survival?

```csharp
// Survival rate by class
var byClass = df.Rows
    .GroupBy(row => (int)row["Pclass"])
    .Select(g => new
    {
        Class = g.Key,
        Total = g.Count(),
        Survived = g.Count(r => (int)r["Survived"] == 1),
        Rate = 100.0 * g.Count(r => (int)r["Survived"] == 1) / g.Count()
    })
    .OrderBy(x => x.Class);

Console.WriteLine("\nSurvival by Passenger Class:");
foreach (var c in byClass)
{
    Console.WriteLine($"  Class {c.Class}: {c.Survived}/{c.Total} ({c.Rate:F1}%)");
}

// Stacked bar chart
var classes = new[] { "First", "Second", "Third" };
var survivedByClass = byClass.Select(c => c.Survived).ToArray();
var diedByClass = byClass.Select(c => c.Total - c.Survived).ToArray();

var chart = Chart.Combine(new[]
{
    Chart.StackedBar<int, string, string>(
        values: survivedByClass,
        keys: classes,
        Name: "Survived"
    ),
    Chart.StackedBar<int, string, string>(
        values: diedByClass,
        keys: classes,
        Name: "Did Not Survive"
    )
})
.WithTitle("Survival by Passenger Class")
.WithYAxisStyle(Title.init("Number of Passengers"));

chart.SaveHtml("survival_by_class.html");
```

Output:
```
Survival by Passenger Class:
  Class 1: 136/216 (63.0%)
  Class 2: 87/184 (47.3%)
  Class 3: 119/491 (24.2%)
```

[FIGURE: Stacked bar chart with three bars (First, Second, Third class). Each bar shows survived (green) and did not survive (red) portions. First class has a high survival portion, third class is predominantly red. The wealth disparity in survival is immediately visible]

**Interpretation**: The class divide is stark. First-class passengers had a 63% survival rate—nearly three times better than third class (24%). "Women and children first" didn't apply equally to all classes.

### Survival by Gender

```csharp
// Survival rate by sex
var bySex = df.Rows
    .GroupBy(row => (string)row["Sex"])
    .Select(g => new
    {
        Sex = g.Key,
        Total = g.Count(),
        Survived = g.Count(r => (int)r["Survived"] == 1),
        Rate = 100.0 * g.Count(r => (int)r["Survived"] == 1) / g.Count()
    });

Console.WriteLine("\nSurvival by Gender:");
foreach (var s in bySex)
{
    Console.WriteLine($"  {s.Sex}: {s.Survived}/{s.Total} ({s.Rate:F1}%)");
}

// Grouped bar chart
var genders = bySex.Select(s => s.Sex).ToArray();
var survivalRates = bySex.Select(s => s.Rate).ToArray();

var chart = Chart.Bar<double, string, string>(
    values: survivalRates,
    keys: genders
)
.WithTitle("Survival Rate by Gender")
.WithYAxisStyle(Title.init("Survival Rate (%)"));

chart.SaveHtml("survival_by_gender.html");
```

Output:
```
Survival by Gender:
  female: 233/314 (74.2%)
  male: 109/577 (18.9%)
```

[FIGURE: Bar chart comparing female (74.2%) and male (18.9%) survival rates. The female bar towers over the male bar, illustrating the dramatic gender disparity in survival]

**Interpretation**: The "women and children first" protocol was clearly enforced—women were nearly four times more likely to survive than men. This is one of the strongest predictors in the dataset.

### Age Distribution and Survival

```csharp
// Age statistics (excluding nulls)
var ages = df["Age"].DropNulls().Cast<float>().Select(a => (double)a).ToArray();

Console.WriteLine("\nAge Statistics:");
Console.WriteLine($"  Count (non-null): {ages.Length}");
Console.WriteLine($"  Mean: {ages.Average():F1} years");
Console.WriteLine($"  Median: {Median(ages):F1} years");
Console.WriteLine($"  Std Dev: {StandardDeviation(ages):F1} years");
Console.WriteLine($"  Min: {ages.Min():F1} years");
Console.WriteLine($"  Max: {ages.Max():F1} years");

// Age distribution histogram
var plt = new Plot(700, 400);
plt.AddHistogram(ages, binCount: 30);
plt.Title("Age Distribution of Passengers");
plt.XLabel("Age (years)");
plt.YLabel("Frequency");
plt.SaveFig("age_distribution.png");
```

Output:
```
Age Statistics:
  Count (non-null): 714
  Mean: 29.7 years
  Median: 28.0 years
  Std Dev: 14.5 years
  Min: 0.4 years
  Max: 80.0 years
```

[FIGURE: Histogram of passenger ages. The distribution is roughly normal, centered around 28-30 years, with a slight right tail. Notable smaller peaks around ages 0-5 (young children) and 15-25 (young adults). Few passengers over 60]

Now let's examine how age relates to survival:

```csharp
// Age distribution by survival status
var survivedAges = df.Filter(df["Survived"].ElementwiseEquals(1))
    ["Age"].DropNulls().Cast<float>().Select(a => (double)a).ToArray();
var diedAges = df.Filter(df["Survived"].ElementwiseEquals(0))
    ["Age"].DropNulls().Cast<float>().Select(a => (double)a).ToArray();

// Overlapping histograms
var chart = Chart.Combine(new[]
{
    Chart.Histogram<double, double>(
        X: survivedAges,
        Name: "Survived",
        Opacity: 0.6
    ),
    Chart.Histogram<double, double>(
        X: diedAges,
        Name: "Did Not Survive",
        Opacity: 0.6
    )
})
.WithTitle("Age Distribution by Survival Status")
.WithXAxisStyle(Title.init("Age"))
.WithYAxisStyle(Title.init("Count"));

chart.SaveHtml("age_by_survival.html");
```

[FIGURE: Overlapping histograms showing age distributions for survivors (blue) and non-survivors (red). Both distributions are similar in shape, but the survivor distribution shows relatively more children (under 10) and fewer middle-aged men (20-40)]

**Interpretation**: Young children had better survival rates—they were prioritized in evacuation. The working-age male bulge in the "did not survive" distribution reflects the "women and children first" protocol.

### Fare Analysis

```csharp
// Fare statistics by class
var fareByClass = df.Rows
    .GroupBy(row => (int)row["Pclass"])
    .Select(g => new
    {
        Class = g.Key,
        MedianFare = Median(g.Select(r => (double)(float)r["Fare"])),
        MaxFare = g.Max(r => (float)r["Fare"])
    })
    .OrderBy(x => x.Class);

Console.WriteLine("\nFare by Class:");
foreach (var c in fareByClass)
{
    Console.WriteLine($"  Class {c.Class}: Median £{c.MedianFare:F2}, Max £{c.MaxFare:F2}");
}

// Box plot of fares by class
var plt = new Plot(600, 400);
var fareGroups = Enumerable.Range(1, 3)
    .Select(pclass => df.Filter(df["Pclass"].ElementwiseEquals(pclass))
        ["Fare"].Cast<float>().Select(f => (double)f).ToArray())
    .ToArray();

plt.AddBoxPlots(fareGroups);
plt.XTicks(new[] { 0.0, 1.0, 2.0 }, new[] { "First", "Second", "Third" });
plt.Title("Fare Distribution by Class");
plt.YLabel("Fare (£)");
plt.SaveFig("fare_by_class.png");
```

[FIGURE: Box plots showing fare distributions across classes. First class shows high median and extreme outliers (up to £500+). Second class is moderate. Third class is tightly clustered near the bottom with few outliers. The economic stratification is dramatic]

### Correlation Analysis

```csharp
// Correlation between numeric features
var numericCols = new[] { "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare" };

// We need complete cases for correlation
var completeRows = df.Rows
    .Where(r => r["Age"] != null)
    .ToList();

var corrData = numericCols
    .Select(col => completeRows.Select(r => Convert.ToDouble(r[col])).ToArray())
    .ToArray();

// Build correlation matrix
var corrMatrix = new double[numericCols.Length, numericCols.Length];
for (int i = 0; i < numericCols.Length; i++)
{
    for (int j = 0; j < numericCols.Length; j++)
    {
        corrMatrix[i, j] = PearsonCorrelation(corrData[i], corrData[j]);
    }
}

// Print correlations with Survived
Console.WriteLine("\nCorrelations with Survival:");
for (int i = 1; i < numericCols.Length; i++)
{
    Console.WriteLine($"  {numericCols[i],-10}: {corrMatrix[0, i]:+0.000;-0.000}");
}

// Heatmap
var heatmap = Chart.Heatmap<double, string, string>(
    zData: corrMatrix.Cast<double>().ToArray().Chunk(numericCols.Length).ToArray(),
    X: numericCols,
    Y: numericCols,
    Colorscale: StyleParam.Colorscale.RdBu
)
.WithTitle("Feature Correlation Matrix");

heatmap.SaveHtml("titanic_correlation.html");
```

Output:
```
Correlations with Survival:
  Pclass    : -0.339
  Age       : -0.077
  SibSp     : -0.035
  Parch     : +0.082
  Fare      : +0.257
```

[FIGURE: Correlation heatmap for Titanic numeric features. Survival shows moderate negative correlation with Pclass (-0.34) and positive correlation with Fare (+0.26). Pclass and Fare have strong negative correlation (-0.55) as expected. Age shows minimal correlation with most features]

**Interpretation**: 
- **Pclass vs Survived** (-0.34): Lower class numbers (1=first) correlate with survival—first class passengers survived more
- **Fare vs Survived** (+0.26): Higher fares correlate with survival—aligns with class effect
- **Age vs Survived** (-0.08): Weak correlation; age alone isn't a strong predictor
- **Pclass vs Fare** (-0.55): Strong inverse correlation confirms that first class (Pclass=1) paid more

### Key Findings Summary

Let's consolidate what our EDA revealed:

```csharp
Console.WriteLine("\n" + new string('=', 60));
Console.WriteLine("TITANIC EDA: KEY FINDINGS");
Console.WriteLine(new string('=', 60));

Console.WriteLine(@"
1. OVERALL SURVIVAL: 38.4% of passengers survived

2. CLASS MATTERS: 
   - First class: 63.0% survival
   - Third class: 24.2% survival
   → First class passengers were 2.6x more likely to survive

3. GENDER MATTERS MOST:
   - Female: 74.2% survival  
   - Male: 18.9% survival
   → Women were 3.9x more likely to survive

4. AGE EFFECT:
   - Children (under 10) had higher survival rates
   - Middle-aged men had the lowest survival rates
   - Overall correlation with survival is weak (-0.08)

5. FARE AS PROXY FOR CLASS:
   - Strong correlation between fare and class (-0.55)
   - Higher fares correlate with survival (+0.26)

6. DATA QUALITY ISSUES:
   - Age: 19.9% missing values
   - Cabin: 77.1% missing values
   - Any model using these features needs imputation strategy

CONCLUSIONS:
- Gender is the strongest survival predictor
- Class/wealth significantly impacted survival chances
- ""Women and children first"" was enforced, but unevenly across classes
- Third class passengers faced a double disadvantage
");
```

### What This EDA Tells Us About Modeling

The exploration phase naturally leads to modeling hypotheses:

1. **Feature importance**: Gender and class should be the strongest predictors in any survival model

2. **Missing data strategy**: Age has 20% missing—too much to drop, needs imputation. Cabin is mostly missing—either drop or engineer a "had cabin" boolean

3. **Feature engineering opportunities**: 
   - Family size (SibSp + Parch) might matter
   - Titles extracted from names (Mr., Mrs., Master, Miss) encode both gender and social status
   - Deck from cabin letters might show survival patterns by ship location

4. **Class imbalance**: The 38/62 split isn't severe but should be considered in model evaluation

This is the value of EDA: before writing a single line of ML code, you understand your data's structure, limitations, and the story it tells. You won't be surprised when your model weights gender heavily—you already know it matters.

## EDA Best Practices for Production Code

Before we wrap up, let's discuss how EDA fits into a professional development workflow. The exploratory code we've written serves its purpose—understanding data—but transitioning to production requires additional considerations.

### Create Reusable Statistical Utilities

The helper methods we've built throughout this chapter belong in a dedicated library:

```csharp
namespace DataScience.Statistics
{
    public static class DescriptiveStats
    {
        public static double Median(this IEnumerable<double> source)
        {
            var sorted = source.OrderBy(x => x).ToArray();
            if (sorted.Length == 0) 
                throw new InvalidOperationException("Sequence contains no elements");
            
            int mid = sorted.Length / 2;
            return sorted.Length % 2 == 0 
                ? (sorted[mid - 1] + sorted[mid]) / 2.0 
                : sorted[mid];
        }
        
        public static double StandardDeviation(this IEnumerable<double> source, 
            bool population = false)
        {
            var list = source.ToArray();
            if (list.Length <= 1) return 0;
            
            double mean = list.Average();
            double sumSquaredDiffs = list.Sum(v => Math.Pow(v - mean, 2));
            int divisor = population ? list.Length : list.Length - 1;
            
            return Math.Sqrt(sumSquaredDiffs / divisor);
        }
        
        public static (double Q1, double Median, double Q3, double IQR) 
            Quartiles(this IEnumerable<double> source)
        {
            var sorted = source.OrderBy(x => x).ToArray();
            // ... implementation
        }
    }
}
```

Extension methods make statistics feel natural with LINQ:

```csharp
var salaries = employees.Select(e => e.Salary);
Console.WriteLine($"Median salary: {salaries.Median():C}");
Console.WriteLine($"Std Dev: {salaries.StandardDeviation():C}");
```

### Document Your Findings

EDA produces insights that need to survive beyond your terminal session. Consider creating a structured report:

```csharp
public class EdaReport
{
    public string DatasetName { get; set; }
    public DateTime AnalysisDate { get; set; }
    public int RowCount { get; set; }
    public int ColumnCount { get; set; }
    public List<ColumnProfile> Columns { get; set; } = new();
    public List<string> KeyFindings { get; set; } = new();
    public List<string> DataQualityIssues { get; set; } = new();
    public List<string> ModelingRecommendations { get; set; } = new();
    
    public void SaveAsMarkdown(string path)
    {
        var sb = new StringBuilder();
        sb.AppendLine($"# EDA Report: {DatasetName}");
        sb.AppendLine($"*Generated: {AnalysisDate:yyyy-MM-dd HH:mm}*\n");
        sb.AppendLine($"## Dataset Overview");
        sb.AppendLine($"- **Rows:** {RowCount:N0}");
        sb.AppendLine($"- **Columns:** {ColumnCount}");
        // ... continue building report
        File.WriteAllText(path, sb.ToString());
    }
}
```

### Choosing the Right Visualization

Different questions demand different charts. Here's a quick reference:

| Question | Chart Type | When to Use |
|----------|-----------|-------------|
| How is this variable distributed? | Histogram, KDE plot | Single continuous variable |
| How do categories compare? | Bar chart | Categorical data |
| How are two continuous variables related? | Scatter plot | Looking for correlation |
| How do distributions compare across groups? | Box plot, violin plot | Comparing numeric across categories |
| What are all pairwise correlations? | Heatmap | Multiple numeric features |
| How does something change over time? | Line chart | Time series data |
| What are the proportions? | Pie chart, stacked bar | Part-to-whole relationships |

The right visualization makes patterns obvious. The wrong one hides them. When in doubt, try multiple approaches—the data will tell you which works best.

### Automating EDA

For datasets you'll analyze repeatedly, consider building an automated EDA pipeline:

```csharp
public class AutomatedEda
{
    private readonly DataFrame _df;
    private readonly string _outputPath;
    
    public AutomatedEda(DataFrame df, string outputPath)
    {
        _df = df;
        _outputPath = outputPath;
        Directory.CreateDirectory(outputPath);
    }
    
    public void RunFullAnalysis()
    {
        GenerateDataProfile();
        AnalyzeNumericColumns();
        AnalyzeCategoricalColumns();
        GenerateCorrelationMatrix();
        DetectOutliers();
        GenerateReport();
    }
    
    private void AnalyzeNumericColumns()
    {
        foreach (var col in _df.Columns.Where(c => c.IsNumeric))
        {
            var values = col.Cast<double>().Where(v => !double.IsNaN(v)).ToArray();
            
            // Generate histogram
            var plt = new Plot(600, 400);
            plt.AddHistogram(values, binCount: 30);
            plt.Title($"Distribution: {col.Name}");
            plt.SaveFig(Path.Combine(_outputPath, $"hist_{col.Name}.png"));
            
            // Calculate statistics
            var stats = new
            {
                Mean = values.Average(),
                Median = values.Median(),
                StdDev = values.StandardDeviation(),
                Min = values.Min(),
                Max = values.Max(),
                NullCount = col.NullCount,
                NullPct = 100.0 * col.NullCount / col.Length
            };
            
            // Log findings
            Console.WriteLine($"{col.Name}: μ={stats.Mean:F2}, σ={stats.StdDev:F2}, " +
                            $"nulls={stats.NullPct:F1}%");
        }
    }
}
```

### Common EDA Pitfalls to Avoid

**Pitfall 1: Confirmation Bias**
It's tempting to look for patterns that confirm your hypotheses. Fight this by looking for evidence that contradicts your assumptions. If you think age predicts survival, also check cases where it doesn't.

**Pitfall 2: P-Hacking Through Visualization**
When you create enough charts, some will show patterns that are pure noise. Be skeptical of patterns found after extensive exploration. If a pattern is real, it should be consistent across subsets of your data.

**Pitfall 3: Ignoring Missing Data**
Missing data isn't just an inconvenience—it's information. *Why* is it missing? In the Titanic dataset, missing cabin numbers are mostly from third-class passengers. That's not random—it reflects how records were kept for different classes.

**Pitfall 4: Over-Interpreting Small Samples**
A 90% survival rate among left-handed passengers sounds significant until you realize there were only 10 of them. Always check sample sizes before drawing conclusions.

**Pitfall 5: Forgetting to Validate**
EDA on your training data might reveal patterns that don't hold in the real world. Save some data for validation, or at minimum, check if your findings hold when you subset the data.

## Chapter Summary

Exploratory Data Analysis is where data science begins. Before algorithms and predictions, there's understanding.

**Key takeaways**:

- **Statistical fundamentals** give you vocabulary: mean, median, standard deviation, correlation. These aren't abstract—they're measurements that reveal data characteristics.

- **Descriptive statistics** summarize what you have: central tendency, spread, distribution shape. Microsoft.Data.Analysis gives you pandas-like power in C#.

- **Visualization** reveals what numbers hide: distributions, relationships, outliers, patterns. ScottPlot for quick static charts, Plotly.NET for interactive exploration.

- **Correlation analysis** quantifies relationships between variables—but remember that correlation isn't causation.

- **Pattern and anomaly detection** finds both problems (data quality issues) and insights (interesting subgroups, unexpected relationships).

The Titanic project demonstrated a complete EDA workflow: load, inspect, describe, visualize, correlate, conclude. By the time you finish EDA, you should be able to tell the story of your data—and have clear hypotheses for what comes next.

Your data has a story to tell. EDA is how you learn to listen.

**For the next chapter**: Armed with your EDA insights, you're ready to prepare your data for modeling. Chapter 6 dives into data preprocessing—handling missing values, encoding categorical variables, scaling features, and engineering new ones. The patterns you've discovered here will guide those decisions. When you know that age has many missing values and that class strongly predicts survival, you'll make informed choices about imputation strategies and feature importance.

EDA isn't a one-time phase—it's a mindset. Throughout your data science projects, you'll return to exploration whenever results surprise you, new data arrives, or models behave unexpectedly. Keep these tools sharp. You'll use them constantly.
