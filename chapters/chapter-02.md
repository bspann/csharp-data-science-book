# Chapter 2: Setting Up Your Data Science Environment

As a C# developer, you already have powerful tools at your fingertips. Visual Studio has been the gold standard for .NET development for decades, and VS Code has emerged as a lightweight, extensible alternative. In this chapter, we'll transform these familiar environments into fully-equipped data science workstations, install the essential packages, and introduce you to Polyglot Notebooks—the .NET ecosystem's answer to Jupyter.

By the end of this chapter, you'll have a complete data science environment ready for exploration, experimentation, and building production-ready machine learning solutions.

## Choosing Your IDE: Visual Studio vs. VS Code

Both Visual Studio and VS Code are excellent choices for data science work in C#. Your decision comes down to workflow preferences and specific needs.

### Visual Studio 2022

Visual Studio remains the most feature-complete IDE for .NET development. For data science work, it offers:

- **IntelliSense on steroids**: Deep understanding of ML.NET types and methods
- **Integrated NuGet management**: Visual package browsing and installation
- **Debugging excellence**: Step through training pipelines and inspect data transformations
- **Memory profilers**: Essential when working with large datasets

**Recommended Setup for Data Science:**

1. Install Visual Studio 2022 (Community edition is free and sufficient)
2. During installation, select these workloads:
   - **.NET desktop development**
   - **ASP.NET and web development** (useful for deploying models)
   - **Data storage and processing**

[SCREENSHOT: Visual Studio Installer showing workload selection with .NET desktop development and Data storage and processing checked]

After installation, configure a few settings for optimal data science work:

```
Tools → Options → Text Editor → C# → Advanced
✓ Enable full solution analysis
✓ Show inline parameter name hints
```

### Visual Studio Code

VS Code shines when you want a lighter, more flexible environment—especially for notebook-style exploration. It's also cross-platform, making it ideal if you work across Windows, macOS, and Linux.

**Essential Extensions for C# Data Science:**

Install these extensions from the VS Code marketplace:

| Extension | Publisher | Purpose |
|-----------|-----------|---------|
| **C# Dev Kit** | Microsoft | Core C# language support |
| **Polyglot Notebooks** | Microsoft | .NET Interactive notebooks |
| **.NET Install Tool** | Microsoft | Manages .NET SDK versions |
| **Data Wrangler** | Microsoft | Visual data exploration |

To install via command line:

```bash
code --install-extension ms-dotnettools.csdevkit
code --install-extension ms-dotnettools.dotnet-interactive-vscode
code --install-extension ms-dotnettools.vscode-dotnet-runtime
code --install-extension ms-toolsai.datawrangler
```

[SCREENSHOT: VS Code Extensions panel showing Polyglot Notebooks and C# Dev Kit installed]

**Configure VS Code for Data Science:**

Create or update your `settings.json`:

```json
{
    "dotnet-interactive.minimumInteractiveToolVersion": "1.0.556801",
    "notebook.cellToolbarLocation": {
        "default": "right"
    },
    "notebook.output.textLineLimit": 100,
    "notebook.formatOnSave.enabled": true,
    "csharp.semanticHighlighting.enabled": true
}
```

### My Recommendation

For this book, I recommend using **VS Code with Polyglot Notebooks** for exploration and learning, then switching to **Visual Studio** when building production applications. This mirrors how Python data scientists use Jupyter for exploration and PyCharm or VS Code for production code.

## Installing .NET 8 SDK

Before we proceed, ensure you have the .NET 8 SDK installed. Open a terminal and check your version:

```bash
dotnet --version
```

If you don't have .NET 8 or later, download it from https://dot.net. After installation, verify:

```bash
dotnet --list-sdks
```

You should see output similar to:

```
8.0.401 [C:\Program Files\dotnet\sdk]
```

## Installing ML.NET and Essential Packages

ML.NET is Microsoft's open-source machine learning framework designed specifically for .NET developers. Unlike TensorFlow.NET or other ports, ML.NET was built from the ground up for the .NET ecosystem, offering native performance and seamless integration.

### Creating Your First Data Science Project

Let's create a project that we'll use throughout this chapter:

```bash
mkdir DataSciencePlayground
cd DataSciencePlayground
dotnet new console -n DataExploration
cd DataExploration
```

### Core NuGet Packages

Install the essential packages for data science work:

```bash
# ML.NET core framework
dotnet add package Microsoft.ML --version 5.0.0

# DataFrame support (like pandas for C#)
dotnet add package Microsoft.Data.Analysis --version 0.22.0

# Additional ML.NET components
dotnet add package Microsoft.ML.AutoML --version 0.22.0
dotnet add package Microsoft.ML.DataView --version 5.0.0
```

For Visual Studio users, you can also use the NuGet Package Manager:

[SCREENSHOT: Visual Studio NuGet Package Manager showing Microsoft.ML package with version 5.0.0 selected]

1. Right-click your project → **Manage NuGet Packages**
2. Search for "Microsoft.ML"
3. Select version 4.0.0 and click Install
4. Repeat for other packages

### Understanding the Package Ecosystem

Here's what each package provides:

| Package | Purpose | When to Use |
|---------|---------|-------------|
| `Microsoft.ML` | Core ML framework | Always—this is the foundation |
| `Microsoft.Data.Analysis` | DataFrame operations | Data loading, cleaning, exploration |
| `Microsoft.ML.AutoML` | Automated ML | Quick model selection and hyperparameter tuning |
| `Microsoft.ML.DataView` | Data abstraction layer | Working with IDataView directly |
| `Microsoft.ML.TensorFlow` | TensorFlow integration | Using pre-trained TensorFlow models |
| `Microsoft.ML.OnnxTransformer` | ONNX model support | Cross-framework model deployment |

### Your Project File

After installation, your `.csproj` should look like this:

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.Data.Analysis" Version="0.22.0" />
    <PackageReference Include="Microsoft.ML" Version="5.0.0" />
    <PackageReference Include="Microsoft.ML.AutoML" Version="0.22.0" />
    <PackageReference Include="Microsoft.ML.DataView" Version="5.0.0" />
  </ItemGroup>

</Project>
```

## Polyglot Notebooks: Interactive C# for Data Science

If you've ever envied Python developers and their Jupyter notebooks, those days are over. **Polyglot Notebooks** (formerly .NET Interactive) brings the same interactive, exploratory workflow to C#—and honestly, the experience is even better.

### What Are Polyglot Notebooks?

Polyglot Notebooks combine the rich typing and IntelliSense of C# with the iterative, cell-based execution model of Jupyter. You can:

- Write and execute C# code in cells
- See results immediately below each cell
- Mix C#, F#, SQL, PowerShell, and even Python in the same notebook
- Create rich visualizations
- Share reproducible analyses

### Installing Polyglot Notebooks

**For VS Code:**

The Polyglot Notebooks extension should already be installed if you followed the earlier steps. If not:

```bash
code --install-extension ms-dotnettools.dotnet-interactive-vscode
```

**For Visual Studio:**

Install the .NET Interactive Notebooks extension from the Visual Studio Marketplace.

[SCREENSHOT: VS Code with a new .dib file open, showing the kernel selector dropdown]

### Creating Your First Notebook

In VS Code:

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Type "Polyglot Notebook: Create new blank notebook"
3. Select `.dib` format (or `.ipynb` for Jupyter compatibility)
4. Choose **C#** as your default language

Save your notebook as `exploration.dib` in your project folder.

### Notebook Basics

Let's explore the notebook interface with some hands-on examples. Create cells with the following code:

**Cell 1: Add NuGet packages**

```csharp
#r "nuget: Microsoft.Data.Analysis, 0.22.0"
#r "nuget: Microsoft.ML, 5.0.0"
#r "nuget: XPlot.Plotly, 4.1.0"
```

When you run this cell (`Shift+Enter`), the packages are downloaded and referenced. You'll see output like:

```
Installing package Microsoft.Data.Analysis, version 0.22.0
Installed package Microsoft.Data.Analysis version 0.22.0
```

**Cell 2: Import namespaces**

```csharp
using Microsoft.Data.Analysis;
using Microsoft.ML;
using XPlot.Plotly;
```

**Cell 3: Test the setup**

```csharp
var mlContext = new MLContext(seed: 42);
Console.WriteLine($"ML.NET initialized with seed 42");
Console.WriteLine($"Running on .NET {Environment.Version}");
```

Output:
```
ML.NET initialized with seed 42
Running on .NET 8.0.8
```

### Rich Output and Formatting

Polyglot Notebooks automatically render objects with rich formatting. Try this:

```csharp
// Create a simple DataFrame
var dates = new DateTimeDataFrameColumn("Date", 
    Enumerable.Range(0, 5).Select(i => DateTime.Today.AddDays(i)));
var values = new DoubleDataFrameColumn("Value", 
    new double[] { 100, 105, 102, 110, 108 });

var df = new DataFrame(dates, values);
df
```

The DataFrame renders as a beautiful HTML table directly in the notebook:

[SCREENSHOT: Polyglot Notebook showing a DataFrame rendered as an interactive HTML table with Date and Value columns]

### Cell Magic Commands

Polyglot Notebooks support special commands that start with `#!`:

```csharp
#!about  // Shows information about the kernel
```

```csharp
#!time   // Times the next cell execution

Thread.Sleep(1000);
Console.WriteLine("Done!");
```

Output:
```
Done!
Wall time: 1001.2ms
```

### Sharing Variables Between Languages

One of the most powerful features is polyglot support—mixing languages in the same notebook:

```csharp
// C# cell
var greeting = "Hello from C#!";
```

```fsharp
#!fsharp
// F# cell - access the C# variable
#!share --from csharp greeting
printfn "%s And hello from F#!" greeting
```

This is particularly useful when you want to use F#'s powerful data manipulation libraries alongside C# ML.NET code.

## DataFrame Basics with Microsoft.Data.Analysis

The `DataFrame` class from `Microsoft.Data.Analysis` is the C# equivalent of Python's pandas DataFrame. It provides a tabular data structure with labeled columns, supporting various data types and operations.

### Creating DataFrames

There are several ways to create a DataFrame:

**From columns:**

```csharp
using Microsoft.Data.Analysis;

// Create individual columns
var names = new StringDataFrameColumn("Name", new[] { "Alice", "Bob", "Charlie" });
var ages = new Int32DataFrameColumn("Age", new[] { 28, 35, 42 });
var salaries = new DoubleDataFrameColumn("Salary", new[] { 75000.0, 82000.0, 95000.0 });

// Combine into DataFrame
var employees = new DataFrame(names, ages, salaries);
employees
```

**From CSV files:**

```csharp
// Load from a CSV file
var df = DataFrame.LoadCsv("data/employees.csv");

// With specific options
var df = DataFrame.LoadCsv(
    "data/employees.csv",
    separator: ',',
    header: true,
    encoding: Encoding.UTF8
);
```

**From IDataView (ML.NET integration):**

```csharp
var mlContext = new MLContext();
IDataView dataView = mlContext.Data.LoadFromTextFile<EmployeeData>("data/employees.csv");

// Convert to DataFrame for exploration
var df = dataView.ToDataFrame();
```

### Exploring DataFrames

Once you have a DataFrame, exploration is straightforward:

```csharp
// Basic information
Console.WriteLine($"Rows: {df.Rows.Count}");
Console.WriteLine($"Columns: {df.Columns.Count}");

// Column names and types
foreach (var col in df.Columns)
{
    Console.WriteLine($"{col.Name}: {col.DataType.Name}");
}
```

Output:
```
Rows: 3
Columns: 3
Name: String
Age: Int32
Salary: Double
```

**First and last rows:**

```csharp
// First 5 rows
df.Head(5)

// Last 5 rows
df.Tail(5)
```

**Statistical summary:**

```csharp
// Description of numeric columns
df.Description()
```

This returns a new DataFrame with statistics:

| | Age | Salary |
|------|------|--------|
| Length | 3 | 3 |
| Mean | 35 | 84000 |
| Min | 28 | 75000 |
| Max | 42 | 95000 |

### Selecting and Filtering Data

**Column selection:**

```csharp
// Single column (returns DataFrameColumn)
var ageColumn = df["Age"];

// Multiple columns (returns DataFrame)
var subset = df[new[] { "Name", "Salary" }];
```

**Row filtering:**

```csharp
// Filter using boolean conditions
var highEarners = df.Filter(df["Salary"].ElementwiseGreaterThan(80000));

// Alternative with LINQ-style API
var seniors = df.Rows
    .Where(row => (int)row["Age"] > 40)
    .ToList();
```

**Sorting:**

```csharp
// Sort by salary descending
var sorted = df.OrderByDescending("Salary");

// Sort by multiple columns
var sorted = df.OrderBy("Age").ThenByDescending("Salary");
```

### Transforming Data

**Adding columns:**

```csharp
// Calculated column
var bonus = df["Salary"].Multiply(0.1);
df.Columns.Add(new DoubleDataFrameColumn("Bonus", bonus.Cast<double>()));

// Or with explicit values
df["Department"] = new StringDataFrameColumn("Department", 
    new[] { "Engineering", "Sales", "Engineering" });
```

**Modifying values:**

```csharp
// Apply transformation to a column
var adjustedSalary = df["Salary"].Add(5000);
df["Salary"] = adjustedSalary;
```

**Handling missing values:**

```csharp
// Check for nulls
var nullCount = df["Salary"].NullCount;

// Fill missing values
df["Salary"].FillNulls(0, inPlace: true);

// Drop rows with any null
df.DropNulls();
```

### GroupBy and Aggregation

```csharp
// Group by department
var grouped = df.GroupBy("Department");

// Aggregate
var summary = grouped.Sum("Salary");
var avgAge = grouped.Mean("Age");
```

## Your First Data Exploration in C#

Let's put everything together with a real exploration workflow. We'll work with a classic dataset—the Iris dataset—to demonstrate a complete exploratory data analysis (EDA) pipeline.

### Setting Up the Notebook

Create a new notebook called `iris-exploration.dib`:

```csharp
#r "nuget: Microsoft.Data.Analysis, 0.22.0"
#r "nuget: Microsoft.ML, 5.0.0"
#r "nuget: XPlot.Plotly, 4.1.0"

using Microsoft.Data.Analysis;
using Microsoft.ML;
using XPlot.Plotly;
using System.Linq;
```

### Loading the Data

ML.NET includes several sample datasets. Let's load the Iris dataset:

```csharp
// Download the Iris dataset
var irisUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
var localPath = "iris.csv";

// Add headers since the original file doesn't have them
var headers = "SepalLength,SepalWidth,PetalLength,PetalWidth,Species";

using (var client = new HttpClient())
{
    var data = await client.GetStringAsync(irisUrl);
    // Prepend headers
    await File.WriteAllTextAsync(localPath, headers + "\n" + data);
}

Console.WriteLine("Dataset downloaded successfully!");
```

Now load it into a DataFrame:

```csharp
var iris = DataFrame.LoadCsv(localPath);
iris.Head(10)
```

[SCREENSHOT: DataFrame display showing first 10 rows of Iris dataset with SepalLength, SepalWidth, PetalLength, PetalWidth, and Species columns]

### Initial Exploration

```csharp
Console.WriteLine($"Dataset shape: {iris.Rows.Count} rows × {iris.Columns.Count} columns");
Console.WriteLine("\nColumn types:");
foreach (var col in iris.Columns)
{
    Console.WriteLine($"  {col.Name}: {col.DataType.Name} ({col.NullCount} nulls)");
}
```

Output:
```
Dataset shape: 150 rows × 5 columns

Column types:
  SepalLength: Single (0 nulls)
  SepalWidth: Single (0 nulls)
  PetalLength: Single (0 nulls)
  PetalWidth: Single (0 nulls)
  Species: String (0 nulls)
```

### Statistical Summary

```csharp
// Get summary statistics for numeric columns
iris.Description()
```

| | SepalLength | SepalWidth | PetalLength | PetalWidth |
|------|-------------|------------|-------------|------------|
| Length | 150 | 150 | 150 | 150 |
| Mean | 5.84 | 3.05 | 3.76 | 1.20 |
| StdDev | 0.83 | 0.43 | 1.76 | 0.76 |
| Min | 4.3 | 2.0 | 1.0 | 0.1 |
| Max | 7.9 | 4.4 | 6.9 | 2.5 |

### Visualizing Data

Create visualizations using XPlot.Plotly:

**Histogram:**

```csharp
var sepalLengths = iris["SepalLength"].Cast<float>().ToArray();

var histogram = Chart.Plot(
    new Histogram
    {
        x = sepalLengths,
        name = "Sepal Length Distribution",
        marker = new Marker { color = "steelblue" }
    }
);

histogram.WithLayout(new Layout.Layout
{
    title = "Distribution of Sepal Length",
    xaxis = new Xaxis { title = "Sepal Length (cm)" },
    yaxis = new Yaxis { title = "Frequency" }
});

histogram
```

[SCREENSHOT: Histogram showing distribution of Sepal Length values with bars centered around 5.5-6.0 cm]

**Scatter plot:**

```csharp
// Group by species for colored scatter plot
var species = iris["Species"].Cast<string>().Distinct().ToArray();
var traces = new List<Scattergl>();

foreach (var sp in species)
{
    var subset = iris.Filter(iris["Species"].ElementwiseEquals(sp));
    traces.Add(new Scattergl
    {
        x = subset["SepalLength"].Cast<float>().ToArray(),
        y = subset["PetalLength"].Cast<float>().ToArray(),
        mode = "markers",
        name = sp
    });
}

var scatter = Chart.Plot(traces);
scatter.WithLayout(new Layout.Layout
{
    title = "Sepal Length vs Petal Length by Species",
    xaxis = new Xaxis { title = "Sepal Length (cm)" },
    yaxis = new Yaxis { title = "Petal Length (cm)" }
});

scatter
```

[SCREENSHOT: Scatter plot showing three distinct clusters for Iris-setosa, Iris-versicolor, and Iris-virginica species]

### Species Distribution

```csharp
var speciesCounts = iris["Species"]
    .Cast<string>()
    .GroupBy(s => s)
    .Select(g => new { Species = g.Key, Count = g.Count() });

foreach (var s in speciesCounts)
{
    Console.WriteLine($"{s.Species}: {s.Count}");
}
```

Output:
```
Iris-setosa: 50
Iris-versicolor: 50
Iris-virginica: 50
```

## Project: Hands-On Dataset Exploration

Now it's time to apply what you've learned. In this project, you'll explore the California Housing dataset—a real-world dataset with more complexity than Iris.

### Project Goals

1. Load and inspect the dataset
2. Handle data quality issues
3. Create summary statistics
4. Visualize key relationships
5. Prepare insights for modeling (which we'll do in Chapter 4)

### Step 1: Download and Load the Data

Create a new notebook called `housing-exploration.dib`:

```csharp
#r "nuget: Microsoft.Data.Analysis, 0.22.0"
#r "nuget: Microsoft.ML, 5.0.0"
#r "nuget: XPlot.Plotly, 4.1.0"

using Microsoft.Data.Analysis;
using Microsoft.ML;
using XPlot.Plotly;
using System.Net.Http;
using System.IO;
```

Download the dataset:

```csharp
var housingUrl = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv";
var localPath = "housing.csv";

using (var client = new HttpClient())
{
    var data = await client.GetStringAsync(housingUrl);
    await File.WriteAllTextAsync(localPath, data);
}

var housing = DataFrame.LoadCsv(localPath);
Console.WriteLine($"Loaded {housing.Rows.Count} rows");
housing.Head(5)
```

[SCREENSHOT: DataFrame showing housing data with columns like longitude, latitude, housing_median_age, total_rooms, etc.]

### Step 2: Initial Data Assessment

```csharp
Console.WriteLine("=== Dataset Overview ===\n");
Console.WriteLine($"Shape: {housing.Rows.Count} rows × {housing.Columns.Count} columns\n");

Console.WriteLine("Columns:");
foreach (var col in housing.Columns)
{
    var nullPct = (col.NullCount * 100.0 / housing.Rows.Count).ToString("F1");
    Console.WriteLine($"  {col.Name,-25} {col.DataType.Name,-10} {col.NullCount} nulls ({nullPct}%)");
}
```

Output:
```
=== Dataset Overview ===

Shape: 20640 rows × 10 columns

Columns:
  longitude                 Single     0 nulls (0.0%)
  latitude                  Single     0 nulls (0.0%)
  housing_median_age        Single     0 nulls (0.0%)
  total_rooms               Single     0 nulls (0.0%)
  total_bedrooms            Single     207 nulls (1.0%)
  population                Single     0 nulls (0.0%)
  households                Single     0 nulls (0.0%)
  median_income             Single     0 nulls (0.0%)
  median_house_value        Single     0 nulls (0.0%)
  ocean_proximity           String     0 nulls (0.0%)
```

Notice that `total_bedrooms` has 207 missing values—about 1% of the data. We'll need to handle this.

### Step 3: Statistical Summary

```csharp
housing.Description()
```

This gives us a comprehensive statistical overview. Key observations:
- `median_house_value` ranges from $14,999 to $500,001
- `median_income` is scaled (roughly $10,000 units)
- Geographic coordinates cover California

### Step 4: Handle Missing Values

```csharp
// Check the missing values
var missingBedrooms = housing["total_bedrooms"].NullCount;
Console.WriteLine($"Missing total_bedrooms: {missingBedrooms}");

// Strategy: Fill with median value
var bedroomValues = housing["total_bedrooms"]
    .Cast<float?>()
    .Where(v => v.HasValue)
    .Select(v => v.Value)
    .ToArray();

var medianBedrooms = bedroomValues.OrderBy(v => v).ElementAt(bedroomValues.Length / 2);
Console.WriteLine($"Median total_bedrooms: {medianBedrooms}");

// Fill nulls
housing["total_bedrooms"].FillNulls((float)medianBedrooms, inPlace: true);
Console.WriteLine($"After fill - Missing: {housing["total_bedrooms"].NullCount}");
```

### Step 5: Feature Engineering

Create useful derived features:

```csharp
// Rooms per household
var roomsPerHousehold = housing["total_rooms"]
    .Divide(housing["households"]);
housing.Columns.Add(new SingleDataFrameColumn("rooms_per_household", 
    roomsPerHousehold.Cast<float>()));

// Bedrooms per room
var bedroomsPerRoom = housing["total_bedrooms"]
    .Divide(housing["total_rooms"]);
housing.Columns.Add(new SingleDataFrameColumn("bedrooms_per_room", 
    bedroomsPerRoom.Cast<float>()));

// Population per household
var popPerHousehold = housing["population"]
    .Divide(housing["households"]);
housing.Columns.Add(new SingleDataFrameColumn("population_per_household", 
    popPerHousehold.Cast<float>()));

Console.WriteLine("New features added:");
foreach (var col in housing.Columns.TakeLast(3))
{
    Console.WriteLine($"  {col.Name}");
}
```

### Step 6: Visualize Key Relationships

**Income vs House Value:**

```csharp
var incomeVsValue = Chart.Plot(
    new Scattergl
    {
        x = housing["median_income"].Cast<float>().ToArray(),
        y = housing["median_house_value"].Cast<float>().ToArray(),
        mode = "markers",
        marker = new Marker 
        { 
            size = 3, 
            color = "rgba(31, 119, 180, 0.5)" 
        }
    }
);

incomeVsValue.WithLayout(new Layout.Layout
{
    title = "Median Income vs Median House Value",
    xaxis = new Xaxis { title = "Median Income ($10k)" },
    yaxis = new Yaxis { title = "Median House Value ($)" }
});

incomeVsValue
```

[SCREENSHOT: Scatter plot showing strong positive correlation between income and house value, with visible cap at $500,000]

**Geographic Distribution:**

```csharp
var geoPlot = Chart.Plot(
    new Scattergl
    {
        x = housing["longitude"].Cast<float>().ToArray(),
        y = housing["latitude"].Cast<float>().ToArray(),
        mode = "markers",
        marker = new Marker 
        { 
            size = 3,
            color = housing["median_house_value"].Cast<float>().ToArray(),
            colorscale = "Viridis",
            showscale = true
        }
    }
);

geoPlot.WithLayout(new Layout.Layout
{
    title = "California Housing Prices by Location",
    xaxis = new Xaxis { title = "Longitude" },
    yaxis = new Yaxis { title = "Latitude" }
});

geoPlot
```

[SCREENSHOT: Geographic scatter plot of California showing housing prices, with highest values visible along the coast near San Francisco and Los Angeles]

### Step 7: Correlation Analysis

```csharp
// Calculate correlations with median_house_value
var target = housing["median_house_value"].Cast<float>().ToArray();

Console.WriteLine("Correlation with median_house_value:\n");

foreach (var col in housing.Columns)
{
    if (col.DataType == typeof(float) || col.DataType == typeof(Single))
    {
        if (col.Name == "median_house_value") continue;
        
        var values = housing[col.Name].Cast<float>().ToArray();
        var correlation = CalculateCorrelation(target, values);
        Console.WriteLine($"{col.Name,-25}: {correlation:F3}");
    }
}

// Helper function for Pearson correlation
static double CalculateCorrelation(float[] x, float[] y)
{
    var n = x.Length;
    var sumX = x.Sum();
    var sumY = y.Sum();
    var sumXY = x.Zip(y, (a, b) => a * b).Sum();
    var sumX2 = x.Sum(a => a * a);
    var sumY2 = y.Sum(b => b * b);
    
    var numerator = n * sumXY - sumX * sumY;
    var denominator = Math.Sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    
    return numerator / denominator;
}
```

Output:
```
Correlation with median_house_value:

longitude                : -0.046
latitude                 : -0.144
housing_median_age       :  0.106
total_rooms              :  0.134
total_bedrooms           :  0.049
population               : -0.025
households               :  0.065
median_income            :  0.688
rooms_per_household      :  0.151
bedrooms_per_room        : -0.256
population_per_household : -0.023
```

**Key Insight:** `median_income` has the strongest correlation (0.688) with house value—this will be our most predictive feature for modeling.

### Step 8: Save Your Processed Data

```csharp
// Save the cleaned and enhanced dataset
DataFrame.SaveCsv(housing, "housing_processed.csv");
Console.WriteLine("Processed dataset saved to housing_processed.csv");
Console.WriteLine($"Final shape: {housing.Rows.Count} rows × {housing.Columns.Count} columns");
```

### Project Summary

In this exploration, you:

1. **Loaded** a real-world dataset with 20,640 records
2. **Identified** data quality issues (207 missing values in `total_bedrooms`)
3. **Imputed** missing values using the median strategy
4. **Engineered** 3 new features that capture meaningful relationships
5. **Visualized** the geographic distribution and key correlations
6. **Discovered** that `median_income` is the strongest predictor of house value

This processed dataset is now ready for machine learning, which we'll tackle in Chapter 4.

## Summary

In this chapter, you've set up a complete data science environment for C#:

- **IDE Setup**: Configured both Visual Studio and VS Code with essential extensions
- **Package Installation**: Added ML.NET 4.0 and Microsoft.Data.Analysis to your toolkit
- **Polyglot Notebooks**: Learned to work interactively with .NET's answer to Jupyter
- **DataFrame Operations**: Mastered loading, filtering, transforming, and aggregating data
- **Visualization**: Created histograms, scatter plots, and geographic visualizations
- **Hands-On Project**: Completed a full exploratory analysis of the California Housing dataset

You now have the foundation to explore any dataset using familiar C# syntax and tooling. In the next chapter, we'll dive deeper into data preprocessing and feature engineering—the crucial steps that determine model success.

## Exercises

1. **Extension Exploration**: Install the Data Wrangler extension in VS Code and use it to visually explore the housing dataset. How does it compare to the DataFrame approach?

2. **Notebook Practice**: Create a Polyglot Notebook that mixes C# and F# cells. Use F# for statistical calculations and C# for visualization.

3. **Data Cleaning Challenge**: Download a messy dataset (try Kaggle's Titanic dataset) and practice handling missing values, outliers, and data type conversions.

4. **Visualization Gallery**: Create at least three different visualization types (bar chart, box plot, heatmap) using XPlot.Plotly with the housing dataset.

5. **Performance Comparison**: Load a large CSV file (>100MB) using both `DataFrame.LoadCsv` and ML.NET's `LoadFromTextFile`. Measure and compare the loading times.

---

*Next Chapter: Data Preprocessing and Feature Engineering →*
