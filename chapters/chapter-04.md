# Chapter 4: Data Wrangling in C#

If data science is alchemy, data wrangling is the unglamorous work of mining ore before you can even dream of transmuting lead into gold. Some estimates suggest data scientists spend 60-80% of their time cleaning and preparing data. That's not a failure of process—it's the reality of working with messy, real-world information.

The good news? This is where C# developers shine. The same instincts that make you write null checks, validate inputs, and handle edge cases in application code translate directly to data cleaning. You're not learning a new discipline; you're applying familiar engineering rigor to a new domain.

In this chapter, we'll build the foundational skills for loading, cleaning, and transforming data in C#. We'll work with `Microsoft.Data.Analysis` for tabular data manipulation, explore ML.NET's data pipeline architecture, and get our hands dirty with a real-world messy dataset.

## The Data Loading Landscape

Before you can clean data, you need to get it into your application. Data arrives in many forms: flat files, databases, APIs, and streaming sources. Let's start with the most common formats.

### Loading CSV Files

CSV is the lingua franca of data exchange—simple, human-readable, and universally supported. The `Microsoft.Data.Analysis` package provides the `DataFrame` class, which gives you Pandas-like functionality in C#.

First, add the required packages:

```bash
dotnet add package Microsoft.Data.Analysis
dotnet add package Microsoft.ML
```

Here's the basic pattern for loading a CSV:

```csharp
using Microsoft.Data.Analysis;

// Simple CSV loading
DataFrame df = DataFrame.LoadCsv("customers.csv");

// View basic info
Console.WriteLine($"Rows: {df.Rows.Count}, Columns: {df.Columns.Count}");
Console.WriteLine($"Columns: {string.Join(", ", df.Columns.Select(c => c.Name))}");

// Preview first few rows
Console.WriteLine(df.Head(5));
```

In Pandas, this would be `pd.read_csv('customers.csv')`. The C# version is more verbose, but the core concept is identical.

Real-world CSVs are rarely clean. Here's a more robust loader that handles common issues:

```csharp
public static DataFrame LoadCsvRobust(string path, char separator = ',')
{
    if (!File.Exists(path))
        throw new FileNotFoundException($"CSV file not found: {path}");
    
    var fileInfo = new FileInfo(path);
    if (fileInfo.Length == 0)
        throw new InvalidDataException("CSV file is empty");
    
    try
    {
        // LoadCsv handles headers, type inference, and basic parsing
        return DataFrame.LoadCsv(
            path,
            separator: separator,
            header: true,
            guessRows: 100  // Sample more rows for better type inference
        );
    }
    catch (Exception ex)
    {
        throw new InvalidDataException(
            $"Failed to parse CSV '{path}': {ex.Message}", ex);
    }
}
```

For large files, consider streaming:

```csharp
public static IEnumerable<Dictionary<string, string>> StreamCsv(string path)
{
    using var reader = new StreamReader(path);
    
    // Read header
    var headerLine = reader.ReadLine();
    if (headerLine == null) yield break;
    
    var headers = ParseCsvLine(headerLine);
    
    // Stream data rows
    while (!reader.EndOfStream)
    {
        var line = reader.ReadLine();
        if (string.IsNullOrWhiteSpace(line)) continue;
        
        var values = ParseCsvLine(line);
        var row = new Dictionary<string, string>();
        
        for (int i = 0; i < Math.Min(headers.Length, values.Length); i++)
        {
            row[headers[i]] = values[i];
        }
        
        yield return row;
    }
}

private static string[] ParseCsvLine(string line)
{
    // Simple parser - for production, use CsvHelper or similar
    var result = new List<string>();
    var current = new StringBuilder();
    bool inQuotes = false;
    
    foreach (char c in line)
    {
        if (c == '"')
            inQuotes = !inQuotes;
        else if (c == ',' && !inQuotes)
        {
            result.Add(current.ToString().Trim());
            current.Clear();
        }
        else
            current.Append(c);
    }
    result.Add(current.ToString().Trim());
    
    return result.ToArray();
}
```

For production CSV handling with complex cases (embedded quotes, multi-line fields), use the `CsvHelper` library:

```csharp
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;

public static List<T> LoadCsvTyped<T>(string path)
{
    var config = new CsvConfiguration(CultureInfo.InvariantCulture)
    {
        HasHeaderRecord = true,
        MissingFieldFound = null,  // Ignore missing fields
        HeaderValidated = null,     // Skip validation
        BadDataFound = context =>   // Log bad data instead of throwing
        {
            Console.WriteLine($"Bad data at row {context.Context.Parser.Row}: {context.RawRecord}");
        }
    };
    
    using var reader = new StreamReader(path);
    using var csv = new CsvReader(reader, config);
    
    return csv.GetRecords<T>().ToList();
}
```

### Loading JSON Data

JSON is ubiquitous in modern applications. .NET's `System.Text.Json` handles this elegantly:

```csharp
using System.Text.Json;

public class Customer
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public string? Email { get; set; }
    public DateTime CreatedAt { get; set; }
    public List<Order> Orders { get; set; } = new();
}

public class Order
{
    public int OrderId { get; set; }
    public decimal Amount { get; set; }
    public string Status { get; set; } = "";
}

// Load strongly-typed JSON
public static async Task<List<Customer>> LoadCustomersAsync(string path)
{
    var options = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,  // Match "name" to Name
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true
    };
    
    await using var stream = File.OpenRead(path);
    return await JsonSerializer.DeserializeAsync<List<Customer>>(stream, options) 
           ?? new List<Customer>();
}
```

For JSON with unpredictable schemas, use `JsonDocument` for dynamic access:

```csharp
public static DataFrame JsonToDataFrame(string jsonPath)
{
    var json = File.ReadAllText(jsonPath);
    using var doc = JsonDocument.Parse(json);
    
    var columns = new Dictionary<string, List<object?>>();
    
    foreach (var element in doc.RootElement.EnumerateArray())
    {
        foreach (var property in element.EnumerateObject())
        {
            if (!columns.ContainsKey(property.Name))
                columns[property.Name] = new List<object?>();
            
            columns[property.Name].Add(GetJsonValue(property.Value));
        }
        
        // Fill missing values with null
        foreach (var col in columns.Keys)
        {
            if (columns[col].Count < columns.Values.Max(v => v.Count))
                columns[col].Add(null);
        }
    }
    
    // Convert to DataFrame
    var df = new DataFrame();
    foreach (var (name, values) in columns)
    {
        df.Columns.Add(CreateColumn(name, values));
    }
    
    return df;
}

private static object? GetJsonValue(JsonElement element) => element.ValueKind switch
{
    JsonValueKind.String => element.GetString(),
    JsonValueKind.Number => element.TryGetInt64(out var l) ? l : element.GetDouble(),
    JsonValueKind.True => true,
    JsonValueKind.False => false,
    JsonValueKind.Null => null,
    _ => element.ToString()
};
```

### Loading from Databases with Entity Framework

Most enterprise data lives in databases. Entity Framework Core makes this straightforward:

```csharp
using Microsoft.EntityFrameworkCore;

public class SalesContext : DbContext
{
    public DbSet<Product> Products { get; set; }
    public DbSet<Sale> Sales { get; set; }
    public DbSet<Customer> Customers { get; set; }
    
    protected override void OnConfiguring(DbContextOptionsBuilder options)
    {
        options.UseSqlServer("Server=localhost;Database=Sales;Trusted_Connection=true;");
        // Or for SQLite: options.UseSqlite("Data Source=sales.db");
    }
}

public class Product
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public string Category { get; set; } = "";
    public decimal Price { get; set; }
}

public class Sale
{
    public int Id { get; set; }
    public int ProductId { get; set; }
    public int CustomerId { get; set; }
    public int Quantity { get; set; }
    public DateTime SaleDate { get; set; }
    public decimal TotalAmount { get; set; }
    
    public Product Product { get; set; } = null!;
    public Customer Customer { get; set; } = null!;
}
```

Load and transform data using LINQ—this is where C# really shines:

```csharp
public static DataFrame LoadSalesDataForAnalysis()
{
    using var context = new SalesContext();
    
    // Query with joins and projections
    var salesData = context.Sales
        .Include(s => s.Product)
        .Include(s => s.Customer)
        .Select(s => new
        {
            s.Id,
            s.SaleDate,
            ProductName = s.Product.Name,
            ProductCategory = s.Product.Category,
            CustomerName = s.Customer.Name,
            s.Quantity,
            s.TotalAmount,
            UnitPrice = s.TotalAmount / s.Quantity
        })
        .ToList();
    
    // Convert to DataFrame
    var df = new DataFrame();
    df.Columns.Add(new Int32DataFrameColumn("Id", salesData.Select(s => s.Id)));
    df.Columns.Add(new PrimitiveDataFrameColumn<DateTime>("SaleDate", 
        salesData.Select(s => s.SaleDate)));
    df.Columns.Add(new StringDataFrameColumn("ProductName", 
        salesData.Select(s => s.ProductName)));
    df.Columns.Add(new StringDataFrameColumn("Category", 
        salesData.Select(s => s.ProductCategory)));
    df.Columns.Add(new Int32DataFrameColumn("Quantity", 
        salesData.Select(s => s.Quantity)));
    df.Columns.Add(new DecimalDataFrameColumn("TotalAmount", 
        salesData.Select(s => s.TotalAmount)));
    
    return df;
}
```

For read-heavy analytics, use `AsNoTracking()` to skip change tracking overhead:

```csharp
var data = context.Sales
    .AsNoTracking()  // 30-50% faster for read-only queries
    .Where(s => s.SaleDate >= startDate)
    .ToList();
```

### Loading from APIs

Modern applications often pull data from REST APIs. Here's a pattern for loading API data into analysis-ready structures:

```csharp
using System.Net.Http.Json;

public class ApiDataLoader
{
    private readonly HttpClient _client;
    private readonly JsonSerializerOptions _jsonOptions;
    
    public ApiDataLoader(string baseUrl, string? apiKey = null)
    {
        _client = new HttpClient { BaseAddress = new Uri(baseUrl) };
        
        if (!string.IsNullOrEmpty(apiKey))
            _client.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");
        
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true
        };
    }
    
    public async Task<List<T>> LoadPaginatedAsync<T>(
        string endpoint, 
        int pageSize = 100,
        int maxPages = int.MaxValue)
    {
        var allResults = new List<T>();
        int page = 1;
        
        while (page <= maxPages)
        {
            var url = $"{endpoint}?page={page}&limit={pageSize}";
            var response = await _client.GetFromJsonAsync<ApiResponse<T>>(url, _jsonOptions);
            
            if (response?.Data == null || response.Data.Count == 0)
                break;
            
            allResults.AddRange(response.Data);
            
            if (response.Data.Count < pageSize)
                break;  // Last page
            
            page++;
            await Task.Delay(100);  // Rate limiting
        }
        
        return allResults;
    }
}

public class ApiResponse<T>
{
    public List<T> Data { get; set; } = new();
    public int Total { get; set; }
    public int Page { get; set; }
}

// Usage
var loader = new ApiDataLoader("https://api.example.com/v1", apiKey);
var transactions = await loader.LoadPaginatedAsync<Transaction>("/transactions");
```

## Cleaning Data: The Heart of Data Wrangling

Loading data is just the beginning. Real-world data is messy: missing values, duplicates, inconsistent formats, outliers, and errors. Let's tackle each systematically.

### Handling Missing Values (Nulls)

Missing data is inevitable. The question is how to handle it. Here are the main strategies:

```csharp
public static class DataCleaningExtensions
{
    // Check for missing values
    public static void PrintMissingValueReport(this DataFrame df)
    {
        Console.WriteLine("Missing Value Report:");
        Console.WriteLine("".PadRight(50, '-'));
        
        foreach (var col in df.Columns)
        {
            long nullCount = col.NullCount;
            double pct = (double)nullCount / df.Rows.Count * 100;
            Console.WriteLine($"{col.Name,-25} {nullCount,8} ({pct:F1}%)");
        }
    }
    
    // Drop rows with any null values
    public static DataFrame DropNulls(this DataFrame df)
    {
        var validRows = new List<long>();
        
        for (long i = 0; i < df.Rows.Count; i++)
        {
            bool hasNull = false;
            foreach (var col in df.Columns)
            {
                if (col[i] == null)
                {
                    hasNull = true;
                    break;
                }
            }
            if (!hasNull) validRows.Add(i);
        }
        
        return df.Filter(CreateBoolColumn(df.Rows.Count, validRows));
    }
    
    // Fill numeric columns with mean/median
    public static void FillNumericWithMean(this DataFrame df)
    {
        for (int i = 0; i < df.Columns.Count; i++)
        {
            var col = df.Columns[i];
            if (col is PrimitiveDataFrameColumn<double> doubleCol)
            {
                var nonNullValues = Enumerable.Range(0, (int)doubleCol.Length)
                    .Where(j => doubleCol[j].HasValue)
                    .Select(j => doubleCol[j]!.Value)
                    .ToList();
                
                if (nonNullValues.Count > 0)
                {
                    double mean = nonNullValues.Average();
                    doubleCol.FillNulls(mean);
                }
            }
            else if (col is PrimitiveDataFrameColumn<float> floatCol)
            {
                var nonNullValues = Enumerable.Range(0, (int)floatCol.Length)
                    .Where(j => floatCol[j].HasValue)
                    .Select(j => floatCol[j]!.Value)
                    .ToList();
                
                if (nonNullValues.Count > 0)
                {
                    float mean = (float)nonNullValues.Average();
                    floatCol.FillNulls(mean);
                }
            }
        }
    }
    
    // Fill with forward/backward fill (good for time series)
    public static void ForwardFill<T>(this PrimitiveDataFrameColumn<T> column) 
        where T : unmanaged
    {
        T? lastValid = null;
        for (long i = 0; i < column.Length; i++)
        {
            if (column[i].HasValue)
                lastValid = column[i].Value;
            else if (lastValid.HasValue)
                column[i] = lastValid.Value;
        }
    }
    
    private static PrimitiveDataFrameColumn<bool> CreateBoolColumn(
        long length, IEnumerable<long> trueIndices)
    {
        var col = new PrimitiveDataFrameColumn<bool>("filter", length);
        var trueSet = new HashSet<long>(trueIndices);
        for (long i = 0; i < length; i++)
            col[i] = trueSet.Contains(i);
        return col;
    }
}
```

Here's a comparison to Pandas operations:

| Operation | Pandas | C# DataFrame |
|-----------|--------|--------------|
| Check nulls | `df.isnull().sum()` | `col.NullCount` |
| Drop nulls | `df.dropna()` | `df.DropNulls()` (custom) |
| Fill with value | `df.fillna(0)` | `col.FillNulls(0)` |
| Fill with mean | `df.fillna(df.mean())` | `FillNumericWithMean()` |

### Handling Duplicates

Duplicates can skew analysis. Here's how to identify and remove them:

```csharp
public static class DuplicateHandling
{
    // Find duplicate rows based on specific columns
    public static DataFrame RemoveDuplicates(this DataFrame df, params string[] keyColumns)
    {
        var seen = new HashSet<string>();
        var keepRows = new List<long>();
        
        for (long i = 0; i < df.Rows.Count; i++)
        {
            // Create a key from specified columns
            var key = string.Join("|", keyColumns.Select(col => 
                df.Columns[col][i]?.ToString() ?? "NULL"));
            
            if (seen.Add(key))
                keepRows.Add(i);
        }
        
        Console.WriteLine($"Removed {df.Rows.Count - keepRows.Count} duplicate rows");
        return FilterByIndices(df, keepRows);
    }
    
    // Find and report duplicates
    public static void ReportDuplicates(this DataFrame df, params string[] keyColumns)
    {
        var counts = new Dictionary<string, int>();
        
        for (long i = 0; i < df.Rows.Count; i++)
        {
            var key = string.Join("|", keyColumns.Select(col => 
                df.Columns[col][i]?.ToString() ?? "NULL"));
            
            counts[key] = counts.GetValueOrDefault(key, 0) + 1;
        }
        
        var duplicates = counts.Where(kv => kv.Value > 1)
                               .OrderByDescending(kv => kv.Value)
                               .Take(10);
        
        Console.WriteLine("Top duplicates:");
        foreach (var (key, count) in duplicates)
            Console.WriteLine($"  {key}: {count} occurrences");
    }
    
    private static DataFrame FilterByIndices(DataFrame df, List<long> indices)
    {
        var indexSet = new HashSet<long>(indices);
        var filter = new PrimitiveDataFrameColumn<bool>("filter", df.Rows.Count);
        for (long i = 0; i < df.Rows.Count; i++)
            filter[i] = indexSet.Contains(i);
        return df.Filter(filter);
    }
}
```

### Handling Outliers

Outliers can represent genuine extreme values or data errors. Here are common detection and treatment methods:

```csharp
public static class OutlierHandling
{
    // IQR method (most robust)
    public static (double lower, double upper) GetIQRBounds(
        IEnumerable<double> values, double multiplier = 1.5)
    {
        var sorted = values.Where(v => !double.IsNaN(v)).OrderBy(v => v).ToList();
        if (sorted.Count < 4) return (double.MinValue, double.MaxValue);
        
        double q1 = Percentile(sorted, 25);
        double q3 = Percentile(sorted, 75);
        double iqr = q3 - q1;
        
        return (q1 - multiplier * iqr, q3 + multiplier * iqr);
    }
    
    // Z-score method (assumes normal distribution)
    public static (double lower, double upper) GetZScoreBounds(
        IEnumerable<double> values, double threshold = 3.0)
    {
        var vals = values.Where(v => !double.IsNaN(v)).ToList();
        double mean = vals.Average();
        double stdDev = Math.Sqrt(vals.Average(v => Math.Pow(v - mean, 2)));
        
        return (mean - threshold * stdDev, mean + threshold * stdDev);
    }
    
    // Detect outliers in a column
    public static List<long> FindOutlierIndices(
        PrimitiveDataFrameColumn<double> column,
        OutlierMethod method = OutlierMethod.IQR)
    {
        var values = Enumerable.Range(0, (int)column.Length)
            .Where(i => column[i].HasValue)
            .Select(i => column[i]!.Value)
            .ToList();
        
        var (lower, upper) = method == OutlierMethod.IQR 
            ? GetIQRBounds(values) 
            : GetZScoreBounds(values);
        
        var outliers = new List<long>();
        for (long i = 0; i < column.Length; i++)
        {
            if (column[i].HasValue)
            {
                double val = column[i]!.Value;
                if (val < lower || val > upper)
                    outliers.Add(i);
            }
        }
        
        return outliers;
    }
    
    // Clip outliers to bounds (Winsorization)
    public static void ClipOutliers(
        PrimitiveDataFrameColumn<double> column,
        double? lowerBound = null,
        double? upperBound = null)
    {
        for (long i = 0; i < column.Length; i++)
        {
            if (column[i].HasValue)
            {
                double val = column[i]!.Value;
                if (lowerBound.HasValue && val < lowerBound.Value)
                    column[i] = lowerBound.Value;
                else if (upperBound.HasValue && val > upperBound.Value)
                    column[i] = upperBound.Value;
            }
        }
    }
    
    private static double Percentile(List<double> sorted, double percentile)
    {
        double index = (percentile / 100.0) * (sorted.Count - 1);
        int lower = (int)Math.Floor(index);
        int upper = (int)Math.Ceiling(index);
        
        if (lower == upper) return sorted[lower];
        
        return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
    }
}

public enum OutlierMethod { IQR, ZScore }
```

Usage example:

```csharp
// Find and handle outliers
var priceColumn = df.Columns["Price"] as PrimitiveDataFrameColumn<double>;
var outlierIndices = OutlierHandling.FindOutlierIndices(priceColumn!);
Console.WriteLine($"Found {outlierIndices.Count} outliers in Price column");

// Option 1: Remove outliers
// df = df.RemoveRows(outlierIndices);

// Option 2: Clip to bounds
var (lower, upper) = OutlierHandling.GetIQRBounds(
    Enumerable.Range(0, (int)priceColumn.Length)
        .Where(i => priceColumn[i].HasValue)
        .Select(i => priceColumn[i]!.Value));
        
OutlierHandling.ClipOutliers(priceColumn, lower, upper);
```

## Transforming Data

Once data is clean, we often need to transform it for analysis or model training. Let's cover the essential transformations.

### Normalization and Standardization

Many machine learning algorithms perform better when features are on similar scales:

```csharp
public static class Normalization
{
    // Min-Max scaling to [0, 1]
    public static PrimitiveDataFrameColumn<double> MinMaxScale(
        PrimitiveDataFrameColumn<double> column)
    {
        var values = GetNonNullValues(column);
        double min = values.Min();
        double max = values.Max();
        double range = max - min;
        
        var scaled = new PrimitiveDataFrameColumn<double>(column.Name + "_scaled", column.Length);
        
        for (long i = 0; i < column.Length; i++)
        {
            if (column[i].HasValue)
                scaled[i] = range != 0 ? (column[i]!.Value - min) / range : 0;
            else
                scaled[i] = null;
        }
        
        return scaled;
    }
    
    // Z-score standardization (mean=0, std=1)
    public static PrimitiveDataFrameColumn<double> Standardize(
        PrimitiveDataFrameColumn<double> column)
    {
        var values = GetNonNullValues(column);
        double mean = values.Average();
        double std = Math.Sqrt(values.Average(v => Math.Pow(v - mean, 2)));
        
        var standardized = new PrimitiveDataFrameColumn<double>(
            column.Name + "_standardized", column.Length);
        
        for (long i = 0; i < column.Length; i++)
        {
            if (column[i].HasValue)
                standardized[i] = std != 0 ? (column[i]!.Value - mean) / std : 0;
            else
                standardized[i] = null;
        }
        
        return standardized;
    }
    
    // Robust scaling using median and IQR (handles outliers better)
    public static PrimitiveDataFrameColumn<double> RobustScale(
        PrimitiveDataFrameColumn<double> column)
    {
        var sorted = GetNonNullValues(column).OrderBy(v => v).ToList();
        double median = Percentile(sorted, 50);
        double q1 = Percentile(sorted, 25);
        double q3 = Percentile(sorted, 75);
        double iqr = q3 - q1;
        
        var scaled = new PrimitiveDataFrameColumn<double>(column.Name + "_robust", column.Length);
        
        for (long i = 0; i < column.Length; i++)
        {
            if (column[i].HasValue)
                scaled[i] = iqr != 0 ? (column[i]!.Value - median) / iqr : 0;
            else
                scaled[i] = null;
        }
        
        return scaled;
    }
    
    private static List<double> GetNonNullValues(PrimitiveDataFrameColumn<double> column)
    {
        return Enumerable.Range(0, (int)column.Length)
            .Where(i => column[i].HasValue)
            .Select(i => column[i]!.Value)
            .ToList();
    }
    
    private static double Percentile(List<double> sorted, double p)
    {
        double index = (p / 100.0) * (sorted.Count - 1);
        int lower = (int)Math.Floor(index);
        int upper = Math.Min((int)Math.Ceiling(index), sorted.Count - 1);
        return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
    }
}
```

### Encoding Categorical Variables

Machine learning models need numeric inputs. Here's how to encode categorical variables:

```csharp
public static class CategoricalEncoding
{
    // One-Hot Encoding (creates binary columns for each category)
    public static List<PrimitiveDataFrameColumn<float>> OneHotEncode(
        StringDataFrameColumn column,
        int maxCategories = 100)
    {
        var categories = Enumerable.Range(0, (int)column.Length)
            .Where(i => column[i] != null)
            .Select(i => column[i]!)
            .Distinct()
            .Take(maxCategories)
            .ToList();
        
        var encodedColumns = new List<PrimitiveDataFrameColumn<float>>();
        
        foreach (var category in categories)
        {
            var encoded = new PrimitiveDataFrameColumn<float>(
                $"{column.Name}_{category}", column.Length);
            
            for (long i = 0; i < column.Length; i++)
                encoded[i] = column[i] == category ? 1f : 0f;
            
            encodedColumns.Add(encoded);
        }
        
        return encodedColumns;
    }
    
    // Label Encoding (maps categories to integers)
    public static (PrimitiveDataFrameColumn<int> encoded, Dictionary<string, int> mapping) 
        LabelEncode(StringDataFrameColumn column)
    {
        var mapping = new Dictionary<string, int>();
        var encoded = new PrimitiveDataFrameColumn<int>(column.Name + "_encoded", column.Length);
        
        int nextId = 0;
        for (long i = 0; i < column.Length; i++)
        {
            var value = column[i];
            if (value == null)
            {
                encoded[i] = -1;  // Use -1 for nulls
                continue;
            }
            
            if (!mapping.ContainsKey(value))
                mapping[value] = nextId++;
            
            encoded[i] = mapping[value];
        }
        
        return (encoded, mapping);
    }
    
    // Target Encoding (replaces categories with mean of target variable)
    public static PrimitiveDataFrameColumn<double> TargetEncode(
        StringDataFrameColumn categoryColumn,
        PrimitiveDataFrameColumn<double> targetColumn,
        double smoothing = 10)
    {
        // Calculate global mean
        var targetValues = Enumerable.Range(0, (int)targetColumn.Length)
            .Where(i => targetColumn[i].HasValue)
            .Select(i => targetColumn[i]!.Value)
            .ToList();
        double globalMean = targetValues.Average();
        
        // Calculate mean per category
        var categoryMeans = new Dictionary<string, (double sum, int count)>();
        
        for (long i = 0; i < categoryColumn.Length; i++)
        {
            var cat = categoryColumn[i];
            if (cat != null && targetColumn[i].HasValue)
            {
                if (!categoryMeans.ContainsKey(cat))
                    categoryMeans[cat] = (0, 0);
                
                var (sum, count) = categoryMeans[cat];
                categoryMeans[cat] = (sum + targetColumn[i]!.Value, count + 1);
            }
        }
        
        // Apply smoothed target encoding
        var encoded = new PrimitiveDataFrameColumn<double>(
            categoryColumn.Name + "_target_encoded", categoryColumn.Length);
        
        for (long i = 0; i < categoryColumn.Length; i++)
        {
            var cat = categoryColumn[i];
            if (cat == null || !categoryMeans.ContainsKey(cat))
            {
                encoded[i] = globalMean;
                continue;
            }
            
            var (sum, count) = categoryMeans[cat];
            double catMean = sum / count;
            
            // Smoothing: blend category mean with global mean based on sample size
            encoded[i] = (count * catMean + smoothing * globalMean) / (count + smoothing);
        }
        
        return encoded;
    }
}
```

### Creating Derived Features

Feature engineering often involves creating new columns from existing ones:

```csharp
public static class FeatureEngineering
{
    // Date features
    public static void AddDateFeatures(DataFrame df, string dateColumn)
    {
        var dates = df.Columns[dateColumn] as PrimitiveDataFrameColumn<DateTime>;
        if (dates == null) return;
        
        var year = new PrimitiveDataFrameColumn<int>(dateColumn + "_year", dates.Length);
        var month = new PrimitiveDataFrameColumn<int>(dateColumn + "_month", dates.Length);
        var dayOfWeek = new PrimitiveDataFrameColumn<int>(dateColumn + "_dayofweek", dates.Length);
        var isWeekend = new PrimitiveDataFrameColumn<bool>(dateColumn + "_isweekend", dates.Length);
        
        for (long i = 0; i < dates.Length; i++)
        {
            if (dates[i].HasValue)
            {
                var dt = dates[i]!.Value;
                year[i] = dt.Year;
                month[i] = dt.Month;
                dayOfWeek[i] = (int)dt.DayOfWeek;
                isWeekend[i] = dt.DayOfWeek == DayOfWeek.Saturday || 
                               dt.DayOfWeek == DayOfWeek.Sunday;
            }
        }
        
        df.Columns.Add(year);
        df.Columns.Add(month);
        df.Columns.Add(dayOfWeek);
        df.Columns.Add(isWeekend);
    }
    
    // Binning continuous variables
    public static PrimitiveDataFrameColumn<int> Bin(
        PrimitiveDataFrameColumn<double> column,
        double[] edges)  // e.g., [0, 18, 35, 50, 65, 100] for age groups
    {
        var binned = new PrimitiveDataFrameColumn<int>(column.Name + "_bin", column.Length);
        
        for (long i = 0; i < column.Length; i++)
        {
            if (!column[i].HasValue)
            {
                binned[i] = null;
                continue;
            }
            
            double val = column[i]!.Value;
            int bin = 0;
            for (int j = 1; j < edges.Length; j++)
            {
                if (val >= edges[j])
                    bin = j;
                else
                    break;
            }
            binned[i] = bin;
        }
        
        return binned;
    }
    
    // Interaction features
    public static PrimitiveDataFrameColumn<double> CreateInteraction(
        PrimitiveDataFrameColumn<double> col1,
        PrimitiveDataFrameColumn<double> col2)
    {
        var interaction = new PrimitiveDataFrameColumn<double>(
            $"{col1.Name}_x_{col2.Name}", col1.Length);
        
        for (long i = 0; i < col1.Length; i++)
        {
            if (col1[i].HasValue && col2[i].HasValue)
                interaction[i] = col1[i]!.Value * col2[i]!.Value;
            else
                interaction[i] = null;
        }
        
        return interaction;
    }
}
```

## ML.NET Data Pipeline Architecture

While `Microsoft.Data.Analysis` is great for exploration, ML.NET provides a powerful pipeline architecture designed for machine learning workflows. Understanding `IDataView` and transformers is essential for building production ML systems.

### Understanding IDataView

`IDataView` is ML.NET's core data abstraction—a lazy, immutable, cursor-based view of tabular data. Unlike `DataFrame`, it's designed for streaming large datasets that don't fit in memory:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 42);

// Load data into IDataView
IDataView dataView = mlContext.Data.LoadFromTextFile<HousingData>(
    path: "housing.csv",
    hasHeader: true,
    separatorChar: ',');

// IDataView is lazy - no data loaded yet!
// Data is only read when you enumerate

// Preview first rows (this reads data)
var preview = dataView.Preview(maxRows: 5);
foreach (var row in preview.RowView)
{
    Console.WriteLine(string.Join(", ", row.Values.Select(v => $"{v.Key}={v.Value}")));
}
```

Define your data schema with a class:

```csharp
public class HousingData
{
    [LoadColumn(0)] public float Size { get; set; }
    [LoadColumn(1)] public float Bedrooms { get; set; }
    [LoadColumn(2)] public float Age { get; set; }
    [LoadColumn(3)] public string Neighborhood { get; set; } = "";
    [LoadColumn(4)] public float Price { get; set; }  // Target
}
```

### Building Transform Pipelines

ML.NET pipelines compose transformations that process data from raw input to model-ready features:

```csharp
var pipeline = mlContext.Transforms
    // Handle missing values
    .ReplaceMissingValues(
        outputColumnName: "Size",
        replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
    
    // Normalize numeric features
    .Append(mlContext.Transforms.NormalizeMinMax("Size"))
    .Append(mlContext.Transforms.NormalizeMinMax("Bedrooms"))
    .Append(mlContext.Transforms.NormalizeMinMax("Age"))
    
    // Encode categorical features
    .Append(mlContext.Transforms.Categorical.OneHotEncoding("Neighborhood"))
    
    // Combine all features into a single vector
    .Append(mlContext.Transforms.Concatenate(
        "Features",
        "Size", "Bedrooms", "Age", "Neighborhood"));

// Fit the pipeline to learn transformation parameters
var transformer = pipeline.Fit(dataView);

// Transform the data
IDataView transformedData = transformer.Transform(dataView);
```

This pipeline:
1. Replaces missing values in `Size` with the column mean
2. Normalizes numeric columns to [0,1] range
3. One-hot encodes the `Neighborhood` categorical column
4. Concatenates all features into a single feature vector

### Common ML.NET Transformers

Here's a reference of the most useful transformers:

```csharp
// === MISSING VALUE HANDLING ===
mlContext.Transforms.ReplaceMissingValues("Column", 
    replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean);
    // Modes: Mean, Min, Max, DefaultValue

// === NORMALIZATION ===
mlContext.Transforms.NormalizeMinMax("Column");        // Scale to [0,1]
mlContext.Transforms.NormalizeMeanVariance("Column");  // Z-score
mlContext.Transforms.NormalizeLogMeanVariance("Column");
mlContext.Transforms.NormalizeBinning("Column");

// === CATEGORICAL ENCODING ===
mlContext.Transforms.Categorical.OneHotEncoding("Column");
mlContext.Transforms.Categorical.OneHotHashEncoding("Column", numberOfBits: 10);
mlContext.Transforms.Conversion.MapValueToKey("Column");  // Label encoding

// === TEXT PROCESSING ===
mlContext.Transforms.Text.FeaturizeText("Output", "TextColumn");
mlContext.Transforms.Text.TokenizeIntoWords("Output", "TextColumn");
mlContext.Transforms.Text.NormalizeText("Output", "TextColumn");

// === FEATURE COMBINATION ===
mlContext.Transforms.Concatenate("Features", "Col1", "Col2", "Col3");

// === COLUMN OPERATIONS ===
mlContext.Transforms.DropColumns("ColumnToDrop");
mlContext.Transforms.CopyColumns("NewColumn", "SourceColumn");
mlContext.Transforms.Conversion.ConvertType("Column", DataKind.Single);

// === CUSTOM TRANSFORMATIONS ===
mlContext.Transforms.CustomMapping<InputType, OutputType>(
    (input, output) => { /* custom logic */ },
    contractName: "MyCustomTransform");
```

### Custom Transformers

For complex transformations not covered by built-in transformers, create custom ones:

```csharp
public class CustomInputRow
{
    public float Value { get; set; }
    public string Category { get; set; } = "";
}

public class CustomOutputRow
{
    public float LogValue { get; set; }
    public bool IsHighValue { get; set; }
}

// In your pipeline
var customPipeline = mlContext.Transforms.CustomMapping<CustomInputRow, CustomOutputRow>(
    (input, output) =>
    {
        output.LogValue = (float)Math.Log(input.Value + 1);
        output.IsHighValue = input.Value > 1000;
    },
    contractName: "LogTransform");
```

### Saving and Loading Pipelines

Trained pipelines can be saved for production use:

```csharp
// Save the fitted transformer
mlContext.Model.Save(transformer, dataView.Schema, "data_pipeline.zip");

// Load in another application
DataViewSchema schema;
ITransformer loadedTransformer = mlContext.Model.Load("data_pipeline.zip", out schema);

// Apply to new data
IDataView newData = mlContext.Data.LoadFromTextFile<HousingData>("new_data.csv");
IDataView processedData = loadedTransformer.Transform(newData);
```

## Project: Cleaning a Real-World Messy Dataset

Let's put everything together with a realistic project. We'll clean the "Melbourne Housing" dataset—a commonly used dataset with plenty of real-world messiness.

### The Challenge

Our dataset has:
- Missing values in multiple columns
- Duplicate property listings
- Outliers in price and land size
- Mixed categorical formats (inconsistent naming)
- Dates that need parsing

### The Solution

```csharp
using Microsoft.Data.Analysis;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

namespace HousingDataCleaning;

public class HousingDataCleaner
{
    private readonly MLContext _mlContext;
    
    public HousingDataCleaner()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public async Task<DataFrame> CleanDataAsync(string inputPath, string outputPath)
    {
        Console.WriteLine("=== Melbourne Housing Data Cleaning Pipeline ===\n");
        
        // Step 1: Load the data
        Console.WriteLine("Step 1: Loading data...");
        var df = DataFrame.LoadCsv(inputPath);
        Console.WriteLine($"  Loaded {df.Rows.Count:N0} rows, {df.Columns.Count} columns\n");
        
        // Step 2: Initial data quality report
        Console.WriteLine("Step 2: Data Quality Report (Before Cleaning)");
        PrintDataQualityReport(df);
        
        // Step 3: Handle duplicates
        Console.WriteLine("\nStep 3: Removing duplicates...");
        long beforeCount = df.Rows.Count;
        df = RemoveDuplicates(df, "Address", "Suburb", "Price");
        Console.WriteLine($"  Removed {beforeCount - df.Rows.Count:N0} duplicate rows\n");
        
        // Step 4: Clean and standardize text columns
        Console.WriteLine("Step 4: Standardizing text columns...");
        CleanTextColumns(df);
        Console.WriteLine("  Standardized Suburb, Type, and Method columns\n");
        
        // Step 5: Handle missing values
        Console.WriteLine("Step 5: Handling missing values...");
        HandleMissingValues(df);
        
        // Step 6: Handle outliers
        Console.WriteLine("\nStep 6: Handling outliers...");
        HandleOutliers(df);
        
        // Step 7: Create derived features
        Console.WriteLine("\nStep 7: Creating derived features...");
        CreateDerivedFeatures(df);
        
        // Step 8: Final data quality report
        Console.WriteLine("\nStep 8: Data Quality Report (After Cleaning)");
        PrintDataQualityReport(df);
        
        // Step 9: Save cleaned data
        Console.WriteLine($"\nStep 9: Saving cleaned data to {outputPath}...");
        SaveDataFrame(df, outputPath);
        Console.WriteLine($"  Saved {df.Rows.Count:N0} rows\n");
        
        Console.WriteLine("=== Cleaning Complete ===");
        return df;
    }
    
    private void PrintDataQualityReport(DataFrame df)
    {
        Console.WriteLine("  Column                  Type          Nulls      Null%");
        Console.WriteLine("  " + new string('-', 60));
        
        foreach (var col in df.Columns)
        {
            string typeName = col.DataType.Name;
            long nullCount = col.NullCount;
            double nullPct = (double)nullCount / df.Rows.Count * 100;
            
            Console.WriteLine($"  {col.Name,-22} {typeName,-12} {nullCount,8:N0} {nullPct,8:F1}%");
        }
    }
    
    private DataFrame RemoveDuplicates(DataFrame df, params string[] keyColumns)
    {
        var seen = new HashSet<string>();
        var keepIndices = new List<long>();
        
        for (long i = 0; i < df.Rows.Count; i++)
        {
            var key = string.Join("|", keyColumns.Select(col => 
                df.Columns[col][i]?.ToString()?.ToLowerInvariant() ?? ""));
            
            if (seen.Add(key))
                keepIndices.Add(i);
        }
        
        return FilterByIndices(df, keepIndices);
    }
    
    private void CleanTextColumns(DataFrame df)
    {
        // Standardize suburb names (title case, trim)
        if (df.Columns["Suburb"] is StringDataFrameColumn suburbCol)
        {
            for (long i = 0; i < suburbCol.Length; i++)
            {
                if (suburbCol[i] != null)
                {
                    suburbCol[i] = CultureInfo.CurrentCulture.TextInfo
                        .ToTitleCase(suburbCol[i]!.ToLower().Trim());
                }
            }
        }
        
        // Standardize property type codes
        if (df.Columns["Type"] is StringDataFrameColumn typeCol)
        {
            var typeMapping = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                { "h", "House" },
                { "house", "House" },
                { "u", "Unit" },
                { "unit", "Unit" },
                { "t", "Townhouse" },
                { "townhouse", "Townhouse" }
            };
            
            for (long i = 0; i < typeCol.Length; i++)
            {
                if (typeCol[i] != null && typeMapping.TryGetValue(typeCol[i]!.Trim(), out var mapped))
                    typeCol[i] = mapped;
            }
        }
        
        // Standardize sale method
        if (df.Columns["Method"] is StringDataFrameColumn methodCol)
        {
            var methodMapping = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
            {
                { "S", "Sold" },
                { "SP", "SoldPrior" },
                { "PI", "PassedIn" },
                { "PN", "SoldNotDisclosed" },
                { "SN", "SoldNotDisclosed" },
                { "VB", "VendorBid" },
                { "SA", "SoldAfter" }
            };
            
            for (long i = 0; i < methodCol.Length; i++)
            {
                if (methodCol[i] != null && methodMapping.TryGetValue(methodCol[i]!.Trim(), out var mapped))
                    methodCol[i] = mapped;
            }
        }
    }
    
    private void HandleMissingValues(DataFrame df)
    {
        // Strategy 1: Drop rows where target (Price) is missing
        var priceCol = df.Columns["Price"];
        long priceNulls = priceCol.NullCount;
        if (priceNulls > 0)
        {
            Console.WriteLine($"  Dropping {priceNulls:N0} rows with missing Price");
            df = FilterWhereNotNull(df, "Price");
        }
        
        // Strategy 2: Fill numeric columns with median
        var numericColumns = new[] { "Rooms", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea" };
        foreach (var colName in numericColumns)
        {
            if (df.Columns[colName] is PrimitiveDataFrameColumn<double> col && col.NullCount > 0)
            {
                double median = CalculateMedian(col);
                Console.WriteLine($"  Filling {col.NullCount:N0} nulls in {colName} with median ({median:F1})");
                col.FillNulls(median);
            }
            else if (df.Columns[colName] is PrimitiveDataFrameColumn<float> colF && colF.NullCount > 0)
            {
                float median = (float)CalculateMedian(colF);
                Console.WriteLine($"  Filling {colF.NullCount:N0} nulls in {colName} with median ({median:F1})");
                colF.FillNulls(median);
            }
        }
        
        // Strategy 3: Fill categorical with mode
        if (df.Columns["Regionname"] is StringDataFrameColumn regionCol && regionCol.NullCount > 0)
        {
            string mode = CalculateMode(regionCol);
            Console.WriteLine($"  Filling {regionCol.NullCount:N0} nulls in Regionname with mode ({mode})");
            FillStringNulls(regionCol, mode);
        }
    }
    
    private void HandleOutliers(DataFrame df)
    {
        // Handle Price outliers using IQR method
        if (df.Columns["Price"] is PrimitiveDataFrameColumn<double> priceCol)
        {
            var values = GetNonNullValues(priceCol);
            var (lower, upper) = CalculateIQRBounds(values, 1.5);
            
            int clipped = 0;
            for (long i = 0; i < priceCol.Length; i++)
            {
                if (priceCol[i].HasValue)
                {
                    if (priceCol[i]!.Value > upper)
                    {
                        priceCol[i] = upper;
                        clipped++;
                    }
                    else if (priceCol[i]!.Value < lower)
                    {
                        priceCol[i] = lower;
                        clipped++;
                    }
                }
            }
            Console.WriteLine($"  Clipped {clipped:N0} Price outliers to [{lower:N0}, {upper:N0}]");
        }
        
        // Handle Landsize outliers (remove extreme values, likely errors)
        if (df.Columns["Landsize"] is PrimitiveDataFrameColumn<double> landCol)
        {
            var values = GetNonNullValues(landCol);
            var (lower, upper) = CalculateIQRBounds(values, 3.0);  // More lenient
            
            int clipped = 0;
            for (long i = 0; i < landCol.Length; i++)
            {
                if (landCol[i].HasValue && landCol[i]!.Value > upper)
                {
                    landCol[i] = null;  // Set extreme outliers to null
                    clipped++;
                }
            }
            Console.WriteLine($"  Nullified {clipped:N0} extreme Landsize outliers (>{upper:N0} sqm)");
            
            // Re-fill with median
            double median = CalculateMedian(landCol);
            landCol.FillNulls(median);
        }
    }
    
    private void CreateDerivedFeatures(DataFrame df)
    {
        // Price per room
        if (df.Columns["Price"] is PrimitiveDataFrameColumn<double> priceCol &&
            df.Columns["Rooms"] is PrimitiveDataFrameColumn<double> roomsCol)
        {
            var pricePerRoom = new PrimitiveDataFrameColumn<double>("PricePerRoom", df.Rows.Count);
            for (long i = 0; i < df.Rows.Count; i++)
            {
                if (priceCol[i].HasValue && roomsCol[i].HasValue && roomsCol[i]!.Value > 0)
                    pricePerRoom[i] = priceCol[i]!.Value / roomsCol[i]!.Value;
            }
            df.Columns.Add(pricePerRoom);
            Console.WriteLine("  Created PricePerRoom feature");
        }
        
        // Age of property (if YearBuilt available)
        if (df.Columns.Any(c => c.Name == "YearBuilt"))
        {
            var yearBuilt = df.Columns["YearBuilt"] as PrimitiveDataFrameColumn<double>;
            if (yearBuilt != null)
            {
                var age = new PrimitiveDataFrameColumn<double>("PropertyAge", df.Rows.Count);
                int currentYear = DateTime.Now.Year;
                for (long i = 0; i < df.Rows.Count; i++)
                {
                    if (yearBuilt[i].HasValue && yearBuilt[i]!.Value > 1800)
                        age[i] = currentYear - yearBuilt[i]!.Value;
                }
                df.Columns.Add(age);
                Console.WriteLine("  Created PropertyAge feature");
            }
        }
        
        // Is inner city (based on distance to CBD)
        if (df.Columns["Distance"] is PrimitiveDataFrameColumn<double> distCol)
        {
            var isInnerCity = new PrimitiveDataFrameColumn<bool>("IsInnerCity", df.Rows.Count);
            for (long i = 0; i < df.Rows.Count; i++)
            {
                if (distCol[i].HasValue)
                    isInnerCity[i] = distCol[i]!.Value <= 10;
            }
            df.Columns.Add(isInnerCity);
            Console.WriteLine("  Created IsInnerCity feature (distance <= 10km)");
        }
    }
    
    // Helper methods
    private DataFrame FilterByIndices(DataFrame df, List<long> indices)
    {
        var indexSet = new HashSet<long>(indices);
        var filter = new PrimitiveDataFrameColumn<bool>("filter", df.Rows.Count);
        for (long i = 0; i < df.Rows.Count; i++)
            filter[i] = indexSet.Contains(i);
        return df.Filter(filter);
    }
    
    private DataFrame FilterWhereNotNull(DataFrame df, string columnName)
    {
        var filter = new PrimitiveDataFrameColumn<bool>("filter", df.Rows.Count);
        var col = df.Columns[columnName];
        for (long i = 0; i < df.Rows.Count; i++)
            filter[i] = col[i] != null;
        return df.Filter(filter);
    }
    
    private double CalculateMedian<T>(PrimitiveDataFrameColumn<T> col) where T : unmanaged
    {
        var values = new List<double>();
        for (long i = 0; i < col.Length; i++)
        {
            if (col[i] != null)
                values.Add(Convert.ToDouble(col[i]));
        }
        values.Sort();
        int mid = values.Count / 2;
        return values.Count % 2 == 0 
            ? (values[mid - 1] + values[mid]) / 2 
            : values[mid];
    }
    
    private string CalculateMode(StringDataFrameColumn col)
    {
        var counts = new Dictionary<string, int>();
        for (long i = 0; i < col.Length; i++)
        {
            if (col[i] != null)
                counts[col[i]!] = counts.GetValueOrDefault(col[i]!, 0) + 1;
        }
        return counts.OrderByDescending(kv => kv.Value).First().Key;
    }
    
    private void FillStringNulls(StringDataFrameColumn col, string value)
    {
        for (long i = 0; i < col.Length; i++)
        {
            if (col[i] == null)
                col[i] = value;
        }
    }
    
    private List<double> GetNonNullValues<T>(PrimitiveDataFrameColumn<T> col) where T : unmanaged
    {
        var values = new List<double>();
        for (long i = 0; i < col.Length; i++)
        {
            if (col[i] != null)
                values.Add(Convert.ToDouble(col[i]));
        }
        return values;
    }
    
    private (double lower, double upper) CalculateIQRBounds(List<double> values, double multiplier)
    {
        values.Sort();
        double q1 = Percentile(values, 25);
        double q3 = Percentile(values, 75);
        double iqr = q3 - q1;
        return (q1 - multiplier * iqr, q3 + multiplier * iqr);
    }
    
    private double Percentile(List<double> sorted, double p)
    {
        double index = (p / 100.0) * (sorted.Count - 1);
        int lower = (int)Math.Floor(index);
        int upper = Math.Min((int)Math.Ceiling(index), sorted.Count - 1);
        return sorted[lower] + (index - lower) * (sorted[upper] - sorted[lower]);
    }
    
    private void SaveDataFrame(DataFrame df, string path)
    {
        using var writer = new StreamWriter(path);
        
        // Write header
        writer.WriteLine(string.Join(",", df.Columns.Select(c => c.Name)));
        
        // Write data
        for (long i = 0; i < df.Rows.Count; i++)
        {
            var values = df.Columns.Select(c => 
            {
                var val = c[i];
                if (val == null) return "";
                var str = val.ToString() ?? "";
                return str.Contains(',') ? $"\"{str}\"" : str;
            });
            writer.WriteLine(string.Join(",", values));
        }
    }
}

// Usage
public class Program
{
    public static async Task Main(string[] args)
    {
        var cleaner = new HousingDataCleaner();
        var cleanedData = await cleaner.CleanDataAsync(
            "melbourne_housing_raw.csv",
            "melbourne_housing_clean.csv");
    }
}
```

### Building the ML.NET Pipeline Version

For production ML scenarios, wrap the cleaning logic in an ML.NET pipeline:

```csharp
public IEstimator<ITransformer> BuildMLNetCleaningPipeline(MLContext mlContext)
{
    return mlContext.Transforms
        // Replace missing numeric values
        .ReplaceMissingValues(new[]
        {
            new InputOutputColumnPair("Rooms"),
            new InputOutputColumnPair("Bathroom"),
            new InputOutputColumnPair("Car"),
            new InputOutputColumnPair("Landsize"),
            new InputOutputColumnPair("BuildingArea")
        }, MissingValueReplacingEstimator.ReplacementMode.Mean)
        
        // Normalize numeric features
        .Append(mlContext.Transforms.NormalizeMinMax("Rooms"))
        .Append(mlContext.Transforms.NormalizeMinMax("Bathroom"))
        .Append(mlContext.Transforms.NormalizeMinMax("Distance"))
        .Append(mlContext.Transforms.NormalizeMinMax("Landsize"))
        
        // Encode categoricals
        .Append(mlContext.Transforms.Categorical.OneHotEncoding("Type"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding("Method"))
        .Append(mlContext.Transforms.Categorical.OneHotHashEncoding(
            "Suburb", numberOfBits: 10))  // Hash for high cardinality
        
        // Combine into feature vector
        .Append(mlContext.Transforms.Concatenate("Features",
            "Rooms", "Bathroom", "Car", "Distance", "Landsize",
            "Type", "Method", "Suburb"));
}
```

## Key Takeaways

Data wrangling in C# might feel more verbose than the equivalent Python code, but the trade-off is worth it:

1. **Type safety catches errors early.** Your IDE warns you about null references and type mismatches before runtime.

2. **LINQ provides a powerful, composable transformation API.** The skills you've built with LINQ translate directly to data manipulation.

3. **`Microsoft.Data.Analysis` gives you DataFrame semantics** when you need interactive exploration, while **ML.NET's `IDataView`** handles production-scale data pipelines.

4. **The C# ecosystem has mature solutions** for every data loading scenario—CSV, JSON, databases, and APIs all have battle-tested libraries.

5. **Code organization matters.** The extension methods and utility classes we built in this chapter are reusable across projects. Build your own data cleaning toolkit.

In the next chapter, we'll build on this foundation to explore and visualize data—turning those clean DataFrames into insights and compelling visualizations.
