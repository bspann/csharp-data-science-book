# Appendix A: C# for Data Science Quick Reference

This appendix serves as a quick reference for the most common C# patterns used throughout this book. Keep it bookmarked—you'll return to these snippets often.

---

## LINQ Patterns for Data Manipulation

LINQ is the backbone of data manipulation in C#. Master these patterns and you'll handle 90% of your data transformation needs.

### Essential LINQ Methods

| Method | Purpose | Example |
|--------|---------|---------|
| `Where` | Filter elements | `.Where(x => x.Value > 0)` |
| `Select` | Transform elements | `.Select(x => x.Name)` |
| `SelectMany` | Flatten nested collections | `.SelectMany(x => x.Items)` |
| `OrderBy/ThenBy` | Sort elements | `.OrderBy(x => x.Date).ThenBy(x => x.Name)` |
| `GroupBy` | Group by key | `.GroupBy(x => x.Category)` |
| `Aggregate` | Reduce to single value | `.Aggregate((a, b) => a + b)` |
| `Zip` | Combine two sequences | `.Zip(other, (a, b) => a + b)` |
| `Take/Skip` | Pagination | `.Skip(10).Take(5)` |
| `Distinct` | Remove duplicates | `.Distinct()` |
| `ToLookup` | Create indexed lookup | `.ToLookup(x => x.Key)` |

### Grouping Patterns

**Basic grouping:**

```csharp
var salesByRegion = sales
    .GroupBy(s => s.Region)
    .Select(g => new
    {
        Region = g.Key,
        TotalSales = g.Sum(s => s.Amount),
        Count = g.Count()
    });
```

**Multi-key grouping:**

```csharp
var salesByRegionAndYear = sales
    .GroupBy(s => new { s.Region, s.Date.Year })
    .Select(g => new
    {
        g.Key.Region,
        g.Key.Year,
        Total = g.Sum(s => s.Amount)
    });
```

**Grouping with filtering:**

```csharp
// Only groups with more than 100 items
var significantGroups = data
    .GroupBy(x => x.Category)
    .Where(g => g.Count() > 100)
    .Select(g => new { Category = g.Key, Items = g.ToList() });
```

**Nested grouping (hierarchical):**

```csharp
var hierarchy = sales
    .GroupBy(s => s.Region)
    .Select(regionGroup => new
    {
        Region = regionGroup.Key,
        Cities = regionGroup
            .GroupBy(s => s.City)
            .Select(cityGroup => new
            {
                City = cityGroup.Key,
                Total = cityGroup.Sum(s => s.Amount)
            })
    });
```

### Aggregation Patterns

**Basic aggregations:**

```csharp
var stats = new
{
    Sum = values.Sum(),
    Average = values.Average(),
    Min = values.Min(),
    Max = values.Max(),
    Count = values.Count()
};
```

**Custom aggregation with Aggregate:**

```csharp
// Running product
var product = numbers.Aggregate(1.0, (acc, x) => acc * x);

// Custom accumulator (mean and variance in one pass)
var result = values.Aggregate(
    new { Sum = 0.0, SumSq = 0.0, Count = 0 },
    (acc, x) => new
    {
        Sum = acc.Sum + x,
        SumSq = acc.SumSq + x * x,
        Count = acc.Count + 1
    },
    acc => new
    {
        Mean = acc.Sum / acc.Count,
        Variance = (acc.SumSq / acc.Count) - Math.Pow(acc.Sum / acc.Count, 2)
    });
```

**Conditional aggregation:**

```csharp
var conditionalSum = data
    .Where(x => x.IsValid)
    .Sum(x => x.Value);

// Or inline with null coalescing
var safeSum = data.Sum(x => x.IsValid ? x.Value : 0);
```

### Windowing and Sliding Operations

**Chunking (batching):**

```csharp
// .NET 6+
var batches = data.Chunk(100);

// Pre-.NET 6 equivalent
IEnumerable<T[]> ChunkLegacy<T>(IEnumerable<T> source, int size)
{
    var batch = new List<T>(size);
    foreach (var item in source)
    {
        batch.Add(item);
        if (batch.Count == size)
        {
            yield return batch.ToArray();
            batch.Clear();
        }
    }
    if (batch.Count > 0)
        yield return batch.ToArray();
}
```

**Sliding window:**

```csharp
// Moving average with window size n
IEnumerable<double> MovingAverage(IEnumerable<double> source, int window)
{
    var buffer = new Queue<double>(window);
    foreach (var value in source)
    {
        buffer.Enqueue(value);
        if (buffer.Count == window)
        {
            yield return buffer.Average();
            buffer.Dequeue();
        }
    }
}

// Usage
var smoothed = MovingAverage(prices, 20).ToList();
```

**Pairwise operations:**

```csharp
// Calculate differences between consecutive elements
var differences = data
    .Zip(data.Skip(1), (prev, curr) => curr - prev);

// Generic pairwise
IEnumerable<TResult> Pairwise<T, TResult>(
    IEnumerable<T> source, 
    Func<T, T, TResult> selector)
{
    using var e = source.GetEnumerator();
    if (!e.MoveNext()) yield break;
    var prev = e.Current;
    while (e.MoveNext())
    {
        yield return selector(prev, e.Current);
        prev = e.Current;
    }
}
```

**Lag and lead operations:**

```csharp
// Access previous values (lag)
var withLag = data
    .Select((value, index) => new
    {
        Current = value,
        Previous = index > 0 ? data[index - 1] : default,
        Lag2 = index > 1 ? data[index - 2] : default
    });

// Percent change
var percentChanges = prices
    .Zip(prices.Skip(1), (prev, curr) => (curr - prev) / prev * 100);
```

---

## Parallel Processing with PLINQ

PLINQ parallelizes LINQ queries automatically. Use it when you have CPU-bound operations on large datasets.

### Basic PLINQ Usage

| Pattern | Code |
|---------|------|
| Enable parallelism | `.AsParallel()` |
| Force ordered results | `.AsOrdered()` |
| Set degree of parallelism | `.WithDegreeOfParallelism(4)` |
| Return to sequential | `.AsSequential()` |
| Force execution mode | `.WithExecutionMode(ParallelExecutionMode.ForceParallelism)` |

**Basic parallel query:**

```csharp
var results = data
    .AsParallel()
    .Where(x => ExpensiveFilter(x))
    .Select(x => ExpensiveTransform(x))
    .ToList();
```

**Preserving order (slower but deterministic):**

```csharp
var orderedResults = data
    .AsParallel()
    .AsOrdered()
    .Select(x => ExpensiveTransform(x))
    .ToList();
```

**Controlling parallelism:**

```csharp
// Limit to physical cores for CPU-bound work
var results = data
    .AsParallel()
    .WithDegreeOfParallelism(Environment.ProcessorCount)
    .Select(x => CpuIntensiveWork(x))
    .ToList();
```

### When to Use PLINQ

| ✅ Use PLINQ When | ❌ Avoid PLINQ When |
|------------------|-------------------|
| CPU-bound transformations | I/O-bound operations |
| Large datasets (10,000+ items) | Small datasets |
| Independent operations | Operations with side effects |
| Expensive per-item processing | Simple operations (overhead > benefit) |
| Order doesn't matter | Strict ordering required |

### PLINQ Performance Patterns

**Partition for locality:**

```csharp
// Custom partitioner for better cache behavior
var partitioner = Partitioner.Create(data, loadBalance: true);
var results = partitioner
    .AsParallel()
    .Select(x => Process(x))
    .ToList();
```

**Aggregate in parallel:**

```csharp
// Thread-safe parallel aggregation
var sum = data
    .AsParallel()
    .Aggregate(
        0.0,                           // Seed
        (subtotal, item) => subtotal + item.Value,  // Accumulate
        (total, subtotal) => total + subtotal,      // Combine partitions
        total => total                              // Final result
    );
```

**ForAll for side effects:**

```csharp
// When you don't need results collected
data
    .AsParallel()
    .Where(x => x.NeedsProcessing)
    .ForAll(x => ProcessInPlace(x));
```

---

## Memory-Efficient Data Handling

When working with large datasets, memory efficiency determines whether your code runs or crashes.

### Span<T> for Zero-Allocation Slicing

`Span<T>` provides a view into contiguous memory without allocating new arrays.

**Basic Span usage:**

```csharp
double[] data = LoadLargeArray();

// Zero-allocation slice
Span<double> subset = data.AsSpan(1000, 500);

// Modify in place
foreach (ref var value in subset)
{
    value = Math.Log(value);
}
```

**Span for string parsing (no allocations):**

```csharp
// Parse CSV line without string allocations
void ParseCsvLine(ReadOnlySpan<char> line, Span<double> values)
{
    int fieldIndex = 0;
    int start = 0;
    
    for (int i = 0; i <= line.Length; i++)
    {
        if (i == line.Length || line[i] == ',')
        {
            values[fieldIndex++] = double.Parse(line[start..i]);
            start = i + 1;
        }
    }
}
```

**Span limitations to remember:**

| ✅ Can Do | ❌ Cannot Do |
|----------|-------------|
| Use in synchronous methods | Store as class field |
| Pass as method parameter | Use in async methods |
| Stack allocate with `stackalloc` | Box or use in closures |
| Slice without allocation | Return from iterator |

### ArrayPool for Reusable Buffers

Rent arrays instead of allocating to reduce GC pressure.

```csharp
// Rent a buffer
double[] buffer = ArrayPool<double>.Shared.Rent(minimumLength: 1000);

try
{
    // Use the buffer (may be larger than requested!)
    int actualLength = Math.Min(buffer.Length, dataSize);
    ProcessData(buffer.AsSpan(0, actualLength));
}
finally
{
    // Always return! Optionally clear sensitive data
    ArrayPool<double>.Shared.Return(buffer, clearArray: false);
}
```

**Helper pattern for safe usage:**

```csharp
public readonly struct RentedArray<T> : IDisposable
{
    private readonly T[] _array;
    public readonly int Length;
    
    public RentedArray(int length)
    {
        _array = ArrayPool<T>.Shared.Rent(length);
        Length = length;
    }
    
    public Span<T> Span => _array.AsSpan(0, Length);
    public T this[int i] => _array[i];
    
    public void Dispose() => ArrayPool<T>.Shared.Return(_array);
}

// Usage
using var buffer = new RentedArray<double>(10000);
ProcessData(buffer.Span);
```

### Streaming Large Files

Never load entire files when you can stream.

**Stream CSV with IEnumerable:**

```csharp
IEnumerable<DataPoint> StreamCsv(string path)
{
    using var reader = new StreamReader(path);
    reader.ReadLine(); // Skip header
    
    while (reader.ReadLine() is { } line)
    {
        var parts = line.Split(',');
        yield return new DataPoint
        {
            Date = DateTime.Parse(parts[0]),
            Value = double.Parse(parts[1])
        };
    }
}

// Process without loading all into memory
var average = StreamCsv("huge_file.csv")
    .Where(p => p.Value > 0)
    .Average(p => p.Value);
```

**Batch streaming for efficiency:**

```csharp
IEnumerable<DataPoint[]> StreamCsvBatched(string path, int batchSize = 1000)
{
    using var reader = new StreamReader(path);
    reader.ReadLine(); // Skip header
    
    var batch = new List<DataPoint>(batchSize);
    
    while (reader.ReadLine() is { } line)
    {
        batch.Add(ParseLine(line));
        
        if (batch.Count >= batchSize)
        {
            yield return batch.ToArray();
            batch.Clear();
        }
    }
    
    if (batch.Count > 0)
        yield return batch.ToArray();
}
```

**Memory-mapped files for random access:**

```csharp
using var mmf = MemoryMappedFile.CreateFromFile("huge_data.bin");
using var accessor = mmf.CreateViewAccessor();

// Read doubles directly from disk
double ReadValue(long index)
{
    accessor.Read(index * sizeof(double), out double value);
    return value;
}
```

### Memory Comparison Table

| Approach | Allocation | Best For |
|----------|------------|----------|
| `new T[]` | Full array | Small, short-lived arrays |
| `ArrayPool<T>` | None (reused) | Repeated operations, known sizes |
| `Span<T>` | None (view) | Slicing existing memory |
| `stackalloc` | Stack only | Small buffers (<1KB), hot paths |
| Streaming | Per-item | Files larger than RAM |
| `MemoryMappedFile` | Page-based | Random access to huge files |

---

## Common Gotchas and Solutions

### Deferred Execution Surprises

**Problem:** LINQ queries execute lazily—each enumeration re-runs the query.

```csharp
// BUG: Enumerates twice, may give different results!
var filtered = data.Where(x => x.Value > threshold);
Console.WriteLine(filtered.Count());  // First enumeration
Console.WriteLine(filtered.Sum(x => x.Value));  // Second enumeration
```

**Solution:** Materialize when you need to enumerate multiple times.

```csharp
var filtered = data.Where(x => x.Value > threshold).ToList();
Console.WriteLine(filtered.Count);
Console.WriteLine(filtered.Sum(x => x.Value));
```

### Closure Variable Capture

**Problem:** Lambdas capture variables, not values.

```csharp
// BUG: All tasks use final value of i
var tasks = new List<Task>();
for (int i = 0; i < 10; i++)
{
    tasks.Add(Task.Run(() => Process(i)));  // i is captured by reference!
}
```

**Solution:** Create local copy.

```csharp
for (int i = 0; i < 10; i++)
{
    int localI = i;  // Captured correctly
    tasks.Add(Task.Run(() => Process(localI)));
}

// Or use Parallel.For which handles this
Parallel.For(0, 10, i => Process(i));
```

### PLINQ Ordering Loss

**Problem:** PLINQ doesn't preserve order by default.

```csharp
// Results may be in any order!
var results = data.AsParallel().Select(Transform).ToList();
```

**Solution:** Use `AsOrdered()` when order matters.

```csharp
var results = data.AsParallel().AsOrdered().Select(Transform).ToList();
```

### Span in Async Context

**Problem:** Span cannot cross await boundaries.

```csharp
// COMPILE ERROR: Cannot use Span in async method
async Task ProcessAsync(Span<byte> data) { ... }
```

**Solution:** Use `Memory<T>` for async scenarios.

```csharp
async Task ProcessAsync(Memory<byte> data)
{
    await SomeAsyncOperation();
    // Now work with data.Span
    ProcessSpan(data.Span);
}
```

### GroupBy with Reference Types

**Problem:** GroupBy uses default equality, which may not be what you want.

```csharp
// Groups by reference, not value!
var grouped = data.GroupBy(x => new { x.Year, x.Month });
```

**Solution:** Anonymous types work correctly. For custom types, implement `IEquatable<T>`.

```csharp
public record DateKey(int Year, int Month);  // Records have value equality

var grouped = data.GroupBy(x => new DateKey(x.Year, x.Month));
```

### Premature Optimization Table

| Instinct | Reality | Do Instead |
|----------|---------|------------|
| "Use PLINQ everywhere" | Overhead often exceeds gain | Profile first, parallelize hotspots |
| "Avoid LINQ, use loops" | LINQ is often faster (optimized) | Write clear code, profile later |
| "Pre-allocate everything" | Adds complexity, marginal gain | Use ArrayPool for hot paths only |
| "stackalloc all buffers" | Stack overflow risk | Only for small, known-size buffers |

---

## Quick Performance Tips

1. **Prefer `Count` property over `Count()` method** on collections that have it
2. **Use `Any()` instead of `Count() > 0`** for existence checks
3. **Chain `Where` before `Select`** to reduce transformed items
4. **Use `ToList()` capacity overload** when size is known: `new List<T>(knownCount)`
5. **Avoid repeated `OrderBy`** in hot paths—sort once, cache result
6. **Use `Dictionary` or `HashSet`** for O(1) lookups instead of repeated `Where`
7. **Buffer file I/O** with `BufferedStream` or larger buffer sizes
8. **Use `StringBuilder`** for string concatenation in loops
9. **Consider `ReadOnlySpan<char>`** for string parsing to avoid allocations
10. **Profile before optimizing**—your intuition is often wrong

---

## Cheat Sheet: Common Operations

```csharp
// Top N items
var top10 = data.OrderByDescending(x => x.Score).Take(10);

// Distinct by property
var distinctByName = data.DistinctBy(x => x.Name);  // .NET 6+

// First/Last with default
var first = data.FirstOrDefault(x => x.IsValid);
var last = data.LastOrDefault();

// Index of item
int index = Array.IndexOf(array, target);
int index = list.FindIndex(x => x.Name == "target");

// Combine/Merge sequences
var combined = seq1.Concat(seq2);
var zipped = seq1.Zip(seq2, (a, b) => a + b);

// Running total
var runningTotal = data.Scan(0.0, (acc, x) => acc + x);  // MoreLINQ

// Percentile (approximate)
var sorted = values.OrderBy(x => x).ToList();
var p95 = sorted[(int)(sorted.Count * 0.95)];

// Null-safe navigation
var result = data?.Where(x => x != null).Select(x => x!.Value);
```

---

## Async Streams for Data Pipelines

When processing data from async sources (databases, APIs, message queues), use `IAsyncEnumerable<T>`.

**Consuming async streams:**

```csharp
await foreach (var record in QueryDatabaseAsync(connection))
{
    ProcessRecord(record);
}

// With cancellation
await foreach (var record in stream.WithCancellation(cancellationToken))
{
    if (ShouldStop(record)) break;
    ProcessRecord(record);
}
```

**Creating async streams:**

```csharp
async IAsyncEnumerable<DataPoint> FetchDataAsync(
    HttpClient client,
    [EnumeratorCancellation] CancellationToken ct = default)
{
    int page = 0;
    while (true)
    {
        var batch = await client.GetFromJsonAsync<DataPoint[]>(
            $"/api/data?page={page}", ct);
        
        if (batch is null || batch.Length == 0)
            yield break;
        
        foreach (var item in batch)
            yield return item;
        
        page++;
    }
}
```

**LINQ with async streams (.NET 6+ / System.Linq.Async):**

```csharp
var filtered = await source
    .WhereAwait(async x => await ValidateAsync(x))
    .Take(100)
    .ToListAsync();

var sum = await source
    .Where(x => x.Value > 0)
    .SumAsync(x => x.Value);
```

---

## Immutability Patterns for Data Integrity

Immutable data structures prevent accidental mutations and enable safe parallel processing.

**Records for data models:**

```csharp
public record DataPoint(DateTime Timestamp, double Value, string Label)
{
    // Derived property
    public int Year => Timestamp.Year;
}

// Non-destructive mutation with 'with'
var adjusted = original with { Value = original.Value * 1.1 };
```

**Immutable collections:**

```csharp
using System.Collections.Immutable;

// Create immutable list
ImmutableList<double> values = ImmutableList.Create(1.0, 2.0, 3.0);

// "Add" returns new list (original unchanged)
ImmutableList<double> extended = values.Add(4.0);

// Builder for efficient bulk operations
var builder = ImmutableList.CreateBuilder<double>();
foreach (var item in sourceData)
    builder.Add(Transform(item));
ImmutableList<double> result = builder.ToImmutable();
```

| Collection | Immutable Equivalent | Notes |
|------------|---------------------|-------|
| `List<T>` | `ImmutableList<T>` | O(log n) add/remove |
| `Dictionary<K,V>` | `ImmutableDictionary<K,V>` | O(log n) operations |
| `HashSet<T>` | `ImmutableHashSet<T>` | O(log n) operations |
| `Array` | `ImmutableArray<T>` | Fastest immutable, no add |

---

## Benchmarking Quick Reference

Always measure before optimizing. Use BenchmarkDotNet for reliable results.

**Basic benchmark setup:**

```csharp
[MemoryDiagnoser]
[SimpleJob(RuntimeMoniker.Net80)]
public class DataProcessingBenchmarks
{
    private double[] _data = null!;
    
    [GlobalSetup]
    public void Setup()
    {
        _data = Enumerable.Range(0, 100_000)
            .Select(_ => Random.Shared.NextDouble())
            .ToArray();
    }
    
    [Benchmark(Baseline = true)]
    public double LinqSum() => _data.Sum();
    
    [Benchmark]
    public double ForLoopSum()
    {
        double sum = 0;
        for (int i = 0; i < _data.Length; i++)
            sum += _data[i];
        return sum;
    }
    
    [Benchmark]
    public double SpanSum()
    {
        double sum = 0;
        foreach (var value in _data.AsSpan())
            sum += value;
        return sum;
    }
}
```

**Running benchmarks:**

```bash
dotnet run -c Release -- --filter "*Sum*"
```

**Key metrics to watch:**

| Metric | Meaning | Target |
|--------|---------|--------|
| Mean | Average execution time | Lower is better |
| Allocated | Heap memory per operation | Zero for hot paths |
| Gen 0/1/2 | GC collections triggered | Minimize all |
| StdDev | Consistency | Lower means reliable results |

---

## Extension Methods for Data Science

Build a personal toolkit of extension methods for common operations.

```csharp
public static class DataScienceExtensions
{
    // Standard deviation
    public static double StdDev(this IEnumerable<double> values)
    {
        var list = values.ToList();
        if (list.Count < 2) return 0;
        
        double mean = list.Average();
        double sumSq = list.Sum(v => Math.Pow(v - mean, 2));
        return Math.Sqrt(sumSq / (list.Count - 1));
    }
    
    // Median
    public static double Median(this IEnumerable<double> values)
    {
        var sorted = values.OrderBy(x => x).ToList();
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) / 2.0
            : sorted[mid];
    }
    
    // Z-score normalization
    public static IEnumerable<double> ZScore(this IEnumerable<double> values)
    {
        var list = values.ToList();
        double mean = list.Average();
        double std = list.StdDev();
        return std == 0 
            ? list.Select(_ => 0.0) 
            : list.Select(v => (v - mean) / std);
    }
    
    // Min-Max normalization
    public static IEnumerable<double> MinMaxNormalize(this IEnumerable<double> values)
    {
        var list = values.ToList();
        double min = list.Min();
        double max = list.Max();
        double range = max - min;
        return range == 0 
            ? list.Select(_ => 0.0) 
            : list.Select(v => (v - min) / range);
    }
    
    // Batch/Chunk for pre-.NET 6
    public static IEnumerable<IReadOnlyList<T>> Batch<T>(
        this IEnumerable<T> source, int size)
    {
        var batch = new List<T>(size);
        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count >= size)
            {
                yield return batch;
                batch = new List<T>(size);
            }
        }
        if (batch.Count > 0)
            yield return batch;
    }
    
    // Safe division
    public static double SafeDiv(this double numerator, double denominator, 
        double fallback = 0) =>
        denominator == 0 ? fallback : numerator / denominator;
}
```

---

## Debugging Data Pipelines

When LINQ chains get complex, debug them systematically.

**Inspect intermediate results:**

```csharp
var result = data
    .Where(x => x.Value > 0)
    .Tap(seq => Console.WriteLine($"After filter: {seq.Count()}"))
    .Select(x => Transform(x))
    .Tap(seq => Console.WriteLine($"After transform: {seq.Count()}"))
    .ToList();

// Tap extension
public static IEnumerable<T> Tap<T>(
    this IEnumerable<T> source, 
    Action<IEnumerable<T>> inspector)
{
    var list = source.ToList();
    inspector(list);
    return list;
}
```

**Logging extension:**

```csharp
public static IEnumerable<T> Log<T>(
    this IEnumerable<T> source, 
    string label,
    Func<T, string>? formatter = null)
{
    foreach (var item in source)
    {
        var message = formatter?.Invoke(item) ?? item?.ToString() ?? "null";
        Console.WriteLine($"[{label}] {message}");
        yield return item;
    }
}

// Usage
var results = data
    .Where(x => x.Value > threshold)
    .Log("filtered")
    .Select(Transform)
    .Log("transformed", x => $"{x.Id}: {x.Value:F2}")
    .ToList();
```

---

*This appendix covers the most frequently used patterns. For deeper exploration of any topic, refer to the relevant chapter in the main text.*
