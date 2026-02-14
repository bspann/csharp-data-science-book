namespace DataScience.Core;

/// <summary>
/// Statistical extension methods for data science operations.
/// </summary>
public static class StatisticalExtensions
{
    /// <summary>
    /// Calculates the standard deviation of a sequence.
    /// </summary>
    public static double StdDev(this IEnumerable<double> values)
    {
        var list = values.ToList();
        if (list.Count == 0) return 0;
        
        var mean = list.Average();
        var sumSquaredDiffs = list.Sum(v => Math.Pow(v - mean, 2));
        return Math.Sqrt(sumSquaredDiffs / list.Count);
    }

    /// <summary>
    /// Calculates the median of a sequence.
    /// </summary>
    public static double Median(this IEnumerable<double> values)
    {
        var sorted = values.OrderBy(v => v).ToList();
        if (sorted.Count == 0) return 0;
        
        int mid = sorted.Count / 2;
        return sorted.Count % 2 == 0
            ? (sorted[mid - 1] + sorted[mid]) / 2
            : sorted[mid];
    }

    /// <summary>
    /// Calculates Z-scores for a sequence.
    /// </summary>
    public static IEnumerable<double> ZScore(this IEnumerable<double> values)
    {
        var list = values.ToList();
        var mean = list.Average();
        var std = list.StdDev();
        
        return std == 0 
            ? list.Select(_ => 0.0) 
            : list.Select(v => (v - mean) / std);
    }

    /// <summary>
    /// Normalizes values to [0, 1] range.
    /// </summary>
    public static IEnumerable<double> MinMaxNormalize(this IEnumerable<double> values)
    {
        var list = values.ToList();
        var min = list.Min();
        var max = list.Max();
        var range = max - min;
        
        return range == 0 
            ? list.Select(_ => 0.0) 
            : list.Select(v => (v - min) / range);
    }

    /// <summary>
    /// Batches a sequence into chunks of specified size.
    /// </summary>
    public static IEnumerable<IEnumerable<T>> Batch<T>(this IEnumerable<T> source, int size)
    {
        var batch = new List<T>(size);
        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count == size)
            {
                yield return batch;
                batch = new List<T>(size);
            }
        }
        if (batch.Count > 0)
            yield return batch;
    }

    /// <summary>
    /// Calculates Pearson correlation coefficient between two sequences.
    /// </summary>
    public static double Correlation(this IEnumerable<double> x, IEnumerable<double> y)
    {
        var xList = x.ToList();
        var yList = y.ToList();
        
        if (xList.Count != yList.Count || xList.Count == 0)
            throw new ArgumentException("Sequences must have same non-zero length");

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
}
