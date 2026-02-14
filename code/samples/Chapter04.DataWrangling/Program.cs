// Chapter 4: Data Wrangling with C#
// Demonstrates cleaning, transforming, and preparing data for analysis

using Microsoft.Data.Analysis;

Console.WriteLine("=== Chapter 4: Data Wrangling ===\n");

// 1. LOAD EMBEDDED SAMPLE DATA (simulating CSV import)
Console.WriteLine("1. Loading raw data...\n");

var names = new StringDataFrameColumn("Name", new[] {
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", 
    "Grace", "Henry", "Alice", "Ivan", "Julia", "Kevin"  // Note: "Alice" is duplicate
});
var ages = new PrimitiveDataFrameColumn<float>("Age", new float?[] {
    25, 35, null, 45, 28, 150, 32, null, 25, 42, 31, 29  // null = missing, 150 = outlier
});
var salaries = new PrimitiveDataFrameColumn<float>("Salary", new float?[] {
    50000, 75000, 65000, null, 58000, 72000, 68000, 55000, 50000, 95000, 62000, -5000  // null, negative = outlier
});
var departments = new StringDataFrameColumn("Department", new[] {
    "Engineering", "Sales", "Engineering", "HR", "Marketing", "Sales",
    "Engineering", "HR", "Engineering", "Sales", "Marketing", "Engineering"
});
var yearsExp = new PrimitiveDataFrameColumn<float>("YearsExperience", new float?[] {
    2, 8, 5, 20, 3, 12, 6, null, 2, 15, 4, 3
});

var df = new DataFrame(names, ages, salaries, departments, yearsExp);

Console.WriteLine("Raw data:");
Console.WriteLine(df);
PrintStats(df, "Initial Statistics");

// 2. HANDLE MISSING VALUES
Console.WriteLine("\n2. Handling missing values...\n");

// Strategy: Fill numeric nulls with column median
var ageCol = df.Columns["Age"] as PrimitiveDataFrameColumn<float>;
var salaryCol = df.Columns["Salary"] as PrimitiveDataFrameColumn<float>;
var expCol = df.Columns["YearsExperience"] as PrimitiveDataFrameColumn<float>;

float ageMedian = CalculateMedian(ageCol!);
float salaryMedian = CalculateMedian(salaryCol!);
float expMedian = CalculateMedian(expCol!);

Console.WriteLine($"Filling Age nulls with median: {ageMedian}");
Console.WriteLine($"Filling Salary nulls with median: {salaryMedian}");
Console.WriteLine($"Filling YearsExperience nulls with median: {expMedian}");

FillNulls(ageCol!, ageMedian);
FillNulls(salaryCol!, salaryMedian);
FillNulls(expCol!, expMedian);

Console.WriteLine("\nAfter filling missing values:");
Console.WriteLine(df);

// 3. REMOVE DUPLICATES
Console.WriteLine("\n3. Removing duplicates...\n");

var seen = new HashSet<string>();
var keepRows = new List<long>();

for (long i = 0; i < df.Rows.Count; i++)
{
    // Create composite key from Name + Department
    string key = $"{df["Name"][i]}|{df["Department"][i]}";
    if (seen.Add(key))
        keepRows.Add(i);
    else
        Console.WriteLine($"  Removing duplicate: {df["Name"][i]} in {df["Department"][i]}");
}

df = df.Filter(CreateMask(df.Rows.Count, keepRows));
Console.WriteLine($"\nRows after deduplication: {df.Rows.Count}");

// 4. HANDLE OUTLIERS (IQR Method)
Console.WriteLine("\n4. Handling outliers using IQR method...\n");

// Recalculate column references after filter
ageCol = df.Columns["Age"] as PrimitiveDataFrameColumn<float>;
salaryCol = df.Columns["Salary"] as PrimitiveDataFrameColumn<float>;

var (ageQ1, ageQ3) = CalculateQuartiles(ageCol!);
var (salaryQ1, salaryQ3) = CalculateQuartiles(salaryCol!);

float ageIqr = ageQ3 - ageQ1;
float salaryIqr = salaryQ3 - salaryQ1;

float ageLower = ageQ1 - 1.5f * ageIqr;
float ageUpper = ageQ3 + 1.5f * ageIqr;
float salaryLower = salaryQ1 - 1.5f * salaryIqr;
float salaryUpper = salaryQ3 + 1.5f * salaryIqr;

Console.WriteLine($"Age bounds: [{ageLower:F1}, {ageUpper:F1}]");
Console.WriteLine($"Salary bounds: [{salaryLower:F0}, {salaryUpper:F0}]");

// Cap outliers instead of removing
for (long i = 0; i < df.Rows.Count; i++)
{
    float age = (float)ageCol![i]!;
    float salary = (float)salaryCol![i]!;
    
    if (age < ageLower || age > ageUpper)
    {
        float newAge = Math.Clamp(age, ageLower, ageUpper);
        Console.WriteLine($"  Capping Age: {age} -> {newAge:F1} for {df["Name"][i]}");
        ageCol[i] = newAge;
    }
    if (salary < salaryLower || salary > salaryUpper)
    {
        float newSalary = Math.Clamp(salary, salaryLower, salaryUpper);
        Console.WriteLine($"  Capping Salary: {salary} -> {newSalary:F0} for {df["Name"][i]}");
        salaryCol[i] = newSalary;
    }
}

// 5. CREATE DERIVED FEATURES
Console.WriteLine("\n5. Creating derived features...\n");

// Feature 1: Salary per year of experience
var salaryPerExp = new PrimitiveDataFrameColumn<float>("SalaryPerYearExp", df.Rows.Count);
expCol = df.Columns["YearsExperience"] as PrimitiveDataFrameColumn<float>;

for (long i = 0; i < df.Rows.Count; i++)
{
    float salary = (float)salaryCol![i]!;
    float exp = (float)expCol![i]!;
    salaryPerExp[i] = exp > 0 ? salary / exp : salary;
}
df.Columns.Add(salaryPerExp);

// Feature 2: Experience level category
var expLevel = new StringDataFrameColumn("ExperienceLevel", df.Rows.Count);
for (long i = 0; i < df.Rows.Count; i++)
{
    float exp = (float)expCol![i]!;
    expLevel[i] = exp switch
    {
        < 3 => "Junior",
        < 7 => "Mid",
        < 12 => "Senior",
        _ => "Lead"
    };
}
df.Columns.Add(expLevel);

// Feature 3: Age group
var ageGroup = new StringDataFrameColumn("AgeGroup", df.Rows.Count);
for (long i = 0; i < df.Rows.Count; i++)
{
    float age = (float)ageCol![i]!;
    ageGroup[i] = age switch
    {
        < 30 => "20s",
        < 40 => "30s",
        < 50 => "40s",
        _ => "50+"
    };
}
df.Columns.Add(ageGroup);

Console.WriteLine("Added: SalaryPerYearExp, ExperienceLevel, AgeGroup");

// 6. FINAL RESULTS
Console.WriteLine("\n6. Final cleaned dataset:\n");
Console.WriteLine(df);
PrintStats(df, "Final Statistics");

Console.WriteLine("\n=== Data Wrangling Complete! ===");

// ============ HELPER METHODS ============

static float CalculateMedian(PrimitiveDataFrameColumn<float> col)
{
    var values = col.Where(v => v.HasValue).Select(v => v!.Value).OrderBy(v => v).ToList();
    if (values.Count == 0) return 0;
    int mid = values.Count / 2;
    return values.Count % 2 == 0 ? (values[mid - 1] + values[mid]) / 2 : values[mid];
}

static void FillNulls(PrimitiveDataFrameColumn<float> col, float value)
{
    for (long i = 0; i < col.Length; i++)
        if (!col[i].HasValue) col[i] = value;
}

static (float Q1, float Q3) CalculateQuartiles(PrimitiveDataFrameColumn<float> col)
{
    var values = col.Where(v => v.HasValue).Select(v => v!.Value).OrderBy(v => v).ToList();
    if (values.Count < 4) return (values.FirstOrDefault(), values.LastOrDefault());
    
    int q1Idx = values.Count / 4;
    int q3Idx = 3 * values.Count / 4;
    return (values[q1Idx], values[q3Idx]);
}

static PrimitiveDataFrameColumn<bool> CreateMask(long count, List<long> keepIndices)
{
    var mask = new PrimitiveDataFrameColumn<bool>("mask", count);
    var keepSet = new HashSet<long>(keepIndices);
    for (long i = 0; i < count; i++)
        mask[i] = keepSet.Contains(i);
    return mask;
}

static void PrintStats(DataFrame df, string title)
{
    Console.WriteLine($"\n--- {title} ---");
    Console.WriteLine($"Rows: {df.Rows.Count}, Columns: {df.Columns.Count}");
    
    foreach (var col in df.Columns)
    {
        if (col is PrimitiveDataFrameColumn<float> numCol)
        {
            long nulls = numCol.NullCount;
            var values = numCol.Where(v => v.HasValue).Select(v => v!.Value).ToList();
            if (values.Count > 0)
            {
                Console.WriteLine($"  {col.Name}: min={values.Min():F0}, max={values.Max():F0}, " +
                                  $"mean={values.Average():F1}, nulls={nulls}");
            }
        }
    }
}
