# Chapter 13: Natural Language Processing

> "Language is the source of misunderstandings." — Antoine de Saint-Exupéry

Text is everywhere. Customer reviews, support tickets, emails, social media posts, contracts, documentation—businesses are drowning in unstructured text data that holds valuable insights, if only they could extract them. Natural Language Processing (NLP) is the field that makes this possible, and it's become one of the most impactful applications of machine learning in enterprise software.

As a C# developer, you're well-positioned for NLP work. Text processing is fundamentally string manipulation at scale, combined with statistical learning. Your experience with LINQ, regular expressions, and building data pipelines translates directly. And with ML.NET 5.0's text processing capabilities, you can build production-grade NLP systems without leaving the .NET ecosystem.

In this chapter, we'll journey from the fundamentals of text preprocessing through sentiment analysis and named entity recognition, and conclude with a complete project: a sentiment analysis system for product reviews. By the end, you'll understand how machines "read" text and how to build systems that extract meaning from human language.

## The Challenge of Text Data

Before we dive into techniques, let's understand why text is so difficult for machines.

Consider this simple sentence: "I didn't love the battery life, but the camera is amazing!"

As a human, you instantly understand:
- The overall sentiment is mixed
- Battery life is viewed negatively
- Camera is viewed positively
- "Didn't love" is a softened negative (not the same as "hated")
- The "but" signals a contrast

For a machine, this same sentence is just a sequence of characters. It has no concept of words, sentences, sentiment, or meaning. Everything we'll learn in this chapter is about bridging that gap—transforming raw text into representations that algorithms can learn from.

### Why Text Is Hard

Several factors make text processing challenging:

**Ambiguity**: "The bank was closed" could refer to a financial institution or a riverbank. "They saw her duck" has at least two readings. Context determines meaning, but context is hard to capture.

**Variability**: "great," "GREAT," "gr8," "greaaaaat," and "🔥" can all express the same sentiment. Users don't write in normalized, consistent ways.

**Sparsity**: With tens of thousands of possible words, most documents contain only a tiny fraction of the vocabulary. This creates high-dimensional, sparse data that's challenging for traditional algorithms.

**Structure**: Unlike tabular data, text has inherent sequence and structure. Word order matters: "The dog bit the man" and "The man bit the dog" contain identical words but opposite meanings.

The NLP techniques we'll explore are all strategies for handling these challenges—transforming messy, ambiguous text into clean, machine-learnable representations.

## Text Preprocessing in C#

Before any machine learning can happen, raw text needs preprocessing. This phase cleans, normalizes, and structures the text into a consistent format. In my experience, good preprocessing is worth more than algorithmic sophistication—garbage in, garbage out applies doubly to text data.

### Setting Up Your Project

Let's create a project for our NLP work:

```bash
mkdir NLPExploration
cd NLPExploration
dotnet new console -n TextAnalysis
cd TextAnalysis
dotnet add package Microsoft.ML --version 5.0.0
dotnet add package Microsoft.ML.Transforms.Text --version 5.0.0
```

### Basic Text Cleaning

Start with the fundamentals: normalization, noise removal, and standardization.

```csharp
using System.Text;
using System.Text.RegularExpressions;

public class TextPreprocessor
{
    /// <summary>
    /// Performs basic text cleaning operations
    /// </summary>
    public string CleanText(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return string.Empty;
        
        // Convert to lowercase for consistency
        var result = input.ToLowerInvariant();
        
        // Remove HTML tags if present
        result = Regex.Replace(result, @"<[^>]+>", " ");
        
        // Remove URLs
        result = Regex.Replace(result, @"https?://\S+|www\.\S+", " ");
        
        // Remove email addresses
        result = Regex.Replace(result, @"\S+@\S+\.\S+", " ");
        
        // Remove special characters, keeping basic punctuation
        result = Regex.Replace(result, @"[^\w\s\.\,\!\?\-\']", " ");
        
        // Normalize whitespace
        result = Regex.Replace(result, @"\s+", " ").Trim();
        
        return result;
    }
    
    /// <summary>
    /// Removes common stop words that add little semantic value
    /// </summary>
    public string RemoveStopWords(string input)
    {
        var stopWords = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would", 
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "it", "its", "this", "that", "these", "those", "i", "me", "my",
            "myself", "we", "our", "ours", "you", "your", "yours", "he", "him",
            "his", "she", "her", "hers", "they", "them", "their", "what", "which",
            "who", "whom", "when", "where", "why", "how", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just"
        };
        
        var words = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var filtered = words.Where(w => !stopWords.Contains(w));
        
        return string.Join(" ", filtered);
    }
}
```

### Handling Unicode and Encodings

Real-world text often contains Unicode characters, emojis, and encoding issues. Here's how to handle them:

```csharp
public class UnicodeHandler
{
    /// <summary>
    /// Normalizes Unicode text and handles common encoding issues
    /// </summary>
    public string NormalizeUnicode(string input)
    {
        // Normalize to Form C (canonical composition)
        var normalized = input.Normalize(NormalizationForm.FormC);
        
        // Remove diacritics (accents) for consistent matching
        // "café" becomes "cafe"
        var sb = new StringBuilder();
        foreach (var c in normalized.Normalize(NormalizationForm.FormD))
        {
            var category = CharUnicodeInfo.GetUnicodeCategory(c);
            if (category != UnicodeCategory.NonSpacingMark)
            {
                sb.Append(c);
            }
        }
        
        return sb.ToString().Normalize(NormalizationForm.FormC);
    }
    
    /// <summary>
    /// Converts emojis to descriptive text
    /// </summary>
    public string ConvertEmojisToText(string input)
    {
        // Common emoji to text mappings
        var emojiMap = new Dictionary<string, string>
        {
            { "😀", " happy " }, { "😃", " happy " }, { "😊", " happy " },
            { "😢", " sad " }, { "😭", " very_sad " },
            { "😡", " angry " }, { "😠", " angry " },
            { "❤️", " love " }, { "💔", " heartbreak " },
            { "👍", " thumbs_up " }, { "👎", " thumbs_down " },
            { "🔥", " excellent " }, { "💯", " perfect " },
            { "⭐", " star " }, { "🌟", " star " }
        };
        
        var result = input;
        foreach (var (emoji, text) in emojiMap)
        {
            result = result.Replace(emoji, text);
        }
        
        // Remove any remaining emojis (Unicode range for emojis)
        result = Regex.Replace(result, @"\p{So}|\p{Cs}", " ");
        
        return result;
    }
}
```

### Stemming and Lemmatization

These techniques reduce words to their base forms. "Running," "runs," and "ran" all become "run." This reduces vocabulary size and helps the model recognize that these variations represent the same concept.

```csharp
public class TextNormalizer
{
    /// <summary>
    /// Simple Porter Stemmer implementation for English
    /// For production, consider using a library like Porter2Stemmer
    /// </summary>
    public string Stem(string word)
    {
        if (string.IsNullOrEmpty(word) || word.Length < 3)
            return word;
        
        word = word.ToLowerInvariant();
        
        // Step 1: Common suffixes
        var suffixes = new[]
        {
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("isation", "ize"),
            ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
            ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
            ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"),
            ("biliti", "ble"), ("alli", "al"), ("entli", "ent"),
            ("eli", "e"), ("ousli", "ous"), ("logi", "log")
        };
        
        foreach (var (suffix, replacement) in suffixes)
        {
            if (word.EndsWith(suffix))
            {
                return word[..^suffix.Length] + replacement;
            }
        }
        
        // Step 2: Simple suffix removal
        var simpleRemovals = new[] { "ing", "ed", "ly", "ies", "es", "s" };
        foreach (var suffix in simpleRemovals)
        {
            if (word.EndsWith(suffix) && word.Length > suffix.Length + 2)
            {
                var stem = word[..^suffix.Length];
                // Handle doubling: "running" -> "run" not "runn"
                if (stem.Length >= 2 && stem[^1] == stem[^2])
                {
                    stem = stem[..^1];
                }
                return stem;
            }
        }
        
        return word;
    }
    
    /// <summary>
    /// Applies stemming to all words in a text
    /// </summary>
    public string StemText(string text)
    {
        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var stemmed = words.Select(Stem);
        return string.Join(" ", stemmed);
    }
}
```

[FIGURE: Text preprocessing pipeline diagram showing raw text flowing through normalization, cleaning, tokenization, stop word removal, and stemming stages]

## Tokenization and Vectorization

Once text is preprocessed, we need to convert it into numerical representations that machine learning algorithms can process. This happens in two stages: tokenization (breaking text into pieces) and vectorization (converting those pieces to numbers).

### Tokenization Strategies

Tokenization is the process of breaking text into smaller units. The choice of tokenization strategy significantly impacts your model's performance.

```csharp
public class Tokenizer
{
    /// <summary>
    /// Basic word tokenization using whitespace and punctuation
    /// </summary>
    public IEnumerable<string> TokenizeWords(string text)
    {
        // Split on whitespace and common punctuation
        var tokens = Regex.Split(text.ToLowerInvariant(), @"[\s\.\,\!\?\;\:\-\(\)\[\]\"\']+")
            .Where(t => !string.IsNullOrWhiteSpace(t));
        
        return tokens;
    }
    
    /// <summary>
    /// Character n-gram tokenization
    /// Useful for handling misspellings and morphological variations
    /// </summary>
    public IEnumerable<string> TokenizeCharNgrams(string text, int n = 3)
    {
        text = text.ToLowerInvariant();
        
        if (text.Length < n)
            yield break;
        
        for (int i = 0; i <= text.Length - n; i++)
        {
            yield return text.Substring(i, n);
        }
    }
    
    /// <summary>
    /// Word n-gram tokenization (bigrams, trigrams, etc.)
    /// Captures phrase-level patterns like "not good" vs "good"
    /// </summary>
    public IEnumerable<string> TokenizeWordNgrams(string text, int n = 2)
    {
        var words = TokenizeWords(text).ToArray();
        
        if (words.Length < n)
            yield break;
        
        for (int i = 0; i <= words.Length - n; i++)
        {
            yield return string.Join("_", words.Skip(i).Take(n));
        }
    }
}
```

### Bag of Words with ML.NET

The simplest vectorization approach is Bag of Words (BoW): count how many times each word appears in a document. ML.NET makes this straightforward:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

public class TextData
{
    public string Text { get; set; } = string.Empty;
}

public class TransformedTextData
{
    public float[] Features { get; set; } = Array.Empty<float>();
}

public class BagOfWordsExample
{
    public void DemonstrateBagOfWords()
    {
        var mlContext = new MLContext(seed: 42);
        
        var samples = new List<TextData>
        {
            new() { Text = "This product is amazing and I love it" },
            new() { Text = "Terrible quality, complete waste of money" },
            new() { Text = "Good value for the price, would buy again" }
        };
        
        var dataView = mlContext.Data.LoadFromEnumerable(samples);
        
        // Create the text featurization pipeline
        var pipeline = mlContext.Transforms.Text.NormalizeText(
                outputColumnName: "NormalizedText",
                inputColumnName: nameof(TextData.Text),
                caseMode: TextNormalizingEstimator.CaseMode.Lower,
                keepDiacritics: false,
                keepPunctuations: false,
                keepNumbers: true)
            .Append(mlContext.Transforms.Text.TokenizeIntoWords(
                outputColumnName: "Tokens",
                inputColumnName: "NormalizedText"))
            .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
                outputColumnName: "FilteredTokens",
                inputColumnName: "Tokens"))
            .Append(mlContext.Transforms.Text.ProduceWordBags(
                outputColumnName: "Features",
                inputColumnName: "FilteredTokens",
                weighting: NgramExtractingEstimator.WeightingCriteria.Tf));
        
        var transformer = pipeline.Fit(dataView);
        var transformedData = transformer.Transform(dataView);
        
        // Inspect the results
        var features = mlContext.Data
            .CreateEnumerable<TransformedTextData>(transformedData, reuseRowObject: false)
            .ToList();
        
        Console.WriteLine($"Feature vector length: {features[0].Features.Length}");
        Console.WriteLine($"First document features (non-zero): " +
            $"{features[0].Features.Count(f => f > 0)}");
    }
}
```

### TF-IDF: Term Frequency-Inverse Document Frequency

Raw word counts treat all words equally, but some words are more informative than others. "Amazing" appearing in a review is significant; "the" appearing is not. TF-IDF weights words by their distinctiveness:

- **Term Frequency (TF)**: How often does this word appear in *this* document?
- **Inverse Document Frequency (IDF)**: How rare is this word across *all* documents?

Words that appear frequently in one document but rarely across the corpus get high TF-IDF scores.

```csharp
public class TfIdfExample
{
    public void DemonstrateTfIdf()
    {
        var mlContext = new MLContext(seed: 42);
        
        var reviews = new List<TextData>
        {
            new() { Text = "The camera quality is excellent, takes amazing photos" },
            new() { Text = "Battery life is terrible, barely lasts a day" },
            new() { Text = "Great camera but the battery drains too fast" },
            new() { Text = "Excellent build quality, premium materials used" },
            new() { Text = "The screen is beautiful, colors are vibrant" }
        };
        
        var dataView = mlContext.Data.LoadFromEnumerable(reviews);
        
        // TF-IDF pipeline
        var pipeline = mlContext.Transforms.Text.NormalizeText(
                outputColumnName: "NormalizedText",
                inputColumnName: nameof(TextData.Text))
            .Append(mlContext.Transforms.Text.TokenizeIntoWords(
                outputColumnName: "Tokens",
                inputColumnName: "NormalizedText"))
            .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
                outputColumnName: "FilteredTokens",
                inputColumnName: "Tokens"))
            .Append(mlContext.Transforms.Text.ProduceWordBags(
                outputColumnName: "BagOfWords",
                inputColumnName: "FilteredTokens",
                weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf));
        
        var model = pipeline.Fit(dataView);
        var transformed = model.Transform(dataView);
        
        // The TF-IDF weighting gives higher scores to distinctive words
        // "camera" and "battery" will score higher than common words
    }
}
```

[FIGURE: TF-IDF calculation visualization showing how a word's score is computed from term frequency and document frequency]

### Word N-grams: Capturing Phrases

Single words miss important context. "Not good" has opposite meaning from "good," but a bag-of-words model sees them similarly. N-grams capture multi-word phrases:

```csharp
public class NgramExample
{
    public void DemonstrateNgrams()
    {
        var mlContext = new MLContext(seed: 42);
        
        var samples = new List<TextData>
        {
            new() { Text = "This is not good at all" },
            new() { Text = "This is very good indeed" },
            new() { Text = "I would not recommend this product" },
            new() { Text = "I would definitely recommend this product" }
        };
        
        var dataView = mlContext.Data.LoadFromEnumerable(samples);
        
        // Include unigrams, bigrams, and trigrams
        var pipeline = mlContext.Transforms.Text.NormalizeText(
                outputColumnName: "NormalizedText",
                inputColumnName: nameof(TextData.Text))
            .Append(mlContext.Transforms.Text.TokenizeIntoWords(
                outputColumnName: "Tokens",
                inputColumnName: "NormalizedText"))
            .Append(mlContext.Transforms.Text.ProduceNgrams(
                outputColumnName: "Ngrams",
                inputColumnName: "Tokens",
                ngramLength: 3,           // Up to trigrams
                useAllLengths: true,       // Include unigrams and bigrams too
                weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf));
        
        var model = pipeline.Fit(dataView);
        
        // Now "not_good" and "very_good" are separate features,
        // allowing the model to distinguish their different meanings
    }
}
```

## Sentiment Analysis with ML.NET

Sentiment analysis determines the emotional tone of text—positive, negative, or neutral. It's one of the most common NLP tasks in business applications: analyzing customer reviews, monitoring social media, gauging employee feedback.

### Building a Sentiment Classifier

Let's build a complete sentiment analysis pipeline using ML.NET:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

// Input data schema
public class SentimentData
{
    [LoadColumn(0)]
    public string Text { get; set; } = string.Empty;
    
    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment { get; set; }
}

// Prediction output
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
}

public class SentimentAnalyzer
{
    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private PredictionEngine<SentimentData, SentimentPrediction>? _predictionEngine;
    
    public SentimentAnalyzer()
    {
        _mlContext = new MLContext(seed: 42);
    }
    
    public void TrainModel(string dataPath)
    {
        Console.WriteLine("Loading training data...");
        
        // Load data
        var dataView = _mlContext.Data.LoadFromTextFile<SentimentData>(
            dataPath,
            hasHeader: true,
            separatorChar: '\t');
        
        // Split into training and test sets
        var splitData = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
        
        Console.WriteLine("Building pipeline...");
        
        // Define the training pipeline
        var pipeline = _mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(SentimentData.Text))
            .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features"));
        
        Console.WriteLine("Training model...");
        
        // Train the model
        _model = pipeline.Fit(splitData.TrainSet);
        
        // Evaluate on test set
        var predictions = _model.Transform(splitData.TestSet);
        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");
        
        Console.WriteLine($"\nModel Performance:");
        Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
        Console.WriteLine($"  AUC:      {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"  F1 Score: {metrics.F1Score:P2}");
        Console.WriteLine($"  Precision:{metrics.PositivePrecision:P2}");
        Console.WriteLine($"  Recall:   {metrics.PositiveRecall:P2}");
        
        // Create prediction engine for single predictions
        _predictionEngine = _mlContext.Model
            .CreatePredictionEngine<SentimentData, SentimentPrediction>(_model);
    }
    
    public SentimentPrediction Predict(string text)
    {
        if (_predictionEngine == null)
            throw new InvalidOperationException("Model not trained. Call TrainModel first.");
        
        return _predictionEngine.Predict(new SentimentData { Text = text });
    }
    
    public void SaveModel(string modelPath)
    {
        if (_model == null)
            throw new InvalidOperationException("No model to save.");
        
        _mlContext.Model.Save(_model, null, modelPath);
        Console.WriteLine($"Model saved to {modelPath}");
    }
    
    public void LoadModel(string modelPath)
    {
        _model = _mlContext.Model.Load(modelPath, out _);
        _predictionEngine = _mlContext.Model
            .CreatePredictionEngine<SentimentData, SentimentPrediction>(_model);
        Console.WriteLine($"Model loaded from {modelPath}");
    }
}
```

### Using FeaturizeText: ML.NET's Power Feature

The `FeaturizeText` transform is ML.NET's Swiss Army knife for text. It combines multiple operations into a single, optimized pipeline:

```csharp
public class TextFeaturizationOptions
{
    public void DemonstrateFeaturizeText()
    {
        var mlContext = new MLContext(seed: 42);
        
        var samples = new List<SentimentData>
        {
            new() { Text = "I absolutely love this product!", Sentiment = true },
            new() { Text = "Worst purchase I've ever made", Sentiment = false },
            new() { Text = "Does exactly what it says, great value", Sentiment = true },
            new() { Text = "Broke after one week, terrible quality", Sentiment = false }
        };
        
        var dataView = mlContext.Data.LoadFromEnumerable(samples);
        
        // Configure FeaturizeText with specific options
        var options = new TextFeaturizingEstimator.Options
        {
            // Word-level features
            WordFeatureExtractor = new WordBagEstimator.Options
            {
                NgramLength = 2,           // Include bigrams
                UseAllLengths = true,      // Also include unigrams
                Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf
            },
            
            // Character-level features (helps with typos, morphology)
            CharFeatureExtractor = new WordBagEstimator.Options
            {
                NgramLength = 4,
                UseAllLengths = false,
                Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf
            },
            
            // Normalization options
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            KeepDiacritics = false,
            KeepPunctuations = false,
            KeepNumbers = true,
            
            // Include stop words removal
            StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options
            {
                Language = TextNormalizingEstimator.Language.English
            }
        };
        
        var pipeline = mlContext.Transforms.Text.FeaturizeText(
            outputColumnName: "Features",
            options: options,
            inputColumnNames: nameof(SentimentData.Text));
        
        var model = pipeline.Fit(dataView);
        var transformed = model.Transform(dataView);
        
        // Preview the schema
        var schema = transformed.Schema;
        var featuresColumn = schema["Features"];
        Console.WriteLine($"Features column type: {featuresColumn.Type}");
    }
}
```

### Handling Imbalanced Data

Real-world sentiment datasets are often imbalanced—far more positive reviews than negative, or vice versa. Here's how to handle this:

```csharp
public class ImbalancedSentimentTrainer
{
    public void TrainWithBalancing(MLContext mlContext, IDataView data)
    {
        // Check class distribution
        var preview = mlContext.Data.CreateEnumerable<SentimentData>(data, false).ToList();
        var positiveCount = preview.Count(x => x.Sentiment);
        var negativeCount = preview.Count(x => !x.Sentiment);
        
        Console.WriteLine($"Class distribution: {positiveCount} positive, {negativeCount} negative");
        
        // Option 1: Use class weights
        // Higher weight for minority class
        var minorityWeight = (float)Math.Max(positiveCount, negativeCount) / 
                            Math.Min(positiveCount, negativeCount);
        
        var pipeline = mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",
                inputColumnName: nameof(SentimentData.Text))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "Label",
                featureColumnName: "Features",
                exampleWeightColumnName: null)); // Would add weights here
        
        // Option 2: Oversample minority class
        var oversampledData = OversampleMinorityClass(mlContext, data, preview, positiveCount, negativeCount);
        
        // Option 3: Use algorithms robust to imbalance (e.g., LightGbm with built-in handling)
    }
    
    private IDataView OversampleMinorityClass(MLContext mlContext, IDataView data,
        List<SentimentData> samples, int positiveCount, int negativeCount)
    {
        var isPositiveMinority = positiveCount < negativeCount;
        var minorityClass = samples.Where(x => x.Sentiment == isPositiveMinority).ToList();
        var ratio = Math.Max(positiveCount, negativeCount) / 
                   Math.Min(positiveCount, negativeCount);
        
        // Duplicate minority samples
        var oversampled = new List<SentimentData>(samples);
        var random = new Random(42);
        
        for (int i = 0; i < (ratio - 1) * minorityClass.Count; i++)
        {
            oversampled.Add(minorityClass[random.Next(minorityClass.Count)]);
        }
        
        // Shuffle
        oversampled = oversampled.OrderBy(_ => random.Next()).ToList();
        
        return mlContext.Data.LoadFromEnumerable(oversampled);
    }
}
```

[FIGURE: Confusion matrix visualization for sentiment classification showing true positives, false positives, true negatives, and false negatives]

## Named Entity Recognition

Named Entity Recognition (NER) identifies and classifies named entities in text—people, organizations, locations, dates, monetary values, and more. It's essential for information extraction, question answering, and building knowledge graphs.

### Understanding NER

Consider this sentence: "Microsoft announced that CEO Satya Nadella will visit Berlin next Tuesday."

A NER system should identify:
- **Microsoft** → Organization
- **Satya Nadella** → Person
- **Berlin** → Location
- **next Tuesday** → Date

### Rule-Based NER in C#

For domain-specific entities with predictable patterns, rule-based approaches work well:

```csharp
using System.Text.RegularExpressions;

public class Entity
{
    public string Text { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
    public int StartIndex { get; set; }
    public int EndIndex { get; set; }
}

public class RuleBasedNER
{
    private readonly Dictionary<string, Regex> _patterns;
    private readonly Dictionary<string, HashSet<string>> _gazetteers;
    
    public RuleBasedNER()
    {
        // Regex patterns for structured entities
        _patterns = new Dictionary<string, Regex>
        {
            // Email addresses
            ["EMAIL"] = new Regex(@"\b[\w\.-]+@[\w\.-]+\.\w+\b", RegexOptions.Compiled),
            
            // Phone numbers (US format)
            ["PHONE"] = new Regex(@"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", 
                RegexOptions.Compiled),
            
            // Dates
            ["DATE"] = new Regex(@"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|" +
                @"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|" +
                @"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday))\b",
                RegexOptions.Compiled | RegexOptions.IgnoreCase),
            
            // Currency amounts
            ["MONEY"] = new Regex(@"\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)",
                RegexOptions.Compiled),
            
            // Product codes (example: ABC-12345)
            ["PRODUCT_CODE"] = new Regex(@"\b[A-Z]{2,4}-\d{4,6}\b", RegexOptions.Compiled)
        };
        
        // Gazetteers (word lists) for known entities
        _gazetteers = new Dictionary<string, HashSet<string>>
        {
            ["COMPANY"] = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "Microsoft", "Apple", "Google", "Amazon", "Meta", "Netflix",
                "Tesla", "OpenAI", "Anthropic", "IBM", "Intel", "NVIDIA"
            },
            
            ["CITY"] = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                "San Francisco", "Seattle", "Boston", "Denver", "Austin"
            }
        };
    }
    
    public List<Entity> ExtractEntities(string text)
    {
        var entities = new List<Entity>();
        
        // Pattern-based extraction
        foreach (var (entityType, pattern) in _patterns)
        {
            foreach (Match match in pattern.Matches(text))
            {
                entities.Add(new Entity
                {
                    Text = match.Value,
                    Type = entityType,
                    StartIndex = match.Index,
                    EndIndex = match.Index + match.Length
                });
            }
        }
        
        // Gazetteer-based extraction
        foreach (var (entityType, gazetteer) in _gazetteers)
        {
            foreach (var term in gazetteer)
            {
                var index = text.IndexOf(term, StringComparison.OrdinalIgnoreCase);
                while (index >= 0)
                {
                    // Check word boundaries
                    var isWordStart = index == 0 || !char.IsLetterOrDigit(text[index - 1]);
                    var endIndex = index + term.Length;
                    var isWordEnd = endIndex >= text.Length || !char.IsLetterOrDigit(text[endIndex]);
                    
                    if (isWordStart && isWordEnd)
                    {
                        entities.Add(new Entity
                        {
                            Text = text.Substring(index, term.Length),
                            Type = entityType,
                            StartIndex = index,
                            EndIndex = endIndex
                        });
                    }
                    
                    index = text.IndexOf(term, endIndex, StringComparison.OrdinalIgnoreCase);
                }
            }
        }
        
        // Remove overlapping entities (keep longest)
        return ResolveOverlaps(entities);
    }
    
    private List<Entity> ResolveOverlaps(List<Entity> entities)
    {
        var sorted = entities.OrderBy(e => e.StartIndex).ThenByDescending(e => e.EndIndex).ToList();
        var result = new List<Entity>();
        
        foreach (var entity in sorted)
        {
            // Check if this entity overlaps with any already selected
            var overlaps = result.Any(e => 
                entity.StartIndex < e.EndIndex && entity.EndIndex > e.StartIndex);
            
            if (!overlaps)
            {
                result.Add(entity);
            }
        }
        
        return result.OrderBy(e => e.StartIndex).ToList();
    }
}
```

### Using Pre-trained NER Models

For general-purpose NER, pre-trained models offer better accuracy. You can use ONNX Runtime to run models trained in Python:

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class OnnxNERModel
{
    private readonly InferenceSession _session;
    private readonly Dictionary<string, int> _tokenToId;
    private readonly Dictionary<int, string> _idToLabel;
    
    public OnnxNERModel(string modelPath, string vocabPath, string labelsPath)
    {
        _session = new InferenceSession(modelPath);
        _tokenToId = LoadVocabulary(vocabPath);
        _idToLabel = LoadLabels(labelsPath);
    }
    
    private Dictionary<string, int> LoadVocabulary(string path)
    {
        return File.ReadAllLines(path)
            .Select((token, index) => (token, index))
            .ToDictionary(x => x.token, x => x.index);
    }
    
    private Dictionary<int, string> LoadLabels(string path)
    {
        return File.ReadAllLines(path)
            .Select((label, index) => (label, index))
            .ToDictionary(x => x.index, x => x.label);
    }
    
    public List<(string Token, string Label)> Predict(string text)
    {
        // Tokenize
        var tokens = text.Split(' ');
        var tokenIds = tokens
            .Select(t => _tokenToId.GetValueOrDefault(t.ToLower(), _tokenToId["[UNK]"]))
            .ToArray();
        
        // Create input tensor
        var inputTensor = new DenseTensor<long>(
            tokenIds.Select(id => (long)id).ToArray(),
            new[] { 1, tokenIds.Length });
        
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputTensor)
        };
        
        // Run inference
        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();
        
        // Decode predictions
        var predictions = new List<(string Token, string Label)>();
        
        for (int i = 0; i < tokens.Length; i++)
        {
            // Find argmax for this position
            var maxScore = float.MinValue;
            var maxLabel = 0;
            
            for (int j = 0; j < _idToLabel.Count; j++)
            {
                var score = output[0, i, j];
                if (score > maxScore)
                {
                    maxScore = score;
                    maxLabel = j;
                }
            }
            
            predictions.Add((tokens[i], _idToLabel[maxLabel]));
        }
        
        return predictions;
    }
    
    public void Dispose()
    {
        _session?.Dispose();
    }
}
```

## Introduction to Transformers

No discussion of modern NLP would be complete without transformers. Since the landmark "Attention Is All You Need" paper in 2017, transformers have revolutionized the field. BERT, GPT, and their descendants now power everything from search engines to chatbots.

### What Are Transformers?

Traditional approaches processed text sequentially—reading one word at a time, left to right. Transformers introduced **self-attention**, allowing the model to consider all words simultaneously and learn relationships between any pair of words, regardless of distance.

Consider: "The bank was closed because it was a holiday."

A sequential model might struggle to connect "it" with "bank" across several words. Self-attention naturally captures this relationship by computing attention scores between all word pairs.

[FIGURE: Self-attention mechanism visualization showing how the word "it" attends to "bank" in the sentence]

### Transformer Architecture Overview

Without diving into the mathematical details, here's what you need to know:

1. **Tokenization**: Text is split into subword tokens (e.g., "unhappiness" → "un", "happi", "ness")

2. **Embedding**: Each token becomes a dense vector (typically 768-1024 dimensions)

3. **Positional Encoding**: Since attention doesn't inherently understand position, we add position information to embeddings

4. **Self-Attention Layers**: Multiple layers that let each token attend to every other token, learning contextual representations

5. **Output**: Task-specific heads for classification, token labeling, generation, etc.

### Using Transformer Models in C#

While you won't train transformers from scratch in C#, you can absolutely use them:

```csharp
// Using a transformer model exported to ONNX format
public class TransformerSentimentClassifier
{
    private readonly InferenceSession _session;
    private readonly BertTokenizer _tokenizer;
    
    public TransformerSentimentClassifier(string modelPath, string vocabPath)
    {
        _session = new InferenceSession(modelPath);
        _tokenizer = new BertTokenizer(vocabPath);
    }
    
    public (string Label, float Confidence) Classify(string text)
    {
        // Tokenize input
        var encoding = _tokenizer.Encode(text, maxLength: 512);
        
        // Create tensors
        var inputIds = new DenseTensor<long>(
            encoding.InputIds.Select(x => (long)x).ToArray(),
            new[] { 1, encoding.InputIds.Length });
        
        var attentionMask = new DenseTensor<long>(
            encoding.AttentionMask.Select(x => (long)x).ToArray(),
            new[] { 1, encoding.AttentionMask.Length });
        
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask)
        };
        
        // Run inference
        using var results = _session.Run(inputs);
        var logits = results.First().AsTensor<float>();
        
        // Softmax to get probabilities
        var expScores = new[] { MathF.Exp(logits[0, 0]), MathF.Exp(logits[0, 1]) };
        var sum = expScores.Sum();
        var probs = expScores.Select(s => s / sum).ToArray();
        
        // Return prediction
        var isPositive = probs[1] > probs[0];
        var confidence = isPositive ? probs[1] : probs[0];
        
        return (isPositive ? "Positive" : "Negative", confidence);
    }
}

// Simple BERT tokenizer implementation
public class BertTokenizer
{
    private readonly Dictionary<string, int> _vocab;
    private const int ClsTokenId = 101;
    private const int SepTokenId = 102;
    private const int PadTokenId = 0;
    private const int UnkTokenId = 100;
    
    public BertTokenizer(string vocabPath)
    {
        _vocab = File.ReadAllLines(vocabPath)
            .Select((token, idx) => (token, idx))
            .ToDictionary(x => x.token, x => x.idx);
    }
    
    public BertEncoding Encode(string text, int maxLength = 512)
    {
        // Basic word-piece tokenization
        var tokens = new List<int> { ClsTokenId };
        
        var words = text.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        foreach (var word in words)
        {
            if (_vocab.TryGetValue(word, out var id))
            {
                tokens.Add(id);
            }
            else
            {
                // Simple subword fallback
                tokens.Add(UnkTokenId);
            }
            
            if (tokens.Count >= maxLength - 1) break;
        }
        
        tokens.Add(SepTokenId);
        
        // Pad to max length
        var attentionMask = Enumerable.Repeat(1, tokens.Count)
            .Concat(Enumerable.Repeat(0, maxLength - tokens.Count))
            .ToArray();
        
        while (tokens.Count < maxLength)
            tokens.Add(PadTokenId);
        
        return new BertEncoding
        {
            InputIds = tokens.ToArray(),
            AttentionMask = attentionMask
        };
    }
}

public class BertEncoding
{
    public int[] InputIds { get; set; } = Array.Empty<int>();
    public int[] AttentionMask { get; set; } = Array.Empty<int>();
}
```

### When to Use Transformers vs. Traditional ML

**Use transformers when:**
- You have sufficient data (thousands of examples minimum)
- Accuracy is critical
- You can accept higher latency (10-100ms vs. <1ms)
- Pre-trained models exist for your domain or language
- You have GPU resources available

**Use traditional ML (TF-IDF + classifier) when:**
- You need very fast inference (<1ms)
- Your data is limited
- The task is straightforward (binary sentiment, simple classification)
- You need interpretability
- Resources are constrained

For many production applications, traditional ML still wins on the cost-benefit tradeoff. Transformers are powerful, but not always necessary.

## Project: Sentiment Analysis for Product Reviews

Let's bring everything together with a complete project: building a production-ready sentiment analysis system for e-commerce product reviews.

### Project Setup

```bash
mkdir ProductReviewAnalyzer
cd ProductReviewAnalyzer
dotnet new console -n ReviewAnalyzer
cd ReviewAnalyzer

dotnet add package Microsoft.ML --version 5.0.0
dotnet add package Microsoft.ML.Transforms.Text --version 5.0.0
dotnet add package Microsoft.ML.FastTree --version 5.0.0
dotnet add package CsvHelper --version 30.0.1
```

### Data Model

```csharp
using CsvHelper.Configuration.Attributes;
using Microsoft.ML.Data;

namespace ReviewAnalyzer.Models;

/// <summary>
/// Raw review data as loaded from CSV
/// </summary>
public class ProductReview
{
    [Name("review_id")]
    public string ReviewId { get; set; } = string.Empty;
    
    [Name("product_id")]
    public string ProductId { get; set; } = string.Empty;
    
    [Name("review_text")]
    public string ReviewText { get; set; } = string.Empty;
    
    [Name("rating")]
    public int Rating { get; set; }
    
    [Name("verified_purchase")]
    public bool VerifiedPurchase { get; set; }
    
    [Name("review_date")]
    public DateTime ReviewDate { get; set; }
}

/// <summary>
/// ML.NET training data with sentiment label
/// </summary>
public class ReviewTrainingData
{
    [LoadColumn(0)]
    public string ReviewText { get; set; } = string.Empty;
    
    [LoadColumn(1)]
    public string ProductCategory { get; set; } = string.Empty;
    
    [LoadColumn(2)]
    public bool VerifiedPurchase { get; set; }
    
    [LoadColumn(3), ColumnName("Label")]
    public bool Sentiment { get; set; } // true = positive (4-5 stars), false = negative (1-2 stars)
}

/// <summary>
/// Model output with confidence scores
/// </summary>
public class ReviewPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Sentiment { get; set; }
    
    public float Probability { get; set; }
    
    public float Score { get; set; }
    
    public string SentimentLabel => Sentiment ? "Positive" : "Negative";
    
    public string Confidence => Probability switch
    {
        > 0.9f => "Very High",
        > 0.75f => "High",
        > 0.6f => "Moderate",
        _ => "Low"
    };
}
```

### Data Preprocessing Pipeline

```csharp
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace ReviewAnalyzer.Processing;

public class ReviewPreprocessor
{
    private static readonly HashSet<string> StopWords = new(StringComparer.OrdinalIgnoreCase)
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "was", "are", "were", "been", "be",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "it", "its", "this", "that", "i", "me", "my", "we", "our",
        "you", "your", "he", "him", "his", "she", "her", "they", "them", "their"
    };
    
    private static readonly Dictionary<string, string> Contractions = new()
    {
        { "won't", "will not" }, { "can't", "cannot" }, { "n't", " not" },
        { "'re", " are" }, { "'s", " is" }, { "'d", " would" },
        { "'ll", " will" }, { "'ve", " have" }, { "'m", " am" }
    };
    
    public string Preprocess(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;
        
        var result = text.ToLowerInvariant();
        
        // Expand contractions (important for sentiment!)
        // "don't like" should become "do not like"
        foreach (var (contraction, expansion) in Contractions)
        {
            result = result.Replace(contraction, expansion);
        }
        
        // Remove HTML
        result = Regex.Replace(result, @"<[^>]+>", " ");
        
        // Remove URLs
        result = Regex.Replace(result, @"https?://\S+|www\.\S+", " ");
        
        // Handle emphasis markers
        result = result.Replace("***", " very_emphasized ")
                      .Replace("**", " emphasized ")
                      .Replace("!!", " very_exclaimed ");
        
        // Convert repeated characters: "soooo" -> "so"
        result = Regex.Replace(result, @"(.)\1{2,}", "$1");
        
        // Remove special characters but keep sentiment punctuation
        result = Regex.Replace(result, @"[^\w\s\!\?\-]", " ");
        
        // Normalize whitespace
        result = Regex.Replace(result, @"\s+", " ").Trim();
        
        return result;
    }
    
    public string PreprocessWithStopWordRemoval(string text)
    {
        var cleaned = Preprocess(text);
        var words = cleaned.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var filtered = words.Where(w => !StopWords.Contains(w) && w.Length > 1);
        return string.Join(" ", filtered);
    }
    
    /// <summary>
    /// Extract aspect-sentiment pairs from review text
    /// e.g., "great battery life but poor camera" -> 
    /// [("battery life", positive), ("camera", negative)]
    /// </summary>
    public List<(string Aspect, string Sentiment)> ExtractAspectSentiments(string text)
    {
        var aspects = new List<(string Aspect, string Sentiment)>();
        
        // Common product aspects
        var aspectPatterns = new[]
        {
            @"(battery\s*life|battery)",
            @"(screen|display)",
            @"(camera|photo\s*quality|picture\s*quality)",
            @"(sound|audio|speaker)",
            @"(build\s*quality|construction|durability)",
            @"(price|value|cost)",
            @"(performance|speed)",
            @"(design|look|appearance)"
        };
        
        var positiveMarkers = new[] { "great", "good", "excellent", "amazing", "love", "best", "perfect", "fantastic" };
        var negativeMarkers = new[] { "bad", "poor", "terrible", "awful", "worst", "hate", "horrible", "disappointing" };
        
        var processedText = text.ToLowerInvariant();
        
        foreach (var pattern in aspectPatterns)
        {
            var match = Regex.Match(processedText, pattern);
            if (!match.Success) continue;
            
            var aspect = match.Value;
            var startIndex = Math.Max(0, match.Index - 30);
            var length = Math.Min(processedText.Length - startIndex, 60 + match.Length);
            var context = processedText.Substring(startIndex, length);
            
            var hasPositive = positiveMarkers.Any(m => context.Contains(m));
            var hasNegative = negativeMarkers.Any(m => context.Contains(m));
            
            // Check for negation
            var negationPattern = @"not\s+\w*|no\s+\w*|never\s+\w*|n't\s+\w*";
            var hasNegation = Regex.IsMatch(context, negationPattern);
            
            if (hasNegation)
            {
                // Flip sentiment
                (hasPositive, hasNegative) = (hasNegative, hasPositive);
            }
            
            if (hasPositive && !hasNegative)
                aspects.Add((aspect, "positive"));
            else if (hasNegative && !hasPositive)
                aspects.Add((aspect, "negative"));
            else if (hasPositive && hasNegative)
                aspects.Add((aspect, "mixed"));
        }
        
        return aspects;
    }
}
```

### Model Training

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using ReviewAnalyzer.Models;

namespace ReviewAnalyzer.Training;

public class SentimentModelTrainer
{
    private readonly MLContext _mlContext;
    private readonly string _modelPath;
    
    public SentimentModelTrainer(string modelPath)
    {
        _mlContext = new MLContext(seed: 42);
        _modelPath = modelPath;
    }
    
    public TrainingResult Train(string dataPath)
    {
        Console.WriteLine("=== Review Sentiment Model Training ===\n");
        
        // Load data
        Console.WriteLine("Loading training data...");
        var dataView = _mlContext.Data.LoadFromTextFile<ReviewTrainingData>(
            dataPath,
            hasHeader: true,
            separatorChar: '\t');
        
        var totalRows = dataView.GetRowCount() ?? 0;
        Console.WriteLine($"Loaded {totalRows:N0} reviews\n");
        
        // Split data
        var split = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 42);
        
        // Build pipeline
        Console.WriteLine("Building training pipeline...");
        
        var pipeline = BuildPipeline();
        
        // Train
        Console.WriteLine("Training model (this may take a few minutes)...\n");
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        
        var model = pipeline.Fit(split.TrainSet);
        
        stopwatch.Stop();
        Console.WriteLine($"Training completed in {stopwatch.Elapsed.TotalSeconds:F1} seconds\n");
        
        // Evaluate
        Console.WriteLine("Evaluating model...");
        var predictions = model.Transform(split.TestSet);
        var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");
        
        // Print metrics
        PrintMetrics(metrics);
        
        // Save model
        Console.WriteLine($"\nSaving model to {_modelPath}...");
        _mlContext.Model.Save(model, dataView.Schema, _modelPath);
        Console.WriteLine("Model saved successfully!");
        
        return new TrainingResult
        {
            Accuracy = metrics.Accuracy,
            F1Score = metrics.F1Score,
            AUC = metrics.AreaUnderRocCurve,
            TrainingTimeSeconds = stopwatch.Elapsed.TotalSeconds,
            ModelPath = _modelPath
        };
    }
    
    private IEstimator<ITransformer> BuildPipeline()
    {
        // Text featurization with optimized settings
        var textOptions = new TextFeaturizingEstimator.Options
        {
            WordFeatureExtractor = new WordBagEstimator.Options
            {
                NgramLength = 2,
                UseAllLengths = true,
                Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf,
                MaximumNgramsCount = new[] { 10000, 5000 }
            },
            CharFeatureExtractor = new WordBagEstimator.Options
            {
                NgramLength = 4,
                UseAllLengths = false,
                Weighting = NgramExtractingEstimator.WeightingCriteria.TfIdf,
                MaximumNgramsCount = new[] { 5000 }
            },
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            KeepDiacritics = false,
            KeepPunctuations = true, // Keep ! and ? for sentiment
            KeepNumbers = false,
            StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options
            {
                Language = TextNormalizingEstimator.Language.English
            }
        };
        
        // Convert boolean verified purchase to float
        var pipeline = _mlContext.Transforms.Conversion.ConvertType(
                outputColumnName: "VerifiedPurchaseFloat",
                inputColumnName: nameof(ReviewTrainingData.VerifiedPurchase),
                outputKind: DataKind.Single)
            
            // Featurize review text
            .Append(_mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "TextFeatures",
                options: textOptions,
                inputColumnNames: nameof(ReviewTrainingData.ReviewText)))
            
            // One-hot encode product category
            .Append(_mlContext.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "CategoryFeatures",
                inputColumnName: nameof(ReviewTrainingData.ProductCategory)))
            
            // Combine all features
            .Append(_mlContext.Transforms.Concatenate(
                outputColumnName: "Features",
                "TextFeatures",
                "CategoryFeatures",
                "VerifiedPurchaseFloat"))
            
            // Train with FastTree (gradient boosted trees)
            .Append(_mlContext.BinaryClassification.Trainers.FastTree(
                labelColumnName: "Label",
                featureColumnName: "Features",
                numberOfLeaves: 50,
                numberOfTrees: 100,
                minimumExampleCountPerLeaf: 10,
                learningRate: 0.1));
        
        return pipeline;
    }
    
    private void PrintMetrics(BinaryClassificationMetrics metrics)
    {
        Console.WriteLine("╔═══════════════════════════════════════════╗");
        Console.WriteLine("║          Model Performance Metrics        ║");
        Console.WriteLine("╠═══════════════════════════════════════════╣");
        Console.WriteLine($"║  Accuracy:          {metrics.Accuracy,8:P2}            ║");
        Console.WriteLine($"║  AUC-ROC:           {metrics.AreaUnderRocCurve,8:P2}            ║");
        Console.WriteLine($"║  F1 Score:          {metrics.F1Score,8:P2}            ║");
        Console.WriteLine($"║  Precision:         {metrics.PositivePrecision,8:P2}            ║");
        Console.WriteLine($"║  Recall:            {metrics.PositiveRecall,8:P2}            ║");
        Console.WriteLine("╠═══════════════════════════════════════════╣");
        Console.WriteLine($"║  Negative Precision:{metrics.NegativePrecision,8:P2}            ║");
        Console.WriteLine($"║  Negative Recall:   {metrics.NegativeRecall,8:P2}            ║");
        Console.WriteLine("╚═══════════════════════════════════════════╝");
    }
}

public class TrainingResult
{
    public double Accuracy { get; set; }
    public double F1Score { get; set; }
    public double AUC { get; set; }
    public double TrainingTimeSeconds { get; set; }
    public string ModelPath { get; set; } = string.Empty;
}
```

### Inference Service

```csharp
using Microsoft.ML;
using ReviewAnalyzer.Models;
using ReviewAnalyzer.Processing;

namespace ReviewAnalyzer.Services;

public class ReviewAnalysisService : IDisposable
{
    private readonly MLContext _mlContext;
    private readonly PredictionEngine<ReviewTrainingData, ReviewPrediction> _predictionEngine;
    private readonly ReviewPreprocessor _preprocessor;
    
    public ReviewAnalysisService(string modelPath)
    {
        _mlContext = new MLContext();
        _preprocessor = new ReviewPreprocessor();
        
        // Load the trained model
        var model = _mlContext.Model.Load(modelPath, out _);
        _predictionEngine = _mlContext.Model
            .CreatePredictionEngine<ReviewTrainingData, ReviewPrediction>(model);
    }
    
    /// <summary>
    /// Analyze a single review
    /// </summary>
    public ReviewAnalysisResult AnalyzeReview(string reviewText, string productCategory = "General", bool verifiedPurchase = false)
    {
        // Preprocess the text
        var processedText = _preprocessor.Preprocess(reviewText);
        
        // Get sentiment prediction
        var input = new ReviewTrainingData
        {
            ReviewText = processedText,
            ProductCategory = productCategory,
            VerifiedPurchase = verifiedPurchase
        };
        
        var prediction = _predictionEngine.Predict(input);
        
        // Extract aspect-level sentiments
        var aspectSentiments = _preprocessor.ExtractAspectSentiments(reviewText);
        
        return new ReviewAnalysisResult
        {
            OriginalText = reviewText,
            ProcessedText = processedText,
            OverallSentiment = prediction.SentimentLabel,
            Confidence = prediction.Probability,
            ConfidenceLevel = prediction.Confidence,
            AspectSentiments = aspectSentiments
        };
    }
    
    /// <summary>
    /// Analyze multiple reviews in batch
    /// </summary>
    public IEnumerable<ReviewAnalysisResult> AnalyzeBatch(
        IEnumerable<ProductReview> reviews,
        IProgress<int>? progress = null)
    {
        var reviewList = reviews.ToList();
        var results = new List<ReviewAnalysisResult>();
        var processed = 0;
        
        foreach (var review in reviewList)
        {
            var result = AnalyzeReview(
                review.ReviewText,
                "Electronics", // Default category
                review.VerifiedPurchase);
            
            result.ReviewId = review.ReviewId;
            result.ProductId = review.ProductId;
            result.Rating = review.Rating;
            
            results.Add(result);
            
            processed++;
            progress?.Report((processed * 100) / reviewList.Count);
        }
        
        return results;
    }
    
    /// <summary>
    /// Generate summary statistics for a collection of reviews
    /// </summary>
    public ReviewSummaryStatistics GenerateSummary(IEnumerable<ReviewAnalysisResult> results)
    {
        var resultList = results.ToList();
        
        var positiveCount = resultList.Count(r => r.OverallSentiment == "Positive");
        var negativeCount = resultList.Count(r => r.OverallSentiment == "Negative");
        
        // Aggregate aspect sentiments
        var aspectCounts = resultList
            .SelectMany(r => r.AspectSentiments)
            .GroupBy(a => a.Aspect)
            .Select(g => new AspectSummary
            {
                Aspect = g.Key,
                TotalMentions = g.Count(),
                PositiveCount = g.Count(x => x.Sentiment == "positive"),
                NegativeCount = g.Count(x => x.Sentiment == "negative"),
                MixedCount = g.Count(x => x.Sentiment == "mixed")
            })
            .OrderByDescending(a => a.TotalMentions)
            .ToList();
        
        return new ReviewSummaryStatistics
        {
            TotalReviews = resultList.Count,
            PositiveCount = positiveCount,
            NegativeCount = negativeCount,
            PositivePercentage = (double)positiveCount / resultList.Count,
            AverageConfidence = resultList.Average(r => r.Confidence),
            AspectSummaries = aspectCounts,
            HighConfidencePositive = resultList.Count(r => r.OverallSentiment == "Positive" && r.Confidence > 0.8),
            HighConfidenceNegative = resultList.Count(r => r.OverallSentiment == "Negative" && r.Confidence > 0.8)
        };
    }
    
    public void Dispose()
    {
        _predictionEngine?.Dispose();
    }
}

public class ReviewAnalysisResult
{
    public string ReviewId { get; set; } = string.Empty;
    public string ProductId { get; set; } = string.Empty;
    public string OriginalText { get; set; } = string.Empty;
    public string ProcessedText { get; set; } = string.Empty;
    public string OverallSentiment { get; set; } = string.Empty;
    public float Confidence { get; set; }
    public string ConfidenceLevel { get; set; } = string.Empty;
    public int Rating { get; set; }
    public List<(string Aspect, string Sentiment)> AspectSentiments { get; set; } = new();
}

public class ReviewSummaryStatistics
{
    public int TotalReviews { get; set; }
    public int PositiveCount { get; set; }
    public int NegativeCount { get; set; }
    public double PositivePercentage { get; set; }
    public double AverageConfidence { get; set; }
    public int HighConfidencePositive { get; set; }
    public int HighConfidenceNegative { get; set; }
    public List<AspectSummary> AspectSummaries { get; set; } = new();
}

public class AspectSummary
{
    public string Aspect { get; set; } = string.Empty;
    public int TotalMentions { get; set; }
    public int PositiveCount { get; set; }
    public int NegativeCount { get; set; }
    public int MixedCount { get; set; }
    public double SentimentScore => TotalMentions > 0 
        ? (PositiveCount - NegativeCount) / (double)TotalMentions 
        : 0;
}
```

### Main Application

```csharp
using ReviewAnalyzer.Services;
using ReviewAnalyzer.Training;

namespace ReviewAnalyzer;

class Program
{
    static async Task Main(string[] args)
    {
        Console.WriteLine("╔══════════════════════════════════════════════════╗");
        Console.WriteLine("║     Product Review Sentiment Analysis System     ║");
        Console.WriteLine("╚══════════════════════════════════════════════════╝\n");
        
        var modelPath = "sentiment_model.zip";
        var dataPath = "reviews_training.tsv";
        
        // Check if model exists, train if not
        if (!File.Exists(modelPath))
        {
            if (!File.Exists(dataPath))
            {
                Console.WriteLine("Creating sample training data...");
                CreateSampleTrainingData(dataPath);
            }
            
            var trainer = new SentimentModelTrainer(modelPath);
            trainer.Train(dataPath);
            Console.WriteLine();
        }
        
        // Initialize the service
        using var service = new ReviewAnalysisService(modelPath);
        
        // Interactive demo
        Console.WriteLine("Enter product reviews to analyze (or 'quit' to exit):\n");
        
        while (true)
        {
            Console.Write("Review: ");
            var input = Console.ReadLine();
            
            if (string.IsNullOrWhiteSpace(input) || input.Equals("quit", StringComparison.OrdinalIgnoreCase))
                break;
            
            var result = service.AnalyzeReview(input, "Electronics", verifiedPurchase: true);
            
            Console.WriteLine($"\n┌─────────────────────────────────────────────────┐");
            Console.WriteLine($"│ Sentiment: {result.OverallSentiment,-10} Confidence: {result.Confidence:P0} ({result.ConfidenceLevel})");
            Console.WriteLine($"└─────────────────────────────────────────────────┘");
            
            if (result.AspectSentiments.Any())
            {
                Console.WriteLine("  Aspects detected:");
                foreach (var (aspect, sentiment) in result.AspectSentiments)
                {
                    var emoji = sentiment switch
                    {
                        "positive" => "✓",
                        "negative" => "✗",
                        _ => "~"
                    };
                    Console.WriteLine($"    {emoji} {aspect}: {sentiment}");
                }
            }
            
            Console.WriteLine();
        }
        
        Console.WriteLine("Thank you for using the Review Analyzer!");
    }
    
    static void CreateSampleTrainingData(string path)
    {
        var samples = new List<string>
        {
            // Positive reviews
            "This phone is amazing! Great camera and the battery lasts all day.\tElectronics\ttrue\ttrue",
            "Absolutely love this product. Best purchase I've made this year.\tElectronics\ttrue\ttrue",
            "Excellent quality, fast shipping, and great customer service.\tElectronics\ttrue\ttrue",
            "Works perfectly as described. Very happy with my purchase.\tElectronics\tfalse\ttrue",
            "Amazing value for money. Would definitely recommend to friends.\tElectronics\ttrue\ttrue",
            "The sound quality is incredible. Better than expected!\tElectronics\ttrue\ttrue",
            "Perfect fit, great design, and very comfortable to use.\tElectronics\ttrue\ttrue",
            "Exceeded my expectations in every way. Five stars!\tElectronics\ttrue\ttrue",
            
            // Negative reviews
            "Terrible product. Broke after one week of use.\tElectronics\ttrue\tfalse",
            "Waste of money. Does not work as advertised.\tElectronics\ttrue\tfalse",
            "Very disappointed. Poor quality and slow performance.\tElectronics\tfalse\tfalse",
            "Worst purchase ever. Battery dies in two hours.\tElectronics\ttrue\tfalse",
            "Do not buy this. Complete garbage.\tElectronics\ttrue\tfalse",
            "Cheap materials, broke immediately. Returning it.\tElectronics\ttrue\tfalse",
            "Horrible experience. Customer service was no help.\tElectronics\tfalse\tfalse",
            "Not worth the price. Save your money.\tElectronics\ttrue\tfalse"
        };
        
        var header = "ReviewText\tProductCategory\tVerifiedPurchase\tSentiment";
        File.WriteAllLines(path, new[] { header }.Concat(samples));
        
        Console.WriteLine($"Created {samples.Count} sample reviews in {path}");
    }
}
```

[FIGURE: Architecture diagram showing the complete review analysis pipeline from data ingestion through preprocessing, model inference, and result aggregation]

## Summary

In this chapter, you've journeyed from raw text to production-ready NLP systems:

- **Text Preprocessing**: You learned to clean, normalize, and prepare text for machine learning—handling everything from Unicode to contractions to repeated characters.

- **Tokenization and Vectorization**: You explored how to convert text into numerical representations, from simple bag-of-words to TF-IDF weighting and n-grams that capture phrase-level patterns.

- **Sentiment Analysis**: You built a complete ML.NET pipeline for classifying text sentiment, including handling imbalanced data and interpreting model confidence.

- **Named Entity Recognition**: You implemented both rule-based and model-based approaches for extracting structured information from unstructured text.

- **Transformers**: You gained a conceptual understanding of the architecture revolutionizing NLP and learned how to use pre-trained transformer models in C# via ONNX Runtime.

- **Production Project**: You created a complete product review analysis system with preprocessing, training, inference, and aspect-level sentiment extraction.

The NLP techniques you've learned are immediately applicable to countless business problems: customer feedback analysis, support ticket routing, content moderation, document classification, and more. Your C# skills combine naturally with ML.NET's text processing capabilities to build systems that extract insights from the mountains of text data every organization generates.

In the next chapter, we'll explore time series analysis—another domain where your programming skills create real value.

---

## Exercises

1. **Custom Stop Words**: The default stop word list may not be ideal for your domain. Create a function that analyzes a corpus and identifies domain-specific stop words based on document frequency (words appearing in >80% of documents). Test it on a collection of product reviews and compare model performance with and without your custom stop words.

2. **Negation Handling**: Implement a more sophisticated negation handler that:
   - Detects negation phrases ("not," "never," "no longer," "hardly")
   - Identifies the scope of negation (which words it affects)
   - Modifies affected words (e.g., prefixing with "NOT_")
   
   Test your implementation on reviews containing phrases like "not bad," "never disappointed," and "no complaints."

3. **Multi-Class Sentiment**: Extend the sentiment analyzer to predict three classes (Positive, Neutral, Negative) instead of two. You'll need to:
   - Define what constitutes "neutral" (consider using 3-star reviews as neutral)
   - Modify the ML.NET pipeline to use multiclass classification
   - Evaluate using appropriate metrics (macro-averaged F1)
   
   How does adding the neutral class affect your model's performance on clear positive/negative cases?

4. **Aspect-Based Sentiment**: Enhance the `ExtractAspectSentiments` method to use a trained ML model instead of rule-based matching. You'll need to:
   - Create training data with labeled aspects and their sentiments
   - Train separate models for aspect detection and aspect sentiment
   - Handle reviews with multiple aspects of different sentiments
   
   Compare the accuracy of your ML-based approach to the rule-based implementation.

5. **Real-Time Streaming Analysis**: Build a service that analyzes reviews in real-time as they arrive:
   - Use `System.Threading.Channels` for the producer-consumer pattern
   - Implement batching for efficiency (process reviews in groups of 10-50)
   - Add a sliding window that computes sentiment trends over the last hour/day
   - Create alerts when negative sentiment exceeds a threshold
   
   Test your service with simulated review streams of varying volumes.

---

*Next Chapter: Time Series Analysis →*
