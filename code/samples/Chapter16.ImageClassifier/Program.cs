// Chapter 16: Computer Vision - Image Classification with ONNX
// This sample demonstrates image classification using ONNX Runtime in C#

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

// ============================================================================
// ONNX Model Sources
// ============================================================================
// Download pre-trained models from ONNX Model Zoo:
// https://github.com/onnx/models
//
// Popular image classification models:
// - ResNet-50:    https://github.com/onnx/models/tree/main/validated/vision/classification/resnet
// - MobileNet:    https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet
// - EfficientNet: https://github.com/onnx/models/tree/main/validated/vision/classification/efficientnet
// - SqueezeNet:   https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet
//
// Place the .onnx file in the project directory and update MODEL_PATH below.
// ============================================================================

const string MODEL_PATH = "resnet50-v2-7.onnx";
const int IMAGE_SIZE = 224;  // Standard ImageNet input size
const int NUM_CHANNELS = 3;  // RGB
const int NUM_CLASSES = 1000; // ImageNet classes

Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
Console.WriteLine("║     Chapter 16: Image Classification with ONNX Runtime       ║");
Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
Console.WriteLine();

// Check if model exists - run in demo mode if not
bool demoMode = !File.Exists(MODEL_PATH);

if (demoMode)
{
    Console.WriteLine("⚠️  No ONNX model found. Running in DEMO MODE.");
    Console.WriteLine($"   To use real inference, download a model to: {MODEL_PATH}");
    Console.WriteLine();
    RunDemoMode();
}
else
{
    await RunRealInferenceAsync(MODEL_PATH);
}

// ============================================================================
// DEMO MODE - Shows the complete pattern without requiring a real model
// ============================================================================
void RunDemoMode()
{
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("STEP 1: Image Preprocessing");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Create a synthetic test image (gradient pattern)
    using var image = new Image<Rgb24>(320, 240);
    image.Mutate(ctx => ctx
        .BackgroundColor(Color.SkyBlue)
        .Fill(Color.ForestGreen, new Rectangle(50, 100, 220, 100)));
    
    Console.WriteLine($"✓ Created test image: 320x240 RGB");
    
    // Demonstrate preprocessing pipeline
    var tensor = PreprocessImage(image);
    Console.WriteLine($"✓ Resized to: {IMAGE_SIZE}x{IMAGE_SIZE}");
    Console.WriteLine($"✓ Tensor shape: [1, {NUM_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}]");
    Console.WriteLine($"✓ Normalized with ImageNet mean/std");
    Console.WriteLine();
    
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("STEP 2: Model Loading (Simulated)");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("  // In real code:");
    Console.WriteLine("  using var session = new InferenceSession(MODEL_PATH);");
    Console.WriteLine("  var inputName = session.InputMetadata.Keys.First();");
    Console.WriteLine("  var outputName = session.OutputMetadata.Keys.First();");
    Console.WriteLine();
    
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("STEP 3: Running Inference (Simulated)");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Simulate model output (logits for 1000 ImageNet classes)
    var simulatedLogits = GenerateSimulatedLogits();
    Console.WriteLine($"✓ Model produced {NUM_CLASSES} class logits");
    Console.WriteLine();
    
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("STEP 4: Post-processing & Results");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    // Apply softmax and get top-K predictions
    var probabilities = Softmax(simulatedLogits);
    var topK = GetTopKPredictions(probabilities, k: 5);
    
    Console.WriteLine("🏆 Top-5 Predictions:");
    Console.WriteLine();
    
    foreach (var (classId, probability, rank) in topK)
    {
        string className = GetImageNetClassName(classId);
        string bar = new string('█', (int)(probability * 40));
        Console.WriteLine($"  {rank}. {className,-25} {probability:P2} {bar}");
    }
    
    Console.WriteLine();
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Console.WriteLine("✅ Demo complete! Download a real ONNX model to run actual inference.");
    Console.WriteLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

// ============================================================================
// REAL INFERENCE - Uses actual ONNX model
// ============================================================================
async Task RunRealInferenceAsync(string modelPath)
{
    Console.WriteLine($"📦 Loading model: {modelPath}");
    Console.WriteLine();
    
    // Configure session options
    var sessionOptions = new SessionOptions
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        ExecutionMode = ExecutionMode.ORT_PARALLEL
    };
    
    // Enable GPU acceleration if available (uncomment if CUDA is installed)
    // sessionOptions.AppendExecutionProvider_CUDA();
    
    using var session = new InferenceSession(modelPath, sessionOptions);
    
    // Print model metadata
    Console.WriteLine("Model Information:");
    Console.WriteLine($"  Inputs:  {string.Join(", ", session.InputMetadata.Keys)}");
    Console.WriteLine($"  Outputs: {string.Join(", ", session.OutputMetadata.Keys)}");
    Console.WriteLine();
    
    // Get input/output names
    var inputName = session.InputMetadata.Keys.First();
    var outputName = session.OutputMetadata.Keys.First();
    
    // Create or load a test image
    string testImagePath = "test_image.jpg";
    Image<Rgb24> image;
    
    if (File.Exists(testImagePath))
    {
        image = await Image.LoadAsync<Rgb24>(testImagePath);
        Console.WriteLine($"📷 Loaded image: {testImagePath}");
    }
    else
    {
        // Create a synthetic test image
        image = new Image<Rgb24>(640, 480);
        image.Mutate(ctx => ctx
            .BackgroundColor(Color.LightBlue)
            .Fill(Color.Orange, new Rectangle(200, 150, 240, 180)));
        Console.WriteLine("📷 Created synthetic test image");
    }
    
    Console.WriteLine($"   Original size: {image.Width}x{image.Height}");
    
    // Preprocess the image
    var inputTensor = PreprocessImage(image);
    Console.WriteLine($"   Preprocessed tensor: [1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}]");
    Console.WriteLine();
    
    // Run inference
    Console.WriteLine("🔄 Running inference...");
    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
    
    var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
    };
    
    using var results = session.Run(inputs);
    stopwatch.Stop();
    
    Console.WriteLine($"⏱️  Inference time: {stopwatch.ElapsedMilliseconds}ms");
    Console.WriteLine();
    
    // Process results
    var outputTensor = results.First().AsTensor<float>();
    var logits = outputTensor.ToArray();
    
    // Apply softmax to get probabilities
    var probabilities = Softmax(logits);
    
    // Get top-K predictions
    var topK = GetTopKPredictions(probabilities, k: 5);
    
    Console.WriteLine("🏆 Top-5 Predictions:");
    Console.WriteLine();
    
    foreach (var (classId, probability, rank) in topK)
    {
        string className = GetImageNetClassName(classId);
        string bar = new string('█', (int)(probability * 40));
        Console.WriteLine($"  {rank}. {className,-25} {probability:P2} {bar}");
    }
    
    image.Dispose();
}

// ============================================================================
// IMAGE PREPROCESSING
// ============================================================================
DenseTensor<float> PreprocessImage(Image<Rgb24> image)
{
    // Resize image to model's expected input size
    image.Mutate(ctx => ctx.Resize(new ResizeOptions
    {
        Size = new Size(IMAGE_SIZE, IMAGE_SIZE),
        Mode = ResizeMode.Crop,  // Center crop to maintain aspect ratio
        Position = AnchorPositionMode.Center
    }));
    
    // Create tensor with shape [batch, channels, height, width] (NCHW format)
    var tensor = new DenseTensor<float>([1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE]);
    
    // ImageNet normalization parameters
    float[] mean = [0.485f, 0.456f, 0.406f];  // RGB means
    float[] std = [0.229f, 0.224f, 0.225f];   // RGB standard deviations
    
    // Process each pixel
    for (int y = 0; y < IMAGE_SIZE; y++)
    {
        for (int x = 0; x < IMAGE_SIZE; x++)
        {
            var pixel = image[x, y];
            
            // Normalize: (pixel / 255.0 - mean) / std
            tensor[0, 0, y, x] = ((pixel.R / 255f) - mean[0]) / std[0];  // Red channel
            tensor[0, 1, y, x] = ((pixel.G / 255f) - mean[1]) / std[1];  // Green channel
            tensor[0, 2, y, x] = ((pixel.B / 255f) - mean[2]) / std[2];  // Blue channel
        }
    }
    
    return tensor;
}

// ============================================================================
// POST-PROCESSING FUNCTIONS
// ============================================================================

/// <summary>
/// Apply softmax to convert logits to probabilities
/// </summary>
float[] Softmax(float[] logits)
{
    // Find max for numerical stability
    float max = logits.Max();
    
    // Compute exp(x - max) for each element
    var exps = logits.Select(x => MathF.Exp(x - max)).ToArray();
    
    // Normalize by sum
    float sum = exps.Sum();
    return exps.Select(x => x / sum).ToArray();
}

/// <summary>
/// Get top-K predictions sorted by probability
/// </summary>
IEnumerable<(int ClassId, float Probability, int Rank)> GetTopKPredictions(float[] probabilities, int k)
{
    return probabilities
        .Select((prob, idx) => (ClassId: idx, Probability: prob))
        .OrderByDescending(x => x.Probability)
        .Take(k)
        .Select((x, rank) => (x.ClassId, x.Probability, Rank: rank + 1));
}

/// <summary>
/// Generate simulated logits for demo mode
/// </summary>
float[] GenerateSimulatedLogits()
{
    var random = new Random(42);  // Fixed seed for reproducibility
    var logits = new float[NUM_CLASSES];
    
    // Generate random logits with some "peaks" for realistic-looking results
    for (int i = 0; i < NUM_CLASSES; i++)
    {
        logits[i] = (float)(random.NextDouble() * 2 - 1);  // Range [-1, 1]
    }
    
    // Add some higher values to simulate confident predictions
    // These class IDs correspond to nature-related ImageNet classes
    logits[970] = 5.2f;   // alp (mountain)
    logits[975] = 4.8f;   // lakeside
    logits[979] = 3.9f;   // valley
    logits[973] = 3.5f;   // coral reef
    logits[971] = 3.2f;   // cliff
    
    return logits;
}

/// <summary>
/// Get human-readable class name from ImageNet class ID
/// Note: In production, load the full labels file from:
/// https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
/// </summary>
string GetImageNetClassName(int classId)
{
    // Subset of ImageNet class names for demo purposes
    var classNames = new Dictionary<int, string>
    {
        // Animals
        { 0, "tench (fish)" },
        { 1, "goldfish" },
        { 7, "cock (rooster)" },
        { 22, "bald eagle" },
        { 76, "tarantula" },
        { 101, "tusker" },
        { 207, "golden retriever" },
        { 281, "tabby cat" },
        { 285, "Egyptian cat" },
        { 291, "lion" },
        { 340, "zebra" },
        { 386, "African elephant" },
        { 388, "giant panda" },
        
        // Objects
        { 409, "analog clock" },
        { 417, "balloon" },
        { 457, "bow tie" },
        { 508, "computer keyboard" },
        { 531, "digital watch" },
        { 610, "iPod" },
        { 620, "laptop" },
        { 665, "monitor" },
        { 673, "mouse (computer)" },
        { 703, "park bench" },
        { 751, "racket" },
        { 817, "sports car" },
        { 852, "tennis ball" },
        
        // Nature/Scenery
        { 970, "alp (mountain)" },
        { 971, "cliff" },
        { 973, "coral reef" },
        { 975, "lakeside" },
        { 979, "valley" },
        { 980, "volcano" },
        
        // Food
        { 924, "guacamole" },
        { 927, "ice cream" },
        { 950, "orange (fruit)" },
        { 954, "banana" },
        { 959, "broccoli" },
        { 963, "pizza" },
        { 965, "burrito" },
    };
    
    return classNames.GetValueOrDefault(classId, $"class_{classId}");
}
