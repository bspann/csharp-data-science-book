# Chapter 16: Computer Vision Basics

You've spent this book mastering tabular data—rows and columns, features and labels, the bread and butter of enterprise machine learning. But the world doesn't come in spreadsheets. The world comes in images: product photos that need classification, documents that need scanning, security feeds that need monitoring, medical scans that need analysis.

Computer vision is where machine learning becomes visceral. You're not predicting a number or a category from numerical features—you're teaching a computer to *see*. And here's what might surprise you: it's not as intimidating as it sounds, especially in 2026.

The heavy lifting of computer vision has been done. Researchers have spent decades and billions of GPU-hours training models that understand images at a level that rivals (and sometimes exceeds) human perception. Your job isn't to replicate that work. Your job is to leverage it—to take these pre-trained powerhouses and deploy them in production C# applications.

In this chapter, we'll cover the fundamentals of image processing in .NET, explore how to use ONNX models for inference, understand the magic of transfer learning, and build a complete image classifier using a pre-trained ResNet model. By the end, you'll have a production-ready computer vision pipeline running entirely in C#.

## Understanding Images as Data

Before we process images, we need to understand what they actually are from a data perspective. As a C# developer, you're comfortable with strong types and structured data. Images are just another data structure—a particularly dense one.

### The Anatomy of a Digital Image

A digital image is a multi-dimensional array of numbers. For a color image, you typically have:

- **Width**: Number of pixels horizontally
- **Height**: Number of pixels vertically
- **Channels**: Usually 3 (Red, Green, Blue) or 4 (RGBA with alpha/transparency)

A 224×224 RGB image—the standard input size for many neural networks—contains 224 × 224 × 3 = 150,528 individual values. Each value typically ranges from 0 to 255 (for 8-bit images) representing intensity.

[FIGURE: Diagram showing a 3D representation of an image tensor with height, width, and RGB channel dimensions labeled]

```csharp
// Conceptually, an image is a 3D tensor
// Dimensions: [Height, Width, Channels]
float[,,] image = new float[224, 224, 3];

// Accessing the red channel value at position (10, 20)
float redValue = image[10, 20, 0];  // Channel 0 = Red
float greenValue = image[10, 20, 1]; // Channel 1 = Green
float blueValue = image[10, 20, 2];  // Channel 2 = Blue
```

### Channel Ordering: A Source of Bugs

Here's something that will save you hours of debugging: different libraries use different channel orderings.

| Format | Dimension Order | Common Users |
|--------|-----------------|--------------|
| **HWC** | Height, Width, Channels | TensorFlow, most image libraries |
| **CHW** | Channels, Height, Width | PyTorch, ONNX (typically) |
| **NHWC** | Batch, Height, Width, Channels | TensorFlow default |
| **NCHW** | Batch, Channels, Height, Width | PyTorch, ONNX default |

When your model produces garbage outputs, the first thing to check is whether you've got the dimensions in the order the model expects. ONNX models typically expect **NCHW** format—batch size first, then channels, then height, then width.

```csharp
// Converting from HWC (common in image loading) to NCHW (ONNX expected)
// Original shape: [224, 224, 3] (HWC)
// Target shape: [1, 3, 224, 224] (NCHW - batch of 1)

float[] ConvertHwcToNchw(float[,,] hwcImage)
{
    int height = hwcImage.GetLength(0);
    int width = hwcImage.GetLength(1);
    int channels = hwcImage.GetLength(2);
    
    float[] nchw = new float[1 * channels * height * width];
    
    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int nchwIndex = c * height * width + h * width + w;
                nchw[nchwIndex] = hwcImage[h, w, c];
            }
        }
    }
    
    return nchw;
}
```

## Image Preprocessing Fundamentals

Raw images rarely go directly into models. They need preprocessing—a series of transformations that prepare the image data for the neural network. This isn't optional; it's essential for model accuracy.

### Resizing: Getting Dimensions Right

Neural networks have fixed input dimensions. ResNet expects 224×224. Some models expect 299×299 or 384×384. Your input images can be any size, so resizing is typically the first step.

```csharp
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

public static Bitmap ResizeImage(Image image, int width, int height)
{
    var destRect = new Rectangle(0, 0, width, height);
    var destImage = new Bitmap(width, height);

    destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

    using (var graphics = Graphics.FromImage(destImage))
    {
        graphics.CompositingMode = CompositingMode.SourceCopy;
        graphics.CompositingQuality = CompositingQuality.HighQuality;
        graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
        graphics.SmoothingMode = SmoothingMode.HighQuality;
        graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

        using (var wrapMode = new ImageAttributes())
        {
            wrapMode.SetWrapMode(WrapMode.TileFlipXY);
            graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, 
                GraphicsUnit.Pixel, wrapMode);
        }
    }

    return destImage;
}
```

For cross-platform scenarios, consider using **ImageSharp** instead of `System.Drawing`:

```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

public static Image ResizeImageCrossPlatform(string inputPath, int width, int height)
{
    using var image = Image.Load(inputPath);
    image.Mutate(x => x.Resize(width, height));
    return image.Clone();
}
```

**Aspect ratio considerations:** Simply squashing an image to fit can distort features. Common strategies include:

1. **Stretch**: Resize directly (fast, may distort)
2. **Pad**: Resize while maintaining aspect ratio, pad remaining space
3. **Crop**: Resize while maintaining aspect ratio, crop excess
4. **Center crop**: Take the center region after resizing the shorter dimension

[FIGURE: Visual comparison of stretch vs. pad vs. center crop resizing strategies on a sample image]

```csharp
public static Image<Rgb24> ResizeWithPadding(Image<Rgb24> image, int targetSize)
{
    // Calculate scaling to fit within target while maintaining aspect ratio
    float scale = Math.Min(
        (float)targetSize / image.Width,
        (float)targetSize / image.Height
    );
    
    int newWidth = (int)(image.Width * scale);
    int newHeight = (int)(image.Height * scale);
    
    // Create padded canvas
    var result = new Image<Rgb24>(targetSize, targetSize, new Rgb24(0, 0, 0));
    
    // Resize original
    image.Mutate(x => x.Resize(newWidth, newHeight));
    
    // Center on canvas
    int offsetX = (targetSize - newWidth) / 2;
    int offsetY = (targetSize - newHeight) / 2;
    
    result.Mutate(x => x.DrawImage(image, new Point(offsetX, offsetY), 1f));
    
    return result;
}
```

### Normalization: Scaling Pixel Values

Raw pixel values range from 0-255. Neural networks typically expect values in different ranges, depending on how they were trained. Common normalizations include:

| Normalization | Formula | Use Case |
|--------------|---------|----------|
| **[0, 1]** | `pixel / 255.0` | General purpose |
| **[-1, 1]** | `(pixel / 127.5) - 1` | Some TensorFlow models |
| **ImageNet mean/std** | `(pixel - mean) / std` | ResNet, VGG, etc. |

ImageNet normalization is the most common for pre-trained models. The magic numbers are:

```csharp
// ImageNet mean and standard deviation (per channel, RGB)
// These come from computing statistics across the ImageNet training set
public static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
public static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

public static float[] NormalizeImageNetStyle(Bitmap image)
{
    int width = image.Width;
    int height = image.Height;
    float[] normalized = new float[3 * height * width];
    
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Color pixel = image.GetPixel(x, y);
            
            // Scale to [0, 1]
            float r = pixel.R / 255f;
            float g = pixel.G / 255f;
            float b = pixel.B / 255f;
            
            // Apply ImageNet normalization
            r = (r - ImageNetMean[0]) / ImageNetStd[0];
            g = (g - ImageNetMean[1]) / ImageNetStd[1];
            b = (b - ImageNetMean[2]) / ImageNetStd[2];
            
            // Store in NCHW format
            int pixelIndex = y * width + x;
            normalized[0 * height * width + pixelIndex] = r; // Red channel
            normalized[1 * height * width + pixelIndex] = g; // Green channel
            normalized[2 * height * width + pixelIndex] = b; // Blue channel
        }
    }
    
    return normalized;
}
```

**Why these specific numbers?** The mean and standard deviation were computed across the entire ImageNet dataset (over 1 million images). Normalizing your input images the same way aligns them with what the model saw during training. Using wrong normalization is one of the most common causes of poor model performance.

### Data Augmentation

While preprocessing is for inference, augmentation is for training. The idea: artificially expand your training set by creating variations of your images. A single photo of a cat becomes hundreds of training examples through random transformations.

Common augmentations include:

- **Horizontal flip**: Mirror the image left-to-right
- **Random rotation**: Rotate by a few degrees
- **Color jitter**: Randomly adjust brightness, contrast, saturation
- **Random crop**: Take a random sub-region
- **Gaussian noise**: Add small random noise to pixels

```csharp
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

public class ImageAugmenter
{
    private readonly Random _random = new Random(42);
    
    public Image<Rgb24> Augment(Image<Rgb24> image)
    {
        var augmented = image.Clone();
        
        // Random horizontal flip (50% chance)
        if (_random.NextDouble() > 0.5)
        {
            augmented.Mutate(x => x.Flip(FlipMode.Horizontal));
        }
        
        // Random rotation (-15 to +15 degrees)
        float rotation = (float)(_random.NextDouble() * 30 - 15);
        augmented.Mutate(x => x.Rotate(rotation));
        
        // Random brightness adjustment (0.8 to 1.2)
        float brightness = (float)(_random.NextDouble() * 0.4 + 0.8);
        augmented.Mutate(x => x.Brightness(brightness));
        
        // Random contrast adjustment (0.9 to 1.1)
        float contrast = (float)(_random.NextDouble() * 0.2 + 0.9);
        augmented.Mutate(x => x.Contrast(contrast));
        
        return augmented;
    }
}
```

**A word of caution**: Augmentation must respect the semantics of your problem. Horizontal flipping is fine for cats and dogs but dangerous for text recognition (a 'd' becomes a 'b'). Vertical flipping might work for satellite imagery but not for photos of people. Think about what transformations preserve the meaning of your data.

[FIGURE: Grid showing original image alongside various augmented versions: flipped, rotated, brightness-adjusted, and cropped variations]

## Transfer Learning: Standing on Giants' Shoulders

Here's the single most important concept in practical computer vision: **you almost never train a model from scratch**.

Training a state-of-the-art image classifier from random weights requires:
- Millions of labeled images
- Thousands of GPU-hours
- Expertise in neural architecture design
- Weeks or months of experimentation

Transfer learning sidesteps all of this. You take a model that someone else trained on a massive dataset (usually ImageNet, with 14 million images across 1,000 categories) and adapt it to your specific task. The model has already learned to understand images—edges, textures, shapes, objects. You just redirect that understanding to your problem.

### Why Transfer Learning Works

Neural networks learn hierarchically. The early layers learn low-level features:
- Edge detection
- Color gradients
- Simple textures

Middle layers combine these into higher-level patterns:
- Shapes
- Parts of objects
- Complex textures

Final layers learn task-specific features:
- "This combination of features means 'golden retriever'"
- "This pattern indicates 'sports car'"

[FIGURE: Visualization of convolutional neural network layer activations, showing progression from edges to textures to object parts to full objects]

Here's the key insight: **the early and middle layers are reusable**. What a network learns about detecting edges from ImageNet applies equally well to medical images, satellite photos, or product thumbnails. Only the final, task-specific layers need to change.

### Transfer Learning Strategies

**Feature extraction** (simplest): Freeze the pre-trained model and use it purely as a feature extractor. Add a new classifier layer on top that learns your specific categories.

```
[Pre-trained ResNet - Frozen] → [New Dense Layer] → [Your Categories]
```

**Fine-tuning** (more powerful): Start with pre-trained weights, but allow training to update them. Usually you freeze most layers and only fine-tune the final few, or use a lower learning rate for pre-trained layers.

```
[Pre-trained ResNet - Partially Trainable] → [New Dense Layer] → [Your Categories]
```

**When to use which:**
- **Feature extraction**: Small dataset (<1,000 images), limited compute, quick iteration
- **Fine-tuning**: Larger dataset, need maximum accuracy, willing to invest training time

In C# with ML.NET, we primarily use feature extraction because ML.NET doesn't support GPU training for deep neural networks. We leverage ONNX for inference and use the transfer learning capabilities that ML.NET provides for image classification.

### The ONNX Bridge

ONNX (Open Neural Network Exchange) is the key that unlocks the entire deep learning ecosystem for C# developers. It's an open format that allows models trained in any framework—PyTorch, TensorFlow, scikit-learn—to run anywhere ONNX Runtime is available.

The workflow:
1. Researchers train state-of-the-art models in Python
2. Models are exported to ONNX format
3. You load and run them in C# with ONNX Runtime
4. Everyone wins

[FIGURE: Diagram showing model flow from PyTorch/TensorFlow training through ONNX export to ONNX Runtime inference in C#]

## Using ONNX Models in ML.NET and ONNX Runtime

Let's get practical. We'll work with ONNX models in two ways: through ML.NET's integration and directly with ONNX Runtime. Both approaches have their place.

### Setting Up Dependencies

```bash
dotnet add package Microsoft.ML --version 4.0.0
dotnet add package Microsoft.ML.OnnxRuntime --version 1.17.0
dotnet add package Microsoft.ML.OnnxTransformer --version 4.0.0
dotnet add package SixLabors.ImageSharp --version 3.1.0
```

### Direct ONNX Runtime Inference

ONNX Runtime gives you the most control. Here's a complete example:

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public class OnnxImageClassifier : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string[] _labels;
    private readonly int _inputWidth;
    private readonly int _inputHeight;
    
    // ImageNet normalization constants
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };
    
    public OnnxImageClassifier(string modelPath, string labelsPath, 
        int inputWidth = 224, int inputHeight = 224)
    {
        // Configure session options
        var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        
        _session = new InferenceSession(modelPath, sessionOptions);
        _labels = File.ReadAllLines(labelsPath);
        _inputWidth = inputWidth;
        _inputHeight = inputHeight;
        
        // Log model information
        Console.WriteLine($"Model loaded: {modelPath}");
        Console.WriteLine($"Input: {_session.InputMetadata.First().Key}");
        Console.WriteLine($"Output: {_session.OutputMetadata.First().Key}");
    }
    
    public (string Label, float Confidence)[] Classify(string imagePath, int topK = 5)
    {
        // Load and preprocess image
        float[] inputTensor = PreprocessImage(imagePath);
        
        // Create input tensor with NCHW shape [1, 3, 224, 224]
        var tensor = new DenseTensor<float>(
            inputTensor, 
            new[] { 1, 3, _inputHeight, _inputWidth }
        );
        
        // Prepare inputs
        var inputName = _session.InputMetadata.First().Key;
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, tensor)
        };
        
        // Run inference
        using var results = _session.Run(inputs);
        
        // Process outputs
        var output = results.First().AsTensor<float>();
        var probabilities = Softmax(output.ToArray());
        
        // Get top-K predictions
        return probabilities
            .Select((prob, index) => (Label: _labels[index], Confidence: prob))
            .OrderByDescending(x => x.Confidence)
            .Take(topK)
            .ToArray();
    }
    
    private float[] PreprocessImage(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        
        // Resize to expected dimensions
        image.Mutate(x => x.Resize(_inputWidth, _inputHeight));
        
        // Convert to float array with normalization, NCHW format
        float[] result = new float[3 * _inputHeight * _inputWidth];
        
        for (int y = 0; y < _inputHeight; y++)
        {
            for (int x = 0; x < _inputWidth; x++)
            {
                var pixel = image[x, y];
                
                // Normalize to [0,1] then apply ImageNet normalization
                float r = ((pixel.R / 255f) - Mean[0]) / Std[0];
                float g = ((pixel.G / 255f) - Mean[1]) / Std[1];
                float b = ((pixel.B / 255f) - Mean[2]) / Std[2];
                
                // NCHW layout: all reds, then all greens, then all blues
                int pixelIndex = y * _inputWidth + x;
                result[0 * _inputHeight * _inputWidth + pixelIndex] = r;
                result[1 * _inputHeight * _inputWidth + pixelIndex] = g;
                result[2 * _inputHeight * _inputWidth + pixelIndex] = b;
            }
        }
        
        return result;
    }
    
    private static float[] Softmax(float[] logits)
    {
        float maxLogit = logits.Max();
        float[] exps = logits.Select(l => MathF.Exp(l - maxLogit)).ToArray();
        float sumExps = exps.Sum();
        return exps.Select(e => e / sumExps).ToArray();
    }
    
    public void Dispose()
    {
        _session?.Dispose();
    }
}
```

Usage:

```csharp
using var classifier = new OnnxImageClassifier(
    "models/resnet50.onnx",
    "models/imagenet_labels.txt"
);

var predictions = classifier.Classify("images/cat.jpg");

foreach (var (label, confidence) in predictions)
{
    Console.WriteLine($"{label}: {confidence:P1}");
}

// Output:
// Egyptian cat: 87.3%
// tabby: 8.2%
// tiger cat: 2.1%
// Persian cat: 1.4%
// Siamese cat: 0.6%
```

### ML.NET ONNX Integration

ML.NET provides a higher-level API through its `OnnxTransformer`:

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath { get; set; }
    
    [LoadColumn(1)]
    public string Label { get; set; }
}

public class ImagePrediction
{
    [ColumnName("output")]
    public float[] PredictedLabels { get; set; }
}

public class MlNetOnnxClassifier
{
    private readonly MLContext _mlContext;
    private readonly ITransformer _model;
    private readonly PredictionEngine<ImageData, ImagePrediction> _predictionEngine;
    
    public MlNetOnnxClassifier(string modelPath)
    {
        _mlContext = new MLContext();
        
        var pipeline = _mlContext.Transforms.LoadImages(
                outputColumnName: "image",
                imageFolder: "",
                inputColumnName: nameof(ImageData.ImagePath))
            .Append(_mlContext.Transforms.ResizeImages(
                outputColumnName: "image",
                imageWidth: 224,
                imageHeight: 224,
                inputColumnName: "image"))
            .Append(_mlContext.Transforms.ExtractPixels(
                outputColumnName: "input",
                inputColumnName: "image",
                interleavePixelColors: false, // NCHW format
                offsetImage: 0,
                scaleImage: 1f / 255f))
            .Append(_mlContext.Transforms.ApplyOnnxModel(
                modelFile: modelPath,
                outputColumnNames: new[] { "output" },
                inputColumnNames: new[] { "input" }));
        
        // Fit on empty data to create transformer
        var emptyData = _mlContext.Data.LoadFromEnumerable(new List<ImageData>());
        _model = pipeline.Fit(emptyData);
        
        _predictionEngine = _mlContext.Model
            .CreatePredictionEngine<ImageData, ImagePrediction>(_model);
    }
    
    public float[] Predict(string imagePath)
    {
        var imageData = new ImageData { ImagePath = imagePath };
        var prediction = _predictionEngine.Predict(imageData);
        return prediction.PredictedLabels;
    }
}
```

### GPU Acceleration

ONNX Runtime supports GPU acceleration, which can provide 10-100x speedups for inference. To enable it:

```bash
# Install GPU-enabled package instead
dotnet add package Microsoft.ML.OnnxRuntime.Gpu --version 1.17.0
```

```csharp
var sessionOptions = new SessionOptions();

// Add CUDA execution provider (NVIDIA GPUs)
sessionOptions.AppendExecutionProvider_CUDA();

// Or DirectML for Windows (AMD and NVIDIA)
// sessionOptions.AppendExecutionProvider_DML();

var session = new InferenceSession(modelPath, sessionOptions);
```

**GPU considerations:**
- Requires appropriate drivers (CUDA for NVIDIA, DirectML for Windows GPU)
- GPU memory limits your batch size
- For single-image inference, CPU might be faster due to GPU transfer overhead
- For batch processing, GPU wins decisively

## Object Detection Concepts

Classification asks "What is in this image?" Object detection asks "What is in this image, and where?"

[FIGURE: Comparison of classification (single label), object detection (multiple bounding boxes with labels), and instance segmentation (pixel-level masks) on the same street scene image]

### Bounding Boxes

An object detection model outputs:
- **Class**: What type of object (person, car, dog)
- **Confidence**: How certain the model is (0.0 to 1.0)
- **Bounding box**: Where the object is (x, y, width, height)

```csharp
public class DetectedObject
{
    public string Label { get; set; }
    public float Confidence { get; set; }
    public float X { get; set; }  // Left edge (0-1 normalized)
    public float Y { get; set; }  // Top edge (0-1 normalized)
    public float Width { get; set; }
    public float Height { get; set; }
    
    public Rectangle ToAbsoluteRectangle(int imageWidth, int imageHeight)
    {
        return new Rectangle(
            (int)(X * imageWidth),
            (int)(Y * imageHeight),
            (int)(Width * imageWidth),
            (int)(Height * imageHeight)
        );
    }
}
```

### Popular Object Detection Models

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| **YOLO v8** | Very fast | Good | Real-time detection, edge devices |
| **SSD** | Fast | Good | Mobile/embedded |
| **Faster R-CNN** | Slow | Excellent | When accuracy matters more than speed |
| **EfficientDet** | Medium | Excellent | Good balance |

For C# applications, YOLO models are particularly popular because they're fast and available in ONNX format.

### Non-Maximum Suppression (NMS)

Object detectors often produce multiple overlapping bounding boxes for the same object. Non-maximum suppression filters these down to the best predictions:

```csharp
public static List<DetectedObject> ApplyNms(
    List<DetectedObject> detections, 
    float iouThreshold = 0.45f,
    float confidenceThreshold = 0.25f)
{
    // Filter by confidence
    var filtered = detections
        .Where(d => d.Confidence >= confidenceThreshold)
        .OrderByDescending(d => d.Confidence)
        .ToList();
    
    var result = new List<DetectedObject>();
    
    while (filtered.Count > 0)
    {
        // Take highest confidence detection
        var best = filtered[0];
        result.Add(best);
        filtered.RemoveAt(0);
        
        // Remove overlapping detections of the same class
        filtered.RemoveAll(d => 
            d.Label == best.Label && 
            CalculateIoU(best, d) > iouThreshold);
    }
    
    return result;
}

private static float CalculateIoU(DetectedObject a, DetectedObject b)
{
    // Calculate intersection
    float x1 = Math.Max(a.X, b.X);
    float y1 = Math.Max(a.Y, b.Y);
    float x2 = Math.Min(a.X + a.Width, b.X + b.Width);
    float y2 = Math.Min(a.Y + a.Height, b.Y + b.Height);
    
    float intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
    
    // Calculate union
    float areaA = a.Width * a.Height;
    float areaB = b.Width * b.Height;
    float union = areaA + areaB - intersection;
    
    return intersection / union;
}
```

[FIGURE: Before and after NMS visualization showing multiple overlapping boxes being reduced to single best detections]

## Project: Building an Image Classifier with Pre-trained ResNet

Let's build a production-ready image classification system. We'll classify product images into categories—a common e-commerce use case.

### Project Structure

```
ProductClassifier/
├── ProductClassifier.csproj
├── Program.cs
├── Services/
│   ├── ImagePreprocessor.cs
│   ├── ClassificationService.cs
│   └── ImageClassificationResult.cs
├── Models/
│   └── (resnet50.onnx goes here)
├── Data/
│   └── labels.txt
└── TestImages/
    └── (sample images for testing)
```

### Step 1: Create the Project

```bash
dotnet new console -n ProductClassifier
cd ProductClassifier

dotnet add package Microsoft.ML.OnnxRuntime --version 1.17.0
dotnet add package SixLabors.ImageSharp --version 3.1.0
dotnet add package System.Numerics.Tensors --version 8.0.0
```

### Step 2: Download the Model

We'll use ResNet-50, a classic architecture that's fast and accurate. Download from the ONNX Model Zoo:

```bash
mkdir Models
curl -L -o Models/resnet50.onnx \
    "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx"
```

Create `Data/labels.txt` with ImageNet class labels (available from the ONNX Model Zoo or ImageNet website).

### Step 3: Image Preprocessor

```csharp
// Services/ImagePreprocessor.cs
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ProductClassifier.Services;

public class ImagePreprocessor
{
    private readonly int _targetWidth;
    private readonly int _targetHeight;
    
    // ResNet-50 v2 uses simpler normalization
    // (Different from v1 which uses ImageNet mean/std)
    private const float Scale = 1f / 255f;
    
    public ImagePreprocessor(int targetWidth = 224, int targetHeight = 224)
    {
        _targetWidth = targetWidth;
        _targetHeight = targetHeight;
    }
    
    public float[] PreprocessImage(string imagePath)
    {
        using var image = Image.Load<Rgb24>(imagePath);
        return PreprocessImage(image);
    }
    
    public float[] PreprocessImage(Stream imageStream)
    {
        using var image = Image.Load<Rgb24>(imageStream);
        return PreprocessImage(image);
    }
    
    public float[] PreprocessImage(Image<Rgb24> image)
    {
        // Resize with center crop to maintain aspect ratio
        var resized = ResizeWithCenterCrop(image);
        
        // Convert to tensor in NCHW format
        var tensor = new float[1 * 3 * _targetHeight * _targetWidth];
        
        for (int y = 0; y < _targetHeight; y++)
        {
            for (int x = 0; x < _targetWidth; x++)
            {
                var pixel = resized[x, y];
                
                // Normalize to [0, 1]
                int pixelIndex = y * _targetWidth + x;
                tensor[0 * _targetHeight * _targetWidth + pixelIndex] = pixel.R * Scale;
                tensor[1 * _targetHeight * _targetWidth + pixelIndex] = pixel.G * Scale;
                tensor[2 * _targetHeight * _targetWidth + pixelIndex] = pixel.B * Scale;
            }
        }
        
        resized.Dispose();
        return tensor;
    }
    
    private Image<Rgb24> ResizeWithCenterCrop(Image<Rgb24> image)
    {
        // Calculate resize dimensions (resize shorter side to target, then crop)
        float scale = Math.Max(
            (float)_targetWidth / image.Width,
            (float)_targetHeight / image.Height
        );
        
        int resizedWidth = (int)(image.Width * scale);
        int resizedHeight = (int)(image.Height * scale);
        
        // Resize
        var resized = image.Clone();
        resized.Mutate(x => x.Resize(resizedWidth, resizedHeight));
        
        // Center crop
        int cropX = (resizedWidth - _targetWidth) / 2;
        int cropY = (resizedHeight - _targetHeight) / 2;
        resized.Mutate(x => x.Crop(new Rectangle(cropX, cropY, _targetWidth, _targetHeight)));
        
        return resized;
    }
}
```

### Step 4: Classification Service

```csharp
// Services/ImageClassificationResult.cs
namespace ProductClassifier.Services;

public record ClassificationResult(
    string Label,
    float Confidence,
    int ClassIndex
);

public record ClassificationResponse(
    string ImagePath,
    ClassificationResult[] TopPredictions,
    TimeSpan InferenceTime
);
```

```csharp
// Services/ClassificationService.cs
using System.Diagnostics;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ProductClassifier.Services;

public class ClassificationService : IDisposable
{
    private readonly InferenceSession _session;
    private readonly ImagePreprocessor _preprocessor;
    private readonly string[] _labels;
    private readonly string _inputName;
    private readonly string _outputName;
    
    public ClassificationService(string modelPath, string labelsPath)
    {
        // Configure ONNX Runtime
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            EnableMemoryPattern = true,
            EnableCpuMemArena = true
        };
        
        // Uncomment for GPU acceleration:
        // options.AppendExecutionProvider_CUDA();
        
        _session = new InferenceSession(modelPath, options);
        _preprocessor = new ImagePreprocessor(224, 224);
        _labels = File.ReadAllLines(labelsPath);
        
        // Get input/output names from model metadata
        _inputName = _session.InputMetadata.First().Key;
        _outputName = _session.OutputMetadata.First().Key;
        
        ValidateModel();
    }
    
    private void ValidateModel()
    {
        var inputMeta = _session.InputMetadata[_inputName];
        var expectedShape = new[] { 1, 3, 224, 224 };
        
        Console.WriteLine($"Model Input: {_inputName}");
        Console.WriteLine($"  Shape: [{string.Join(", ", inputMeta.Dimensions)}]");
        Console.WriteLine($"  Type: {inputMeta.ElementType}");
        Console.WriteLine($"Model Output: {_outputName}");
        Console.WriteLine($"Labels loaded: {_labels.Length}");
    }
    
    public ClassificationResponse Classify(string imagePath, int topK = 5)
    {
        var sw = Stopwatch.StartNew();
        
        // Preprocess
        var tensorData = _preprocessor.PreprocessImage(imagePath);
        var inputTensor = new DenseTensor<float>(tensorData, new[] { 1, 3, 224, 224 });
        
        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };
        
        using var results = _session.Run(inputs);
        var output = results.First().AsTensor<float>();
        
        sw.Stop();
        
        // Apply softmax and get top-K
        var probabilities = Softmax(output.ToArray());
        var topPredictions = probabilities
            .Select((prob, idx) => new ClassificationResult(
                Label: idx < _labels.Length ? _labels[idx] : $"class_{idx}",
                Confidence: prob,
                ClassIndex: idx
            ))
            .OrderByDescending(r => r.Confidence)
            .Take(topK)
            .ToArray();
        
        return new ClassificationResponse(
            ImagePath: imagePath,
            TopPredictions: topPredictions,
            InferenceTime: sw.Elapsed
        );
    }
    
    public async Task<ClassificationResponse[]> ClassifyBatchAsync(
        string[] imagePaths, 
        int topK = 5)
    {
        // Process images in parallel for I/O
        var preprocessTasks = imagePaths.Select(async path =>
        {
            return await Task.Run(() => _preprocessor.PreprocessImage(path));
        });
        
        var tensors = await Task.WhenAll(preprocessTasks);
        
        // Batch inference
        var results = new List<ClassificationResponse>();
        var sw = Stopwatch.StartNew();
        
        foreach (var (tensorData, index) in tensors.Select((t, i) => (t, i)))
        {
            var inputTensor = new DenseTensor<float>(tensorData, new[] { 1, 3, 224, 224 });
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };
            
            using var output = _session.Run(inputs);
            var probs = Softmax(output.First().AsTensor<float>().ToArray());
            
            var topPredictions = probs
                .Select((prob, idx) => new ClassificationResult(
                    Label: idx < _labels.Length ? _labels[idx] : $"class_{idx}",
                    Confidence: prob,
                    ClassIndex: idx
                ))
                .OrderByDescending(r => r.Confidence)
                .Take(topK)
                .ToArray();
            
            results.Add(new ClassificationResponse(
                ImagePath: imagePaths[index],
                TopPredictions: topPredictions,
                InferenceTime: sw.Elapsed
            ));
        }
        
        return results.ToArray();
    }
    
    private static float[] Softmax(float[] logits)
    {
        float maxLogit = logits.Max();
        var exps = logits.Select(l => MathF.Exp(l - maxLogit)).ToArray();
        float sumExps = exps.Sum();
        return exps.Select(e => e / sumExps).ToArray();
    }
    
    public void Dispose()
    {
        _session?.Dispose();
    }
}
```

### Step 5: Main Program

```csharp
// Program.cs
using ProductClassifier.Services;

// Configuration
string modelPath = "Models/resnet50.onnx";
string labelsPath = "Data/labels.txt";

Console.WriteLine("=== Product Image Classifier ===\n");

// Initialize service
using var classifier = new ClassificationService(modelPath, labelsPath);

Console.WriteLine("\n--- Ready for classification ---\n");

// Single image classification
string testImage = args.Length > 0 ? args[0] : "TestImages/sample.jpg";

if (File.Exists(testImage))
{
    Console.WriteLine($"Classifying: {testImage}\n");
    
    var result = classifier.Classify(testImage, topK: 5);
    
    Console.WriteLine("Top Predictions:");
    Console.WriteLine(new string('-', 50));
    
    foreach (var prediction in result.TopPredictions)
    {
        var bar = new string('█', (int)(prediction.Confidence * 30));
        Console.WriteLine($"{prediction.Label,-25} {prediction.Confidence,7:P1} {bar}");
    }
    
    Console.WriteLine(new string('-', 50));
    Console.WriteLine($"Inference time: {result.InferenceTime.TotalMilliseconds:F1}ms");
}
else
{
    Console.WriteLine($"Image not found: {testImage}");
    Console.WriteLine("Usage: dotnet run <image_path>");
}

// Batch processing example
var testDir = "TestImages";
if (Directory.Exists(testDir))
{
    var images = Directory.GetFiles(testDir, "*.jpg")
        .Concat(Directory.GetFiles(testDir, "*.png"))
        .Take(10)
        .ToArray();
    
    if (images.Length > 1)
    {
        Console.WriteLine($"\n\n=== Batch Processing {images.Length} images ===\n");
        
        var sw = System.Diagnostics.Stopwatch.StartNew();
        var batchResults = await classifier.ClassifyBatchAsync(images, topK: 3);
        sw.Stop();
        
        foreach (var result in batchResults)
        {
            var filename = Path.GetFileName(result.ImagePath);
            var topPred = result.TopPredictions.First();
            Console.WriteLine($"{filename,-30} → {topPred.Label} ({topPred.Confidence:P1})");
        }
        
        Console.WriteLine($"\nTotal batch time: {sw.ElapsedMilliseconds}ms");
        Console.WriteLine($"Average per image: {sw.ElapsedMilliseconds / images.Length}ms");
    }
}
```

### Step 6: Running the Classifier

```bash
# Download a test image
curl -o TestImages/cat.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"

# Run classification
dotnet run TestImages/cat.jpg
```

Expected output:
```
=== Product Image Classifier ===

Model Input: data
  Shape: [1, 3, 224, 224]
  Type: Float
Model Output: resnetv24_dense0_fwd
Labels loaded: 1000

--- Ready for classification ---

Classifying: TestImages/cat.jpg

Top Predictions:
--------------------------------------------------
Egyptian cat               87.3% █████████████████████████
tabby                       8.2% ██
tiger cat                   2.1% 
Persian cat                 1.4% 
Siamese cat                 0.6% 
--------------------------------------------------
Inference time: 42.3ms
```

### Step 7: Creating a REST API

For production deployment, wrap the classifier in an ASP.NET Core API:

```csharp
// Create a new file: ProductClassifier.Api/Program.cs
using ProductClassifier.Services;

var builder = WebApplication.CreateBuilder(args);

// Register classifier as singleton (model stays in memory)
builder.Services.AddSingleton<ClassificationService>(sp =>
    new ClassificationService("Models/resnet50.onnx", "Data/labels.txt"));

var app = builder.Build();

app.MapPost("/classify", async (IFormFile file, ClassificationService classifier) =>
{
    if (file.Length == 0)
        return Results.BadRequest("No file provided");
    
    // Save temporarily
    var tempPath = Path.GetTempFileName();
    try
    {
        using (var stream = File.Create(tempPath))
        {
            await file.CopyToAsync(stream);
        }
        
        var result = classifier.Classify(tempPath, topK: 5);
        
        return Results.Ok(new
        {
            filename = file.FileName,
            predictions = result.TopPredictions.Select(p => new
            {
                label = p.Label,
                confidence = p.Confidence
            }),
            inferenceMs = result.InferenceTime.TotalMilliseconds
        });
    }
    finally
    {
        File.Delete(tempPath);
    }
});

app.MapGet("/health", () => Results.Ok(new { status = "healthy" }));

app.Run();
```

Test with curl:

```bash
curl -X POST -F "file=@TestImages/cat.jpg" http://localhost:5000/classify
```

## Performance Optimization Tips

Real-world deployments need speed. Here are techniques to maximize throughput:

### 1. Warm Up the Model

The first inference is always slow (JIT compilation, memory allocation). Run a dummy inference at startup:

```csharp
public void WarmUp()
{
    var dummyTensor = new float[1 * 3 * 224 * 224];
    var input = new DenseTensor<float>(dummyTensor, new[] { 1, 3, 224, 224 });
    
    using var result = _session.Run(new[]
    {
        NamedOnnxValue.CreateFromTensor(_inputName, input)
    });
    
    Console.WriteLine("Model warmed up");
}
```

### 2. Batch Processing

Process multiple images in a single inference call when possible:

```csharp
// Process 8 images at once
var batchSize = 8;
var batchTensor = new DenseTensor<float>(
    batchData, 
    new[] { batchSize, 3, 224, 224 }
);
```

### 3. Use Memory Pools

Reduce garbage collection pressure by reusing memory:

```csharp
using System.Buffers;

var pool = ArrayPool<float>.Shared;
var tensorData = pool.Rent(3 * 224 * 224);
try
{
    // Fill tensorData...
    // Run inference...
}
finally
{
    pool.Return(tensorData);
}
```

### 4. Parallel Preprocessing

Preprocessing is often the bottleneck. Parallelize it:

```csharp
var preprocessed = await Task.WhenAll(
    images.Select(img => Task.Run(() => _preprocessor.PreprocessImage(img)))
);
```

## Summary

In this chapter, you've learned the fundamentals of computer vision in C#:

- **Images as data**: Understanding tensors, channel ordering (NCHW vs HWC), and why it matters for debugging
- **Preprocessing**: Resizing strategies, normalization (especially ImageNet standards), and data augmentation for training
- **Transfer learning**: Why you almost never train from scratch, and how to leverage pre-trained models
- **ONNX integration**: Using ONNX Runtime directly and through ML.NET for flexible, high-performance inference
- **Object detection concepts**: Bounding boxes, NMS, and popular model architectures
- **Production system**: A complete image classifier with preprocessing, batch processing, and API deployment

The computer vision ecosystem is vast, but you now have the foundation to build real systems. Whether you're classifying product images, detecting defects in manufacturing, or analyzing medical scans, the patterns in this chapter apply.

Remember: the models are commoditized. Your value is in building reliable, maintainable systems that integrate these capabilities into production workflows. That's exactly what your C# background prepares you for.

## Exercises

1. **Multi-Model Comparison**: Download two different classification models (e.g., ResNet-50 and EfficientNet) from the ONNX Model Zoo. Build a comparison tool that runs the same images through both models and reports differences in predictions, confidence levels, and inference time. Which model performs better for your test images?

2. **Custom Preprocessing Pipeline**: The standard ImageNet preprocessing may not be optimal for your domain. Create an `IImagePreprocessor` interface and implement three different strategies: standard center crop, padding with letterboxing, and multi-crop averaging (classify the same image at multiple crops and average the results). Compare accuracy on a test set.

3. **Object Detection Implementation**: Download a YOLO model in ONNX format and implement a complete object detection pipeline including:
   - Image preprocessing for YOLO's expected input
   - Output parsing (YOLO outputs raw detections that need decoding)
   - Non-maximum suppression
   - Visualization (draw bounding boxes on the original image)
   
   Test on images containing multiple objects.

4. **Batch Processing Optimization**: Using the classifier from this chapter, implement and benchmark three batch processing strategies:
   - Sequential processing (one image at a time)
   - Parallel preprocessing with sequential inference
   - True batch inference (if the model supports batch size > 1)
   
   Find the optimal batch size for your hardware. At what point do diminishing returns set in?

5. **Production Hardening**: Take the REST API example and add production-ready features:
   - Input validation (file size limits, allowed formats)
   - Caching of classification results (same image hash = cached result)
   - Request rate limiting
   - Structured logging with inference metrics
   - Health check endpoint that verifies the model is loaded and responsive
   - Graceful shutdown that completes in-flight requests
   
   Deploy to a container and load test with 100 concurrent requests.

---

*Next Chapter: Natural Language Processing with C# →*
