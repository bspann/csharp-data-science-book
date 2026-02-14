// Chapter 15: Deep Learning with TorchSharp
// MNIST Digit Classification using a Convolutional Neural Network (CNN)

using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

Console.WriteLine("=== Chapter 15: Deep Learning with TorchSharp ===");
Console.WriteLine($"TorchSharp version: {torch.__version__}");
Console.WriteLine($"Device: {(torch.cuda.is_available() ? "CUDA" : "CPU")}\n");

// ============================================================================
// TENSOR OPERATIONS DEMO
// ============================================================================
Console.WriteLine("--- Tensor Operations ---");

// Creating tensors
var scalar = torch.tensor(3.14f);
var vector = torch.tensor(new float[] { 1, 2, 3, 4, 5 });
var matrix = torch.tensor(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });

Console.WriteLine($"Scalar: {scalar.item<float>()}");
Console.WriteLine($"Vector shape: [{string.Join(", ", vector.shape)}]");
Console.WriteLine($"Matrix shape: [{string.Join(", ", matrix.shape)}]");

// Tensor operations
var a = torch.randn(3, 3);
var b = torch.randn(3, 3);
var sum = a + b;
var product = torch.matmul(a, b);
Console.WriteLine($"Matrix multiplication result shape: [{string.Join(", ", product.shape)}]");

// Reshaping
var flat = torch.arange(12, dtype: torch.float32);
var reshaped = flat.reshape(3, 4);
Console.WriteLine($"Reshaped from [{string.Join(", ", flat.shape)}] to [{string.Join(", ", reshaped.shape)}]\n");

// ============================================================================
// SYNTHETIC MNIST-LIKE DATA
// ============================================================================
// Generate synthetic 28x28 grayscale digit patterns for demonstration
// In production, you would load the actual MNIST dataset

Console.WriteLine("--- Generating Synthetic Training Data ---");

var (trainImages, trainLabels) = GenerateSyntheticDigits(1000);
var (testImages, testLabels) = GenerateSyntheticDigits(200);

Console.WriteLine($"Training samples: {trainImages.shape[0]}");
Console.WriteLine($"Test samples: {testImages.shape[0]}");
Console.WriteLine($"Image shape: [1, 28, 28] (channels, height, width)\n");

// ============================================================================
// CNN MODEL ARCHITECTURE
// ============================================================================
Console.WriteLine("--- CNN Architecture ---");

var model = new MnistCNN("MnistCNN");
Console.WriteLine(model);

// Count parameters
long totalParams = 0;
foreach (var (name, param) in model.named_parameters())
{
    long count = param.numel();
    totalParams += count;
    Console.WriteLine($"  {name}: {string.Join("x", param.shape)} = {count:N0} params");
}
Console.WriteLine($"Total trainable parameters: {totalParams:N0}\n");

// ============================================================================
// TRAINING LOOP
// ============================================================================
Console.WriteLine("--- Training ---");

// Hyperparameters
int epochs = 10;
int batchSize = 32;
float learningRate = 0.001f;

// Loss function and optimizer
var criterion = nn.CrossEntropyLoss();
var optimizer = torch.optim.Adam(model.parameters(), lr: learningRate);

// Training loop
for (int epoch = 1; epoch <= epochs; epoch++)
{
    model.train();
    float epochLoss = 0;
    int batches = 0;

    // Mini-batch training
    for (int i = 0; i < trainImages.shape[0]; i += batchSize)
    {
        int actualBatchSize = (int)Math.Min(batchSize, trainImages.shape[0] - i);
        
        // Get batch
        using var batchImages = trainImages.slice(0, i, i + actualBatchSize, 1);
        using var batchLabels = trainLabels.slice(0, i, i + actualBatchSize, 1);

        // Forward pass
        optimizer.zero_grad();
        using var outputs = model.forward(batchImages);
        using var loss = criterion.forward(outputs, batchLabels);

        // Backward pass and optimization
        loss.backward();
        optimizer.step();

        epochLoss += loss.item<float>();
        batches++;
    }

    // Print progress every epoch
    float avgLoss = epochLoss / batches;
    
    // Evaluate on test set
    model.eval();
    float accuracy = EvaluateModel(model, testImages, testLabels);
    
    Console.WriteLine($"Epoch {epoch,2}/{epochs} | Loss: {avgLoss:F4} | Test Accuracy: {accuracy:P1}");
}

Console.WriteLine();

// ============================================================================
// FINAL EVALUATION
// ============================================================================
Console.WriteLine("--- Final Evaluation ---");

model.eval();
float finalAccuracy = EvaluateModel(model, testImages, testLabels);
Console.WriteLine($"Final Test Accuracy: {finalAccuracy:P2}");

// Per-class accuracy
var classAccuracy = EvaluatePerClass(model, testImages, testLabels);
Console.WriteLine("\nPer-digit accuracy:");
for (int i = 0; i < 10; i++)
{
    Console.WriteLine($"  Digit {i}: {classAccuracy[i]:P1}");
}
Console.WriteLine();

// ============================================================================
// SINGLE PREDICTION DEMO
// ============================================================================
Console.WriteLine("--- Single Digit Prediction ---");

// Take a few test samples and show predictions
using (torch.no_grad())
{
    for (int i = 0; i < 5; i++)
    {
        using var sample = testImages[i].unsqueeze(0);
        using var output = model.forward(sample);
        using var probabilities = nn.functional.softmax(output, dim: 1);
        
        int predicted = (int)output.argmax(1).item<long>();
        int actual = (int)testLabels[i].item<long>();
        float confidence = probabilities[0, predicted].item<float>();
        
        string status = predicted == actual ? "✓" : "✗";
        Console.WriteLine($"Sample {i + 1}: Predicted={predicted}, Actual={actual}, Confidence={confidence:P1} {status}");
    }
}

Console.WriteLine("\n=== Training Complete ===");

// Cleanup
trainImages.Dispose();
trainLabels.Dispose();
testImages.Dispose();
testLabels.Dispose();

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static (Tensor images, Tensor labels) GenerateSyntheticDigits(int numSamples)
{
    // Generate synthetic digit-like patterns
    // Each digit has a characteristic pattern for demonstration
    
    var images = torch.zeros(numSamples, 1, 28, 28);
    var labels = torch.zeros(numSamples, dtype: torch.int64);
    
    var random = new Random(42);
    
    for (int i = 0; i < numSamples; i++)
    {
        int digit = random.Next(10);
        labels[i] = digit;
        
        // Create synthetic pattern based on digit
        var pattern = CreateDigitPattern(digit, random);
        images[i, 0] = pattern;
    }
    
    // Normalize to [0, 1]
    images = images / 255.0f;
    
    return (images, labels);
}

static Tensor CreateDigitPattern(int digit, Random random)
{
    var pattern = torch.zeros(28, 28);
    float noise = 0.1f;
    
    // Create characteristic patterns for each digit
    // These are simplified representations for demonstration
    switch (digit)
    {
        case 0: // Circle
            DrawCircle(pattern, 14, 14, 8);
            break;
        case 1: // Vertical line
            DrawLine(pattern, 14, 4, 14, 24);
            break;
        case 2: // Top arc + diagonal + bottom line
            DrawArc(pattern, 14, 8, 6, 180, 360);
            DrawLine(pattern, 8, 14, 20, 22);
            DrawLine(pattern, 8, 22, 20, 22);
            break;
        case 3: // Two arcs
            DrawArc(pattern, 14, 10, 5, -90, 90);
            DrawArc(pattern, 14, 18, 5, -90, 90);
            break;
        case 4: // Down + right + vertical
            DrawLine(pattern, 8, 4, 8, 14);
            DrawLine(pattern, 8, 14, 20, 14);
            DrawLine(pattern, 18, 4, 18, 24);
            break;
        case 5: // Top line + left + bottom arc
            DrawLine(pattern, 8, 6, 20, 6);
            DrawLine(pattern, 8, 6, 8, 14);
            DrawArc(pattern, 14, 18, 6, -90, 90);
            break;
        case 6: // Circle with top curve
            DrawCircle(pattern, 14, 16, 6);
            DrawArc(pattern, 14, 8, 6, 90, 270);
            break;
        case 7: // Top line + diagonal
            DrawLine(pattern, 6, 6, 22, 6);
            DrawLine(pattern, 22, 6, 10, 24);
            break;
        case 8: // Two circles
            DrawCircle(pattern, 14, 10, 5);
            DrawCircle(pattern, 14, 18, 5);
            break;
        case 9: // Circle with bottom curve
            DrawCircle(pattern, 14, 10, 6);
            DrawArc(pattern, 14, 18, 6, -90, 90);
            break;
    }
    
    // Add noise for variation
    var noisePattern = torch.randn(28, 28) * noise * 128;
    pattern = pattern + noisePattern;
    pattern = torch.clamp(pattern, 0, 255);
    
    return pattern;
}

static void DrawCircle(Tensor pattern, int cx, int cy, int radius)
{
    for (int y = 0; y < 28; y++)
    {
        for (int x = 0; x < 28; x++)
        {
            double dist = Math.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            if (Math.Abs(dist - radius) < 1.5)
            {
                pattern[y, x] = 200 + new Random().Next(55);
            }
        }
    }
}

static void DrawLine(Tensor pattern, int x1, int y1, int x2, int y2)
{
    int dx = Math.Abs(x2 - x1);
    int dy = Math.Abs(y2 - y1);
    int steps = Math.Max(dx, dy);
    
    if (steps == 0)
    {
        if (x1 >= 0 && x1 < 28 && y1 >= 0 && y1 < 28)
            pattern[y1, x1] = 200 + new Random().Next(55);
        return;
    }
    
    float xStep = (x2 - x1) / (float)steps;
    float yStep = (y2 - y1) / (float)steps;
    
    for (int i = 0; i <= steps; i++)
    {
        int x = (int)(x1 + xStep * i);
        int y = (int)(y1 + yStep * i);
        
        if (x >= 0 && x < 28 && y >= 0 && y < 28)
        {
            pattern[y, x] = 200 + new Random().Next(55);
            // Add thickness
            if (x + 1 < 28) pattern[y, x + 1] = 180 + new Random().Next(55);
            if (y + 1 < 28) pattern[y + 1, x] = 180 + new Random().Next(55);
        }
    }
}

static void DrawArc(Tensor pattern, int cx, int cy, int radius, int startAngle, int endAngle)
{
    for (int angle = startAngle; angle <= endAngle; angle += 5)
    {
        double rad = angle * Math.PI / 180;
        int x = (int)(cx + radius * Math.Cos(rad));
        int y = (int)(cy + radius * Math.Sin(rad));
        
        if (x >= 0 && x < 28 && y >= 0 && y < 28)
        {
            pattern[y, x] = 200 + new Random().Next(55);
            if (x + 1 < 28) pattern[y, x + 1] = 180 + new Random().Next(55);
        }
    }
}

static float EvaluateModel(MnistCNN model, Tensor images, Tensor labels)
{
    int correct = 0;
    int total = (int)images.shape[0];
    
    using (torch.no_grad())
    {
        using var outputs = model.forward(images);
        using var predictions = outputs.argmax(1);
        
        for (int i = 0; i < total; i++)
        {
            if (predictions[i].item<long>() == labels[i].item<long>())
                correct++;
        }
    }
    
    return (float)correct / total;
}

static float[] EvaluatePerClass(MnistCNN model, Tensor images, Tensor labels)
{
    var correct = new int[10];
    var total = new int[10];
    
    using (torch.no_grad())
    {
        using var outputs = model.forward(images);
        using var predictions = outputs.argmax(1);
        
        for (int i = 0; i < images.shape[0]; i++)
        {
            int label = (int)labels[i].item<long>();
            int pred = (int)predictions[i].item<long>();
            
            total[label]++;
            if (pred == label) correct[label]++;
        }
    }
    
    var accuracy = new float[10];
    for (int i = 0; i < 10; i++)
    {
        accuracy[i] = total[i] > 0 ? (float)correct[i] / total[i] : 0;
    }
    
    return accuracy;
}

// ============================================================================
// CNN MODEL CLASS
// ============================================================================

/// <summary>
/// Convolutional Neural Network for MNIST digit classification.
/// Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC -> Output
/// </summary>
public class MnistCNN : Module<Tensor, Tensor>
{
    private readonly Module<Tensor, Tensor> conv1;
    private readonly Module<Tensor, Tensor> conv2;
    private readonly Module<Tensor, Tensor> pool;
    private readonly Module<Tensor, Tensor> fc1;
    private readonly Module<Tensor, Tensor> fc2;
    private readonly Module<Tensor, Tensor> dropout;

    public MnistCNN(string name) : base(name)
    {
        // Convolutional layers
        // Input: 1x28x28 -> Conv1: 32x26x26 -> Pool: 32x13x13
        // Conv2d(inputChannel, outputChannel, kernelSize, stride, padding, dilation)
        conv1 = nn.Conv2d(1, 32, 3, 1, 0, 1);

        // Conv1 output: 32x13x13 -> Conv2: 64x11x11 -> Pool: 64x5x5
        conv2 = nn.Conv2d(32, 64, 3, 1, 0, 1);

        // Max pooling layer (2x2 with stride 2)
        pool = nn.MaxPool2d(2, stride: 2);

        // Fully connected layers
        // After conv2 + pool: 64 * 5 * 5 = 1600 features
        fc1 = nn.Linear(64 * 5 * 5, 128);
        fc2 = nn.Linear(128, 10);  // 10 output classes (digits 0-9)

        // Dropout for regularization
        dropout = nn.Dropout(0.5);

        // Register all modules
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // First convolutional block
        x = conv1.forward(x);
        x = nn.functional.relu(x);
        x = pool.forward(x);

        // Second convolutional block
        x = conv2.forward(x);
        x = nn.functional.relu(x);
        x = pool.forward(x);

        // Flatten for fully connected layers
        x = x.flatten(1);

        // Fully connected layers with dropout
        x = fc1.forward(x);
        x = nn.functional.relu(x);
        x = dropout.forward(x);

        // Output layer (no activation - CrossEntropyLoss applies softmax)
        x = fc2.forward(x);

        return x;
    }
}
