# Chapter 15: Deep Learning with TorchSharp

Throughout this book, we've accomplished remarkable things with ML.NET's classical machine learning algorithms. Gradient boosting, random forests, matrix factorization—these techniques have served us well. But there's a class of problems where classical ML hits a wall, and that's when deep learning enters the picture.

You've probably heard the hype. Neural networks power everything from voice assistants to self-driving cars. Large language models generate human-like text. Computer vision systems recognize faces, detect diseases in medical images, and navigate robots through warehouses. The deep learning revolution is real.

But here's what the hype doesn't tell you: deep learning isn't always the answer. It's computationally expensive, data-hungry, and harder to interpret than classical ML. The key skill isn't knowing *how* to build neural networks—it's knowing *when* to build them.

In this chapter, we'll explore deep learning through **TorchSharp**, Microsoft's .NET bindings for PyTorch. You'll learn to think about neural networks at a conceptual level, understand when they're the right tool, and build a complete image classification system from scratch. By the end, you'll have trained a convolutional neural network on GPU and deployed it for inference—all in C#.

## When to Use Deep Learning vs. Classical ML

Before we write a single line of neural network code, let's develop the intuition for when deep learning is worth the complexity.

### The Deep Learning Sweet Spot

Deep learning excels in specific scenarios:

**1. Unstructured data with complex patterns**

Images, audio, video, and text have inherent spatial or sequential structure that neural networks capture naturally. A random forest doesn't know that pixel (10, 10) is spatially close to pixel (10, 11), but a convolutional neural network does.

If your data is tabular—rows of customers with features like age, income, and purchase history—deep learning probably isn't your best choice. Gradient boosting methods (LightGBM, XGBoost) typically outperform neural networks on tabular data while being faster to train and easier to interpret.

**2. Massive datasets**

Neural networks are hungry. They need thousands to millions of examples to learn effectively. Their power comes from learning hierarchical representations, and that requires data.

If you have 500 training examples, stick with classical ML. If you have 500,000, deep learning becomes viable. If you have 50 million, it's often the clear winner.

**3. Problems where feature engineering is impossible or impractical**

With tabular data, you can engineer features based on domain knowledge. You know that "days since last purchase" is meaningful because you understand the business.

But how do you engineer features for an image of a cat? You could try—edge detectors, color histograms, texture descriptors—but hand-crafted features can only go so far. Deep learning learns its own features, automatically discovering what matters.

**4. State-of-the-art requirements**

For certain problems—image classification, speech recognition, machine translation, text generation—deep learning isn't just better; it's in a different league entirely. If you need the best possible performance on these tasks, you need neural networks.

### When Classical ML Wins

Deep learning is not universally superior. Classical ML wins when:

**You have limited data.** Neural networks overfit quickly on small datasets. Regularization helps, but a well-tuned random forest will often outperform a neural network with only a few thousand examples.

**Interpretability matters.** Try explaining to a regulator why your neural network denied a loan. The weights of a deep network aren't interpretable in any meaningful way. Decision trees, logistic regression, and even gradient boosting offer some transparency; neural networks are black boxes.

**Computational resources are constrained.** Training deep networks requires GPUs and time. Inference can be slow without optimization. If you're running on a Raspberry Pi or need sub-millisecond predictions, simpler models are often necessary.

**Tabular data is your domain.** I'll say it again because it's counterintuitive: for structured, tabular data with well-defined features, gradient boosting methods (which we covered in Chapter 10) typically outperform deep learning. The Kaggle leaderboards don't lie.

### The Decision Framework

Here's my practical framework:

| Scenario | Recommendation |
|----------|----------------|
| Tabular data, <10K rows | Classical ML (Random Forest, Gradient Boosting) |
| Tabular data, >10K rows | Try both; gradient boosting often wins |
| Images, audio, video | Deep learning (CNNs, RNNs, Transformers) |
| Text classification with labeled data | Deep learning or fine-tuned transformers |
| Sequence prediction | Deep learning (LSTM, GRU, Transformer) |
| Need interpretability | Classical ML or attention-based neural networks |
| Limited compute resources | Classical ML or quantized neural networks |

The honest truth? When in doubt, try both. Train a gradient boosting model and a neural network on the same problem. Evaluate them fairly. Let the data decide.

## Introduction to TorchSharp

TorchSharp brings PyTorch to .NET. This matters more than it might sound.

PyTorch is one of the two dominant deep learning frameworks (alongside TensorFlow), and it's the preferred choice in research. Most cutting-edge papers publish PyTorch code. Most pre-trained models are available in PyTorch format. When you use TorchSharp, you're tapping into this ecosystem from C#.

### What TorchSharp Provides

TorchSharp is not a wrapper around Python. It's C# bindings to LibTorch, the C++ library that underlies PyTorch. This means:

- **Native performance**: No Python interpreter overhead
- **Familiar API**: If you've used PyTorch, TorchSharp feels identical
- **GPU support**: Full CUDA integration for NVIDIA GPUs
- **Interoperability**: Load PyTorch models directly into C#

### Installation

Let's set up a TorchSharp project. Create a new console application:

```bash
mkdir DeepLearningDemo
cd DeepLearningDemo
dotnet new console -n MnistClassifier
cd MnistClassifier
```

Add the required packages:

```bash
# Core TorchSharp
dotnet add package TorchSharp

# CPU backend (always needed)
dotnet add package TorchSharp-cpu

# For NVIDIA GPU support (optional, but recommended)
dotnet add package TorchSharp-cuda-windows  # On Windows
# or
dotnet add package TorchSharp-cuda-linux    # On Linux
```

The package structure might seem odd, but it makes sense: `TorchSharp` contains the .NET API, while the `-cpu` and `-cuda` packages contain the native binaries for different hardware. You need at least the CPU backend; CUDA packages are only for GPU acceleration.

### Verifying Installation

Let's verify everything works:

```csharp
using TorchSharp;
using static TorchSharp.torch;

// Check available devices
Console.WriteLine($"TorchSharp version: {torch.__version__}");
Console.WriteLine($"CUDA available: {torch.cuda.is_available()}");

if (torch.cuda.is_available())
{
    Console.WriteLine($"CUDA devices: {torch.cuda.device_count()}");
    Console.WriteLine($"CUDA device name: {torch.cuda.get_device_name(0)}");
}

// Create a simple tensor
var tensor = torch.randn(3, 3);
Console.WriteLine($"\nRandom 3x3 tensor:\n{tensor}");
```

If CUDA is available, you'll see output like:

```
TorchSharp version: 0.102.0
CUDA available: True
CUDA devices: 1
CUDA device name: NVIDIA GeForce RTX 3080

Random 3x3 tensor:
[3, 3], type = Float32, device = cpu
 0.2851  1.1247  0.0842
-0.5421  0.9283  1.8421
 0.1234 -0.7812  0.4521
```

### The Tensor API

Tensors are the fundamental data structure in TorchSharp—multi-dimensional arrays that can live on CPU or GPU. If you've used NumPy arrays, tensors will feel familiar.

```csharp
using TorchSharp;
using static TorchSharp.torch;

// Create tensors
var zeros = torch.zeros(3, 4);           // 3x4 tensor of zeros
var ones = torch.ones(2, 3, 4);          // 2x3x4 tensor of ones
var random = torch.randn(5, 5);          // Random normal distribution
var range = torch.arange(0, 10, 2);      // [0, 2, 4, 6, 8]

// From C# arrays
float[] data = { 1, 2, 3, 4, 5, 6 };
var fromArray = torch.tensor(data).reshape(2, 3);

// Basic operations
var a = torch.randn(3, 3);
var b = torch.randn(3, 3);

var sum = a + b;                         // Element-wise addition
var product = a * b;                     // Element-wise multiplication
var matmul = torch.matmul(a, b);         // Matrix multiplication
var transposed = a.T;                    // Transpose

// Reduction operations
var mean = a.mean();
var max = a.max();
var sumAll = a.sum();

// Move to GPU (if available)
if (torch.cuda.is_available())
{
    var gpuTensor = a.cuda();            // Copy to GPU
    var result = gpuTensor * 2;          // Computation happens on GPU
    var cpuResult = result.cpu();        // Copy back to CPU
}
```

The tensor API is your foundation for everything else. Spend time getting comfortable with it before diving into neural networks.

## Neural Network Basics

Now let's build understanding from first principles. What actually happens inside a neural network?

### The Fundamental Unit: Neurons and Layers

A neural network is a composition of simple functions. The basic building block is a **neuron**, which:

1. Takes multiple inputs
2. Multiplies each by a weight
3. Sums the results
4. Adds a bias
5. Applies an activation function

Mathematically: `output = activation(sum(inputs * weights) + bias)`

A **layer** is a collection of neurons that process the same inputs in parallel. A **network** is multiple layers stacked, where each layer's output becomes the next layer's input.

[FIGURE: Diagram showing a simple neural network with input layer (3 neurons), hidden layer (4 neurons), and output layer (2 neurons), with connections between all neurons in adjacent layers]

In TorchSharp, a simple layer is created with `nn.Linear`:

```csharp
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// A layer that takes 10 inputs and produces 5 outputs
var layer = Linear(10, 5);

// Pass data through
var input = torch.randn(1, 10);   // Batch of 1, with 10 features
var output = layer.forward(input); // Output shape: (1, 5)

Console.WriteLine($"Input shape: {input.shape}");
Console.WriteLine($"Output shape: {output.shape}");
```

### Activation Functions: Adding Non-Linearity

Here's a crucial insight: without activation functions, a neural network is just a linear transformation. No matter how many layers you stack, the result is equivalent to a single matrix multiplication. Deep networks would be no more powerful than shallow ones.

Activation functions introduce **non-linearity**, allowing networks to learn complex, non-linear relationships.

Common activation functions:

| Function | Formula | Use Case |
|----------|---------|----------|
| ReLU | max(0, x) | Default for hidden layers |
| Sigmoid | 1 / (1 + e^-x) | Binary classification output |
| Tanh | (e^x - e^-x) / (e^x + e^-x) | When you need outputs in [-1, 1] |
| Softmax | e^xi / Σe^xj | Multi-class classification output |

[FIGURE: Graph showing ReLU, Sigmoid, and Tanh activation functions plotted on the same axes, with x ranging from -3 to 3]

In TorchSharp:

```csharp
var x = torch.tensor(new float[] { -2, -1, 0, 1, 2 });

var relu = functional.relu(x);        // [0, 0, 0, 1, 2]
var sigmoid = functional.sigmoid(x);   // [0.12, 0.27, 0.5, 0.73, 0.88]
var tanh = functional.tanh(x);        // [-0.96, -0.76, 0, 0.76, 0.96]
```

### Backpropagation: How Networks Learn

Training a neural network means adjusting weights to minimize a **loss function**—a measure of how wrong the predictions are. But how do you know which weights to adjust and by how much?

The answer is **backpropagation**, which is just the chain rule from calculus applied systematically. Here's the intuition:

1. **Forward pass**: Data flows through the network, producing predictions
2. **Loss calculation**: Compare predictions to actual values
3. **Backward pass**: Calculate how much each weight contributed to the error
4. **Update**: Adjust weights in the direction that reduces error

Let's make this concrete. Imagine a simple network with one weight `w` that multiplies an input `x` to produce output `y = w * x`. If the true value should be `t`, the loss is `L = (y - t)²`.

The gradient of the loss with respect to `w` tells us how to adjust `w`:
- If the gradient is positive, increasing `w` increases the loss—so decrease `w`
- If the gradient is negative, increasing `w` decreases the loss—so increase `w`

In a deep network with millions of weights, this calculation becomes complex. Each weight's effect on the output ripples through all subsequent layers. Backpropagation efficiently computes all these gradients by working backward from the output, reusing intermediate calculations.

The beauty of modern frameworks like TorchSharp is that backpropagation is automatic. You define the forward pass; the framework computes gradients for you.

```csharp
// Enable gradient tracking
var x = torch.tensor(3.0f, requires_grad: true);
var y = x * x + 2 * x + 1;  // y = x² + 2x + 1

// Compute gradients
y.backward();

// dy/dx at x=3 should be 2*3 + 2 = 8
Console.WriteLine($"Gradient: {x.grad}");  // Output: 8
```

This automatic differentiation extends to entire neural networks with millions of parameters. You never need to compute gradients manually.

### Understanding the Computation Graph

When you perform operations on tensors with `requires_grad: true`, TorchSharp builds a **computation graph** behind the scenes. This graph records every operation, creating a "tape" of what happened during the forward pass.

When you call `backward()`, the framework "plays the tape backward," computing gradients using the chain rule. This is why you must call `zero_grad()` before each training step—otherwise, gradients accumulate from previous steps.

```csharp
var model = new SimpleModel();
var optimizer = torch.optim.Adam(model.parameters());

// Each training iteration
optimizer.zero_grad();      // Clear gradients from previous iteration
var output = model.forward(input);
var loss = ComputeLoss(output, target);
loss.backward();            // Compute new gradients
optimizer.step();           // Update weights using gradients
```

Understanding the computation graph helps debug issues. If you see an error about "trying to backward through the graph a second time," you've likely forgotten to detach tensors that should not participate in gradient computation.

### Loss Functions

The loss function quantifies how wrong your predictions are. Different problems need different loss functions:

**Regression** (predicting continuous values):
- Mean Squared Error (MSE): Average of (prediction - actual)²
- Mean Absolute Error (MAE): Average of |prediction - actual|

**Classification** (predicting categories):
- Cross-Entropy Loss: Measures divergence between predicted and actual probability distributions
- Binary Cross-Entropy: For two-class problems

```csharp
// Regression loss
var predictions = torch.tensor(new float[] { 1.0f, 2.0f, 3.0f });
var targets = torch.tensor(new float[] { 1.1f, 2.2f, 2.8f });
var mseLoss = functional.mse_loss(predictions, targets);

// Classification loss
var logits = torch.tensor(new float[,] { { 2.0f, 1.0f, 0.1f } });
var labels = torch.tensor(new long[] { 0 }); // Correct class is 0
var crossEntropyLoss = functional.cross_entropy(logits, labels);
```

### Optimizers: Gradient Descent and Beyond

Once you have gradients, you need an **optimizer** to update the weights. The simplest approach is **gradient descent**: move each weight in the direction that reduces loss, proportional to the gradient.

`new_weight = old_weight - learning_rate * gradient`

More sophisticated optimizers improve on this:

- **SGD with momentum**: Adds "velocity" to updates, smoothing out noise
- **Adam**: Adapts learning rates per-parameter, handles sparse gradients well
- **AdamW**: Adam with proper weight decay (regularization)

Adam is the default choice for most problems:

```csharp
var model = new SimpleModel();
var optimizer = torch.optim.Adam(model.parameters(), lr: 0.001);

// Training step
optimizer.zero_grad();           // Clear old gradients
var loss = ComputeLoss(model);   // Forward pass + loss
loss.backward();                 // Compute gradients
optimizer.step();                // Update weights
```

### Regularization: Preventing Overfitting

Neural networks have enormous capacity—they can memorize training data rather than learning general patterns. **Regularization** techniques prevent this overfitting.

**Dropout**: During training, randomly sets a fraction of neuron outputs to zero. This prevents neurons from co-adapting too tightly and forces the network to learn redundant representations.

```csharp
var dropout = nn.Dropout(0.5);  // Drop 50% of neurons during training

// In forward pass
x = dropout.forward(x);  // Only active during model.train() mode
```

**Weight Decay (L2 Regularization)**: Penalizes large weights by adding a term to the loss proportional to the sum of squared weights. This encourages simpler models with smaller weights.

```csharp
// Add weight decay to optimizer
var optimizer = torch.optim.Adam(
    model.parameters(), 
    lr: 0.001, 
    weight_decay: 0.0001  // L2 penalty coefficient
);
```

**Batch Normalization**: Normalizes activations within each mini-batch. This isn't strictly regularization, but it has a regularizing effect by adding noise (the batch statistics) during training.

```csharp
var bn = nn.BatchNorm2d(64);  // For 64 feature channels

// In forward pass
x = bn.forward(x);  // Normalizes across batch dimension
```

**Early Stopping**: Monitor validation loss during training. If it stops improving while training loss keeps decreasing, you're overfitting. Stop training and use the best checkpoint.

```csharp
double bestValLoss = double.MaxValue;
int patienceCounter = 0;
const int patience = 5;

foreach (var epoch in Enumerable.Range(0, maxEpochs))
{
    Train(model, trainLoader);
    var valLoss = Evaluate(model, valLoader);
    
    if (valLoss < bestValLoss)
    {
        bestValLoss = valLoss;
        patienceCounter = 0;
        model.save("best_model.dat");  // Save best checkpoint
    }
    else
    {
        patienceCounter++;
        if (patienceCounter >= patience)
        {
            Console.WriteLine("Early stopping triggered");
            break;
        }
    }
}
```

In practice, you'll use multiple regularization techniques together. A typical setup: dropout in fully-connected layers, batch normalization in convolutional layers, and weight decay in the optimizer.

## Training Deep Models in C#

Let's build a complete training pipeline. We'll start with a simple fully-connected network for classification, then upgrade to a CNN in the project section.

### Data Loading and Batching

Efficient data loading is crucial for training performance. Loading one sample at a time wastes GPU cycles waiting for data. The solution: **batch processing**.

A **batch** is a group of samples processed together. Larger batches:
- Utilize GPU parallelism better
- Provide more stable gradient estimates
- But require more memory and may generalize worse

Typical batch sizes range from 16 to 256, depending on your GPU memory and dataset size.

```csharp
public IEnumerable<(Tensor data, Tensor labels)> CreateBatches(
    Tensor allData, 
    Tensor allLabels, 
    int batchSize, 
    bool shuffle = true)
{
    var numSamples = (int)allData.shape[0];
    var indices = Enumerable.Range(0, numSamples).ToArray();
    
    if (shuffle)
    {
        var rng = new Random();
        indices = indices.OrderBy(_ => rng.Next()).ToArray();
    }
    
    for (int i = 0; i < numSamples; i += batchSize)
    {
        var end = Math.Min(i + batchSize, numSamples);
        var batchIndices = torch.tensor(indices[i..end]);
        
        var batchData = allData.index_select(0, batchIndices);
        var batchLabels = allLabels.index_select(0, batchIndices);
        
        batchIndices.Dispose();
        
        yield return (batchData, batchLabels);
    }
}
```

**Shuffling** is important: without it, the model might learn spurious patterns from the order of training data. Always shuffle training data; test data order doesn't matter.

### Defining a Model

TorchSharp models inherit from `nn.Module`:

```csharp
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class SimpleClassifier : Module<Tensor, Tensor>
{
    private readonly Linear fc1;
    private readonly Linear fc2;
    private readonly Linear fc3;
    private readonly Dropout dropout;

    public SimpleClassifier(int inputSize, int hiddenSize, int numClasses) 
        : base("SimpleClassifier")
    {
        fc1 = Linear(inputSize, hiddenSize);
        fc2 = Linear(hiddenSize, hiddenSize);
        fc3 = Linear(hiddenSize, numClasses);
        dropout = Dropout(0.5);

        // Register all components
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = functional.relu(fc1.forward(x));
        x = dropout.forward(x);
        x = functional.relu(fc2.forward(x));
        x = dropout.forward(x);
        x = fc3.forward(x);  // No activation - loss function handles it
        return x;
    }
}
```

Key points:
- Inherit from `Module<TInput, TOutput>`
- Define layers as fields
- Call `RegisterComponents()` to enable parameter tracking
- Implement `forward()` for the forward pass

### The Training Loop

Here's the standard training loop pattern:

```csharp
public void Train(
    Module<Tensor, Tensor> model,
    IEnumerable<(Tensor data, Tensor labels)> dataLoader,
    int epochs,
    double learningRate,
    Device device)
{
    model.to(device);
    model.train();  // Enable training mode (dropout active, etc.)

    var optimizer = torch.optim.Adam(model.parameters(), lr: learningRate);

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double epochLoss = 0;
        int batchCount = 0;

        foreach (var (data, labels) in dataLoader)
        {
            // Move data to device
            var x = data.to(device);
            var y = labels.to(device);

            // Forward pass
            var predictions = model.forward(x);
            var loss = functional.cross_entropy(predictions, y);

            // Backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epochLoss += loss.item<float>();
            batchCount++;

            // Clean up intermediate tensors
            x.Dispose();
            y.Dispose();
            loss.Dispose();
        }

        Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Loss: {epochLoss / batchCount:F4}");
    }
}
```

### Evaluation

During evaluation, we disable dropout and don't compute gradients:

```csharp
public double Evaluate(
    Module<Tensor, Tensor> model,
    IEnumerable<(Tensor data, Tensor labels)> testLoader,
    Device device)
{
    model.eval();  // Disable dropout, batch norm uses running stats

    int correct = 0;
    int total = 0;

    using (torch.no_grad())  // Don't track gradients
    {
        foreach (var (data, labels) in testLoader)
        {
            var x = data.to(device);
            var y = labels.to(device);

            var predictions = model.forward(x);
            var predicted = predictions.argmax(dim: 1);

            correct += (predicted == y).sum().item<int>();
            total += (int)y.shape[0];

            x.Dispose();
            y.Dispose();
        }
    }

    return (double)correct / total;
}
```

### Saving and Loading Models

Persist trained models for later use:

```csharp
// Save the model
model.save("model.weights.dat");

// Load later
var loadedModel = new SimpleClassifier(inputSize, hiddenSize, numClasses);
loadedModel.load("model.weights.dat");
loadedModel.eval();
```

## GPU Acceleration with CUDA

Training on GPU can be 10-100x faster than CPU for deep learning. Let's make it happen.

### Understanding GPU Computing

GPUs excel at parallel computation. A neural network forward pass involves many independent matrix operations—exactly what GPUs are designed for. An NVIDIA RTX 3080 has 8,704 CUDA cores working in parallel; a CPU has maybe 8-16 cores.

The tradeoff: moving data between CPU and GPU memory takes time. For small operations, this overhead dominates, and GPU is actually slower. For large matrix operations (big batches, many parameters), GPU wins decisively.

### Setting Up CUDA

First, verify CUDA is available:

```csharp
if (torch.cuda.is_available())
{
    Console.WriteLine($"CUDA available: {torch.cuda.device_count()} device(s)");
    Console.WriteLine($"Device 0: {torch.cuda.get_device_name(0)}");
}
else
{
    Console.WriteLine("CUDA not available. Using CPU.");
}
```

If CUDA isn't detected:
1. Verify you have an NVIDIA GPU (AMD and Intel GPUs use different acceleration paths)
2. Install the latest NVIDIA drivers from nvidia.com
3. Ensure you've added the correct `TorchSharp-cuda-*` package for your OS
4. Check that the CUDA toolkit version matches the TorchSharp-cuda package
5. On Windows, ensure the NVIDIA display driver service is running
6. Try restarting Visual Studio/your IDE after package installation

**Troubleshooting CUDA Issues:**

If CUDA is reported as available but you get errors when moving tensors to GPU:

```csharp
try
{
    var test = torch.randn(100, 100).cuda();
    Console.WriteLine("CUDA is working correctly");
    test.Dispose();
}
catch (Exception ex)
{
    Console.WriteLine($"CUDA error: {ex.Message}");
    Console.WriteLine("Falling back to CPU");
}
```

Common issues include:
- **Out of memory**: Reduce batch size or model size
- **CUDA driver mismatch**: Update NVIDIA drivers
- **LibTorch version mismatch**: Ensure TorchSharp and TorchSharp-cuda packages are the same version

### Device-Agnostic Code

Write code that runs on GPU when available, CPU otherwise:

```csharp
// Choose device
var device = torch.cuda.is_available() 
    ? torch.device("cuda") 
    : torch.device("cpu");

Console.WriteLine($"Using device: {device}");

// Move model to device
var model = new SimpleClassifier(784, 256, 10);
model.to(device);

// Move data to device before processing
var input = torch.randn(32, 784);  // Batch of 32, 784 features
input = input.to(device);

var output = model.forward(input);
```

### Memory Management

GPU memory is limited. A RTX 3080 has 10GB; that sounds like a lot until you're training a network with millions of parameters on batches of high-resolution images.

Best practices:

```csharp
// 1. Use appropriate batch sizes
// Larger batches = more GPU memory, but faster training
// If you get out-of-memory errors, reduce batch size

// 2. Dispose tensors when done
using (var tensor = torch.randn(1000, 1000).cuda())
{
    // Use tensor
} // Automatically disposed

// 3. Clear cache periodically during training
torch.cuda.empty_cache();

// 4. Move results back to CPU before logging
var loss = lossGpu.cpu().item<float>();

// 5. Use mixed precision for larger models (advanced)
// TorchSharp supports float16 for reduced memory usage
```

### Benchmarking CPU vs GPU

Let's measure the difference:

```csharp
void BenchmarkDevice(Device device, int iterations = 100)
{
    var model = new SimpleClassifier(784, 1024, 10);
    model.to(device);
    model.train();
    
    var input = torch.randn(128, 784).to(device);
    var labels = torch.randint(0, 10, 128).to(device);
    
    // Warm-up
    for (int i = 0; i < 10; i++)
    {
        var output = model.forward(input);
        output.Dispose();
    }
    
    var sw = Stopwatch.StartNew();
    
    for (int i = 0; i < iterations; i++)
    {
        var output = model.forward(input);
        var loss = functional.cross_entropy(output, labels);
        loss.backward();
        output.Dispose();
        loss.Dispose();
    }
    
    // Ensure GPU operations complete
    if (device.type == DeviceType.CUDA)
        torch.cuda.synchronize();
    
    sw.Stop();
    
    Console.WriteLine($"{device}: {sw.ElapsedMilliseconds}ms for {iterations} iterations");
    Console.WriteLine($"Average: {sw.ElapsedMilliseconds / (double)iterations:F2}ms per iteration");
}

// Run benchmarks
BenchmarkDevice(torch.device("cpu"));
if (torch.cuda.is_available())
    BenchmarkDevice(torch.device("cuda"));
```

Typical results on a workstation with RTX 3080:
```
cpu: 4523ms for 100 iterations
Average: 45.23ms per iteration
cuda: 312ms for 100 iterations
Average: 3.12ms per iteration
```

That's a 14x speedup. For larger models and larger batches, the difference grows even more.

## Project: Image Classification with a CNN (MNIST)

Now let's put everything together. We'll build a convolutional neural network to classify handwritten digits from the MNIST dataset—the "Hello World" of deep learning.

### Convolutional Neural Networks

For image data, **Convolutional Neural Networks (CNNs)** dramatically outperform fully-connected networks. Why?

A fully-connected network treats each pixel independently. It doesn't know that pixel (10, 10) is next to pixel (10, 11). It has to learn spatial relationships from scratch, which requires enormous amounts of data.

CNNs build in spatial awareness. A **convolutional layer** slides a small filter (kernel) across the image, detecting local patterns—edges, textures, shapes. Early layers detect simple patterns; deeper layers combine these into complex features.

[FIGURE: Illustration of a 3x3 convolution kernel sliding across a 5x5 image, producing a 3x3 feature map. Show the element-wise multiplication and sum at one position.]

Key CNN components:

- **Conv2d**: Convolutional layer that detects local patterns
- **MaxPool2d**: Downsamples by taking maximum values in regions
- **BatchNorm2d**: Normalizes activations for faster, more stable training
- **Flatten**: Converts 2D feature maps to 1D for classification layers

### The MNIST Dataset

MNIST contains 70,000 grayscale images of handwritten digits (0-9), each 28x28 pixels. It's split into 60,000 training and 10,000 test images.

[FIGURE: Grid showing sample MNIST digits - examples of each digit 0-9, showing the handwritten style variations]

Let's download and prepare the data:

```csharp
using System.IO.Compression;
using System.Net.Http;

public class MnistDataLoader
{
    private const string BaseUrl = "https://ossci-datasets.s3.amazonaws.com/mnist/";
    private readonly string _dataPath;

    public MnistDataLoader(string dataPath = "mnist_data")
    {
        _dataPath = dataPath;
        Directory.CreateDirectory(dataPath);
    }

    public async Task DownloadDatasetAsync()
    {
        var files = new[]
        {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        };

        using var client = new HttpClient();

        foreach (var file in files)
        {
            var filePath = Path.Combine(_dataPath, file);
            if (File.Exists(filePath)) continue;

            Console.WriteLine($"Downloading {file}...");
            var data = await client.GetByteArrayAsync(BaseUrl + file);
            await File.WriteAllBytesAsync(filePath, data);
        }

        Console.WriteLine("Dataset downloaded.");
    }

    public (Tensor images, Tensor labels) LoadTrainingData()
    {
        return LoadDataset("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz");
    }

    public (Tensor images, Tensor labels) LoadTestData()
    {
        return LoadDataset("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz");
    }

    private (Tensor images, Tensor labels) LoadDataset(string imagesFile, string labelsFile)
    {
        var images = ReadImages(Path.Combine(_dataPath, imagesFile));
        var labels = ReadLabels(Path.Combine(_dataPath, labelsFile));
        return (images, labels);
    }

    private Tensor ReadImages(string path)
    {
        using var fileStream = File.OpenRead(path);
        using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var reader = new BinaryReader(gzipStream);

        // Read header
        var magic = ReadBigEndianInt32(reader);
        var numImages = ReadBigEndianInt32(reader);
        var rows = ReadBigEndianInt32(reader);
        var cols = ReadBigEndianInt32(reader);

        // Read pixel data
        var data = new float[numImages * rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = reader.ReadByte() / 255.0f;  // Normalize to [0, 1]
        }

        // Shape: (N, 1, 28, 28) - batch, channels, height, width
        return torch.tensor(data).reshape(numImages, 1, rows, cols);
    }

    private Tensor ReadLabels(string path)
    {
        using var fileStream = File.OpenRead(path);
        using var gzipStream = new GZipStream(fileStream, CompressionMode.Decompress);
        using var reader = new BinaryReader(gzipStream);

        var magic = ReadBigEndianInt32(reader);
        var numLabels = ReadBigEndianInt32(reader);

        var labels = new long[numLabels];
        for (int i = 0; i < numLabels; i++)
        {
            labels[i] = reader.ReadByte();
        }

        return torch.tensor(labels);
    }

    private static int ReadBigEndianInt32(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(4);
        if (BitConverter.IsLittleEndian)
            Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}
```

### Building the CNN Architecture

Here's our CNN architecture:

```csharp
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

public class MnistCNN : Module<Tensor, Tensor>
{
    private readonly Conv2d conv1;
    private readonly Conv2d conv2;
    private readonly BatchNorm2d bn1;
    private readonly BatchNorm2d bn2;
    private readonly Dropout2d dropout;
    private readonly Linear fc1;
    private readonly Linear fc2;

    public MnistCNN() : base("MnistCNN")
    {
        // First convolutional block
        // Input: (batch, 1, 28, 28) -> Output: (batch, 32, 14, 14)
        conv1 = Conv2d(1, 32, kernelSize: 3, padding: 1);
        bn1 = BatchNorm2d(32);

        // Second convolutional block
        // Input: (batch, 32, 14, 14) -> Output: (batch, 64, 7, 7)
        conv2 = Conv2d(32, 64, kernelSize: 3, padding: 1);
        bn2 = BatchNorm2d(64);

        dropout = Dropout2d(0.25);

        // Fully connected layers
        // After conv layers and pooling: 64 * 7 * 7 = 3136 features
        fc1 = Linear(64 * 7 * 7, 128);
        fc2 = Linear(128, 10);  // 10 digit classes

        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        // First conv block: conv -> batchnorm -> relu -> maxpool
        x = conv1.forward(x);
        x = bn1.forward(x);
        x = functional.relu(x);
        x = functional.max_pool2d(x, kernelSize: 2);

        // Second conv block
        x = conv2.forward(x);
        x = bn2.forward(x);
        x = functional.relu(x);
        x = functional.max_pool2d(x, kernelSize: 2);

        x = dropout.forward(x);

        // Flatten for fully connected layers
        x = x.flatten(startDim: 1);

        // Fully connected layers
        x = functional.relu(fc1.forward(x));
        x = fc2.forward(x);

        return x;  // Return logits; loss function applies softmax
    }
}
```

### The Complete Training Pipeline

Now let's put it all together:

```csharp
using System.Diagnostics;
using TorchSharp;
using static TorchSharp.torch;

public class MnistTrainer
{
    public async Task RunAsync()
    {
        // Configuration
        const int batchSize = 64;
        const int epochs = 10;
        const double learningRate = 0.001;

        // Device selection
        var device = cuda.is_available() 
            ? torch.device("cuda") 
            : torch.device("cpu");
        Console.WriteLine($"Using device: {device}");

        // Load data
        var loader = new MnistDataLoader();
        await loader.DownloadDatasetAsync();

        var (trainImages, trainLabels) = loader.LoadTrainingData();
        var (testImages, testLabels) = loader.LoadTestData();

        Console.WriteLine($"Training samples: {trainImages.shape[0]}");
        Console.WriteLine($"Test samples: {testImages.shape[0]}");

        // Create model and move to device
        var model = new MnistCNN();
        model.to(device);

        // Print model summary
        Console.WriteLine($"\nModel parameters: {model.parameters().Sum(p => p.numel()):N0}");

        // Optimizer
        var optimizer = torch.optim.Adam(model.parameters(), lr: learningRate);

        // Training loop
        var sw = Stopwatch.StartNew();

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            model.train();
            double totalLoss = 0;
            int batches = 0;

            // Process in batches
            for (int i = 0; i < trainImages.shape[0]; i += batchSize)
            {
                var end = Math.Min(i + batchSize, (int)trainImages.shape[0]);
                var batchImages = trainImages[i..end].to(device);
                var batchLabels = trainLabels[i..end].to(device);

                // Forward pass
                var predictions = model.forward(batchImages);
                var loss = functional.cross_entropy(predictions, batchLabels);

                // Backward pass
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();

                totalLoss += loss.item<float>();
                batches++;

                // Cleanup
                batchImages.Dispose();
                batchLabels.Dispose();
                predictions.Dispose();
                loss.Dispose();
            }

            // Evaluate on test set
            var accuracy = Evaluate(model, testImages, testLabels, device, batchSize);

            Console.WriteLine($"Epoch {epoch + 1,2}/{epochs} | " +
                              $"Loss: {totalLoss / batches:F4} | " +
                              $"Test Accuracy: {accuracy:P2}");
        }

        sw.Stop();
        Console.WriteLine($"\nTraining completed in {sw.Elapsed.TotalSeconds:F1}s");

        // Save the model
        model.save("mnist_cnn.weights.dat");
        Console.WriteLine("Model saved to mnist_cnn.weights.dat");

        // Cleanup
        trainImages.Dispose();
        trainLabels.Dispose();
        testImages.Dispose();
        testLabels.Dispose();
    }

    private double Evaluate(
        Module<Tensor, Tensor> model,
        Tensor images,
        Tensor labels,
        Device device,
        int batchSize)
    {
        model.eval();

        int correct = 0;
        int total = 0;

        using (torch.no_grad())
        {
            for (int i = 0; i < images.shape[0]; i += batchSize)
            {
                var end = Math.Min(i + batchSize, (int)images.shape[0]);
                var batchImages = images[i..end].to(device);
                var batchLabels = labels[i..end].to(device);

                var predictions = model.forward(batchImages);
                var predicted = predictions.argmax(dim: 1);

                correct += (predicted == batchLabels).sum().item<int>();
                total += (int)batchLabels.shape[0];

                batchImages.Dispose();
                batchLabels.Dispose();
                predictions.Dispose();
                predicted.Dispose();
            }
        }

        return (double)correct / total;
    }
}

// Entry point
var trainer = new MnistTrainer();
await trainer.RunAsync();
```

### Understanding the Architecture Choices

Let's break down why we made specific architectural decisions:

**Why 3x3 kernels?** Research has shown that stacking multiple 3x3 convolutions is more effective than using larger kernels. Two 3x3 layers have the same receptive field as one 5x5 layer but with fewer parameters and more non-linearity.

**Why increase channels (1 → 32 → 64)?** Early layers detect simple features (edges, textures) that are similar across images. Deeper layers detect complex, abstract features that vary more. More channels allow capturing more varied features where needed.

**Why padding=1?** With a 3x3 kernel and padding=1, the spatial dimensions stay the same after convolution. Size reduction comes from pooling, giving us control over where downsampling happens.

**Why batch normalization?** BatchNorm dramatically accelerates training by normalizing intermediate activations. It also has a mild regularizing effect, reducing the need for dropout in convolutional layers.

**Why dropout only before fully-connected layers?** Convolutional layers have fewer parameters (due to weight sharing) and are less prone to overfitting. The fully-connected layers have the most parameters and benefit most from dropout.

### Hyperparameter Tuning Tips

Getting good results often requires tuning hyperparameters. Here's a practical approach:

**Learning rate**: Start with 0.001 for Adam. If training is unstable (loss oscillates wildly), reduce it. If training is too slow, increase it. Learning rate is the most important hyperparameter to tune.

**Batch size**: Start with 32 or 64. Larger batches train faster but may need larger learning rates to converge. If you have memory issues, reduce batch size.

**Number of epochs**: Watch the validation loss. If it's still decreasing, train longer. If it starts increasing while training loss decreases, you're overfitting—stop earlier or add regularization.

```csharp
// Learning rate finder: train for a few batches at increasing learning rates
// and plot loss vs learning rate. Use the rate just before loss explodes.
var lrRates = new[] { 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 };
foreach (var lr in lrRates)
{
    var tempModel = new MnistCNN().to(device);
    var tempOpt = torch.optim.Adam(tempModel.parameters(), lr: lr);
    
    // Train for a few batches
    var loss = TrainFewBatches(tempModel, tempOpt, trainData, 10);
    Console.WriteLine($"LR: {lr:E1} -> Loss: {loss:F4}");
    
    tempModel.Dispose();
}
```

### Expected Results

Running this training pipeline, you should see output like:

```
Using device: cuda
Training samples: 60000
Test samples: 10000

Model parameters: 1,199,882

Epoch  1/10 | Loss: 0.2134 | Test Accuracy: 98.12%
Epoch  2/10 | Loss: 0.0645 | Test Accuracy: 98.67%
Epoch  3/10 | Loss: 0.0478 | Test Accuracy: 98.89%
Epoch  4/10 | Loss: 0.0372 | Test Accuracy: 99.01%
Epoch  5/10 | Loss: 0.0298 | Test Accuracy: 99.08%
Epoch  6/10 | Loss: 0.0251 | Test Accuracy: 99.15%
Epoch  7/10 | Loss: 0.0212 | Test Accuracy: 99.21%
Epoch  8/10 | Loss: 0.0178 | Test Accuracy: 99.18%
Epoch  9/10 | Loss: 0.0156 | Test Accuracy: 99.24%
Epoch 10/10 | Loss: 0.0132 | Test Accuracy: 99.27%

Training completed in 42.3s
Model saved to mnist_cnn.weights.dat
```

99%+ accuracy on MNIST is achievable with this relatively simple architecture. On GPU, training takes under a minute; on CPU, expect 5-10 minutes.

### Analyzing Model Performance

After training, it's valuable to understand where your model succeeds and fails:

```csharp
public void AnalyzeErrors(
    Module<Tensor, Tensor> model,
    Tensor images,
    Tensor labels,
    Device device)
{
    model.eval();
    var confusionMatrix = new int[10, 10];
    
    using (torch.no_grad())
    {
        for (int i = 0; i < images.shape[0]; i++)
        {
            var image = images[i].unsqueeze(0).to(device);
            var label = (int)labels[i].item<long>();
            
            var output = model.forward(image);
            var predicted = (int)output.argmax(dim: 1).item<long>();
            
            confusionMatrix[label, predicted]++;
            image.Dispose();
            output.Dispose();
        }
    }
    
    // Print confusion matrix
    Console.WriteLine("\nConfusion Matrix (rows=actual, cols=predicted):");
    Console.Write("   ");
    for (int i = 0; i < 10; i++) Console.Write($"{i,5}");
    Console.WriteLine();
    
    for (int i = 0; i < 10; i++)
    {
        Console.Write($"{i}: ");
        for (int j = 0; j < 10; j++)
        {
            Console.Write($"{confusionMatrix[i, j],5}");
        }
        Console.WriteLine();
    }
}
```

Common patterns in MNIST errors:
- 4 and 9 confusion (similar shapes)
- 3 and 8 confusion (similar curves)
- 7 and 1 confusion (depending on writing style)

Understanding these patterns helps you decide whether to improve the model (more data, better architecture) or accept the error rate as reasonable.

### Using the Trained Model

For inference in production:

```csharp
public class MnistInference
{
    private readonly MnistCNN _model;
    private readonly Device _device;

    public MnistInference(string modelPath)
    {
        _device = torch.cuda.is_available() 
            ? torch.device("cuda") 
            : torch.device("cpu");

        _model = new MnistCNN();
        _model.load(modelPath);
        _model.to(_device);
        _model.eval();
    }

    public int Predict(float[,] image28x28)
    {
        // Convert to tensor: (1, 1, 28, 28)
        var flatData = new float[28 * 28];
        Buffer.BlockCopy(image28x28, 0, flatData, 0, flatData.Length * sizeof(float));

        using var input = torch.tensor(flatData)
            .reshape(1, 1, 28, 28)
            .to(_device);

        using (torch.no_grad())
        {
            var output = _model.forward(input);
            var prediction = output.argmax(dim: 1);
            return (int)prediction.item<long>();
        }
    }

    public (int digit, float confidence) PredictWithConfidence(float[,] image28x28)
    {
        var flatData = new float[28 * 28];
        Buffer.BlockCopy(image28x28, 0, flatData, 0, flatData.Length * sizeof(float));

        using var input = torch.tensor(flatData)
            .reshape(1, 1, 28, 28)
            .to(_device);

        using (torch.no_grad())
        {
            var output = _model.forward(input);
            var probabilities = functional.softmax(output, dim: 1);

            var maxProb = probabilities.max(dim: 1);
            var digit = (int)maxProb.indexes.item<long>();
            var confidence = maxProb.values.item<float>();

            return (digit, confidence);
        }
    }
}

// Usage
var inference = new MnistInference("mnist_cnn.weights.dat");

// Assuming you have a 28x28 image as float array
float[,] myDigitImage = GetDigitImage();  // Your image loading code
var (digit, confidence) = inference.PredictWithConfidence(myDigitImage);

Console.WriteLine($"Predicted digit: {digit} (confidence: {confidence:P1})");
```

## Summary

In this chapter, you've crossed the threshold into deep learning:

- **Decision Framework**: You now know when deep learning is worth the complexity—unstructured data, large datasets, and problems where feature engineering is impractical
- **TorchSharp Fundamentals**: You can create tensors, build models, and leverage GPU acceleration
- **Neural Network Theory**: You understand layers, activations, backpropagation, and optimization—not just how to use them, but why they work
- **Practical Implementation**: You've built and trained a CNN that achieves 99%+ accuracy on image classification
- **Production Patterns**: Saving models, loading them for inference, and writing device-agnostic code

Deep learning is a powerful tool, but remember: it's still just a tool. The goal isn't to use neural networks everywhere—it's to recognize the problems where they excel and apply them effectively. For tabular data, gradient boosting often wins. For images, audio, and sequences, deep learning shines.

TorchSharp gives you PyTorch's capabilities with C#'s engineering strengths. You can train models, yes, but more importantly, you can deploy them reliably, integrate them with existing .NET systems, and maintain them as production software. That's where your skills as a C# developer create real value.

In the next chapter, we'll explore how to deploy models to production—containerization, API design, monitoring, and the operational concerns that determine whether your models create value or just consume compute.

## Exercises

1. **Architecture Experimentation**: Modify the CNN architecture to use 3 convolutional layers instead of 2. How does this affect training time, model size, and final accuracy? Try adding residual connections (skip connections) between layers.

2. **Data Augmentation**: Implement data augmentation by randomly rotating, scaling, and translating the training images. TorchSharp includes `torchvision.transforms` equivalents. Measure how augmentation affects the model's robustness to variations in test data.

3. **Learning Rate Scheduling**: Implement a learning rate scheduler that reduces the learning rate when validation loss plateaus. Compare training curves with constant learning rate vs. scheduled learning rate. Try implementing cosine annealing.

4. **Transfer Learning Preparation**: Export your trained MNIST model to ONNX format using `torch.onnx.export()`. Load it using ONNX Runtime and verify the predictions match. This is the foundation for deploying TorchSharp models in production .NET applications.

5. **Fashion-MNIST Challenge**: The Fashion-MNIST dataset (available from the same source with different URLs) contains images of clothing items instead of digits. Adapt your training pipeline to achieve >90% accuracy on Fashion-MNIST. What architectural changes improve performance on this harder dataset?

---

*Next Chapter: Deploying ML Models to Production →*
