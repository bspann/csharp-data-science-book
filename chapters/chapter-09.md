# Chapter 9: Classification — Predicting Categories

In Chapter 8, we explored regression—predicting continuous values like house prices or temperature. But many real-world problems don't involve numbers on a sliding scale. Instead, they require answers to questions like: *Will this customer cancel their subscription? Is this email spam? Which product category does this item belong to?* These are classification problems, and they're among the most common applications of machine learning.

Classification algorithms learn to assign observations to discrete categories based on their features. By the end of this chapter, you'll understand the mathematical foundations of classification, master multiple algorithms in ML.NET, and know exactly how to evaluate whether your classifier is actually working—even when the answer isn't as simple as "percentage correct."

## Binary vs. Multi-Class Classification

Classification problems come in two fundamental flavors, and understanding the distinction shapes everything from algorithm selection to evaluation strategy.

### Binary Classification

Binary classification involves exactly two possible outcomes. The target variable takes one of two values, typically encoded as 0/1, true/false, or positive/negative. Examples include:

- **Spam detection**: Email is spam or not spam
- **Fraud detection**: Transaction is fraudulent or legitimate
- **Medical diagnosis**: Patient has condition or doesn't
- **Customer churn**: Customer will leave or stay

Binary classification is conceptually simple but surprisingly nuanced. Most algorithms naturally produce binary classifiers, and the evaluation metrics we'll explore later are specifically designed for the binary case.

### Multi-Class Classification

Multi-class classification extends to three or more mutually exclusive categories. Only one label can apply to each observation:

- **Image recognition**: Classify images as cats, dogs, birds, or fish
- **Document categorization**: Assign articles to sports, politics, technology, or entertainment
- **Customer segmentation**: Group customers into bronze, silver, gold, or platinum tiers
- **Handwriting recognition**: Identify digits 0-9

Multi-class problems require different algorithmic approaches. Some algorithms handle multiple classes naturally, while others require decomposition strategies like "one-vs-all" (train a binary classifier for each class) or "one-vs-one" (train classifiers for every pair of classes).

### Multi-Label Classification

Don't confuse multi-class with multi-label classification. In multi-label problems, observations can belong to *multiple* categories simultaneously—a movie might be both "comedy" and "romance." We won't cover multi-label classification in depth here, but be aware the distinction exists.

## Logistic Regression: The Foundation

Despite its name, logistic regression is a classification algorithm, not a regression algorithm. It's the natural starting point for understanding classification because it's simple, interpretable, and surprisingly effective.

### The Problem with Linear Regression for Classification

Why not just use linear regression? Suppose you're predicting whether customers will churn (1) or stay (0). Linear regression would fit a line through your data:

```
Predicted = β₀ + β₁×tenure + β₂×monthlyCharges + ...
```

The problem? This produces continuous values that can fall anywhere—including below 0 or above 1. A prediction of -0.3 or 1.7 makes no sense for a probability. We need a function that constrains outputs to the range [0, 1].

[FIGURE: Linear regression vs. logistic regression for binary outcomes. Left panel shows linear regression producing predictions outside [0,1] range. Right panel shows logistic regression with S-shaped curve bounded between 0 and 1.]

### The Logistic Function

Logistic regression solves this problem using the *logistic function* (also called the sigmoid function):

```
σ(z) = 1 / (1 + e^(-z))
```

This S-shaped curve transforms any real number into a value between 0 and 1:

- When z is very negative, σ(z) approaches 0
- When z is very positive, σ(z) approaches 1
- When z = 0, σ(z) = 0.5

The model computes z as a linear combination of features (just like linear regression), then applies the logistic function:

```
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
P(y=1|x) = σ(z) = 1 / (1 + e^(-z))
```

The output represents the probability that the observation belongs to the positive class. To make a final classification, we typically apply a threshold (usually 0.5): if P(y=1|x) ≥ 0.5, predict class 1; otherwise, predict class 0.

### Interpreting Coefficients

Logistic regression coefficients have a specific interpretation. The coefficient βᵢ represents the change in *log-odds* for a one-unit increase in feature xᵢ:

```
log(P/(1-P)) = β₀ + β₁x₁ + β₂x₂ + ...
```

The quantity P/(1-P) is called the *odds ratio*. If P = 0.75, the odds are 0.75/0.25 = 3:1 in favor of the positive class. A coefficient of 0.5 means that increasing that feature by one unit multiplies the odds by e^0.5 ≈ 1.65.

Let's make this concrete with a churn example. Suppose our model learns these coefficients:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| Intercept | -2.5 | Baseline log-odds |
| Tenure (months) | -0.04 | Each month reduces churn odds by ~4% |
| Monthly Charges ($10s) | 0.15 | Each $10 increases churn odds by ~16% |
| Contract: Month-to-month | 1.2 | 3.3× higher odds vs. annual contract |
| Tech Support: No | 0.8 | 2.2× higher odds vs. having support |

This interpretability is logistic regression's superpower. You can explain exactly *why* a customer is flagged as high-risk: "New customer (low tenure), high monthly charges, month-to-month contract, no tech support—each factor contributes to the elevated churn probability."

### Training Logistic Regression

Logistic regression uses *maximum likelihood estimation* to find the best coefficients. The algorithm finds coefficients that maximize the probability of observing the actual training labels given the features.

Unlike linear regression's closed-form solution, logistic regression requires iterative optimization. Common algorithms include:

- **Gradient Descent**: Repeatedly update coefficients in the direction that increases likelihood
- **Stochastic Gradient Descent (SGD)**: Update using one example at a time for faster iteration
- **L-BFGS**: A sophisticated quasi-Newton method that converges quickly on moderately-sized datasets
- **SDCA**: Stochastic Dual Coordinate Ascent, efficient for large-scale problems

Regularization (L1 or L2 penalties on coefficient magnitudes) prevents overfitting and handles correlated features.

### Decision Boundaries

Logistic regression creates a *linear decision boundary*—a hyperplane that separates classes in feature space. In two dimensions, this boundary is a straight line. Points on one side are classified as positive; points on the other side as negative.

[FIGURE: Decision boundary for logistic regression in 2D feature space. Scattered points in two colors (churned/retained customers) with a straight diagonal line separating them. Shows some overlap near the boundary where predictions are uncertain.]

The location and orientation of this boundary depend on the learned coefficients. Features with larger coefficients have more influence on where the boundary falls.

## Decision Trees and Ensemble Methods

While logistic regression creates linear boundaries, many real-world problems require non-linear separation. Decision trees offer a fundamentally different approach—one that's intuitive, powerful, and forms the basis for some of the most effective modern algorithms.

### How Decision Trees Work

A decision tree makes predictions through a series of yes/no questions about feature values. Starting at the root, each internal node tests a condition (e.g., "Is tenure > 24 months?"). Based on the answer, the algorithm follows one branch to the next node. This continues until reaching a leaf node, which provides the final prediction.

```
                    [Monthly Charges > $70?]
                    /                      \
                  Yes                       No
                  /                          \
        [Tenure < 12 months?]          [Contract = Monthly?]
        /              \                /               \
      Yes              No             Yes               No
      /                 \              /                 \
   CHURN            RETAIN         CHURN             RETAIN
```

### Splitting Criteria

The key question is: how do we choose which feature to split on and what threshold to use? The algorithm evaluates every possible split and selects the one that best separates the classes. Common metrics include:

**Gini Impurity**: Measures how often a randomly chosen element would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset.

```
Gini = 1 - Σ(pᵢ)²
```

For a node with 70% class A and 30% class B: Gini = 1 - (0.7² + 0.3²) = 0.42

A pure node (100% one class) has Gini = 0. The algorithm chooses splits that minimize the weighted average Gini impurity of child nodes.

**Entropy/Information Gain**: Measures the expected amount of information needed to classify an element. Entropy is defined as:

```
Entropy = -Σ pᵢ × log₂(pᵢ)
```

A node with 50% class A and 50% class B has maximum entropy (most uncertain). A pure node has entropy = 0. Information gain is the reduction in entropy after splitting:

```
Information Gain = Entropy(parent) - Weighted Average Entropy(children)
```

Splits are chosen to maximize information gain.

### Tree Building Algorithm

The decision tree algorithm works recursively:

1. Start with all training examples at the root node
2. If all examples have the same label, create a leaf with that label
3. Otherwise, for each feature and each possible split point:
   - Calculate the impurity reduction (Gini or information gain)
   - Select the split with the highest reduction
4. Create child nodes and partition examples based on the split
5. Recursively apply steps 2-4 to each child
6. Stop when reaching maximum depth, minimum samples, or pure nodes

This greedy approach doesn't guarantee the globally optimal tree, but it's computationally efficient and works well in practice.

### Decision Boundaries in Trees

Unlike logistic regression's linear boundary, decision trees create *axis-aligned rectangular regions*. Each split divides feature space with a line parallel to one axis. The result is a staircase-like decision boundary that can approximate complex shapes.

[FIGURE: Decision boundary for a decision tree in 2D feature space. Shows rectangular regions created by axis-aligned splits, creating a staircase pattern that separates churned and retained customers. Contrast with the smooth linear boundary of logistic regression.]

### The Problem: Overfitting

Decision trees have a dangerous tendency: they can keep splitting until every training example is perfectly classified. This creates extremely complex trees that memorize the training data rather than learning generalizable patterns. Such trees perform terribly on new data.

Solutions include:

- **Pruning**: Remove branches that don't improve validation performance
- **Maximum depth**: Limit how deep the tree can grow
- **Minimum samples per leaf**: Require leaf nodes to contain multiple examples

But there's a better solution: combine many trees.

### Random Forests: The Power of Ensembles

Random forests build hundreds of decision trees and aggregate their predictions. This *ensemble approach* dramatically improves accuracy and reduces overfitting. The key innovations are:

**Bagging (Bootstrap Aggregating)**: Each tree trains on a random sample of the training data, drawn with replacement. Some examples appear multiple times; others don't appear at all. This introduces diversity among trees.

**Feature Randomization**: At each split, the algorithm considers only a random subset of features. This prevents all trees from making the same splits on dominant features, further increasing diversity.

**Voting**: For classification, each tree casts a vote for its predicted class. The forest's final prediction is the majority vote across all trees.

The magic is that while individual trees may overfit or make errors, their mistakes tend to cancel out when averaged. The ensemble is more robust than any single tree.

```csharp
// Random Forest conceptually
predictions = trees.Select(tree => tree.Predict(features));
finalPrediction = predictions.GroupBy(p => p)
                             .OrderByDescending(g => g.Count())
                             .First().Key;
```

**Why does this work?** Consider a forest of 100 trees where each tree is 70% accurate and makes independent errors. For a majority vote to be wrong, more than 50 trees must err on the same example. The probability of this is vanishingly small—far lower than the 30% error rate of a single tree.

Of course, tree errors aren't truly independent (they all see similar data), but the bagging and feature randomization introduce enough diversity to achieve significant improvement.

**Out-of-Bag Error Estimation**: Because each tree trains on a bootstrap sample, about 37% of examples are "out of bag" (OOB) for each tree. These excluded examples serve as a natural validation set. The OOB error—computed by averaging predictions only from trees that didn't see each example—provides an unbiased estimate of test error without needing a separate validation set.

**Feature Importance**: Random forests naturally compute feature importance. One common measure: for each tree, measure prediction accuracy on OOB samples; then shuffle each feature's values and measure the accuracy drop. Features that cause large drops when shuffled are important. This permutation importance is more reliable than single-tree importance measures.

### Other Ensemble Methods

**Gradient Boosting**: Instead of building trees independently, boosting builds trees sequentially. Each new tree focuses on correcting the errors of previous trees. The key insight: fit each new tree to the *residual errors* of the current ensemble.

Here's the gradient boosting process:

1. Start with a simple initial prediction (e.g., the class probability)
2. Calculate the residuals (errors) between predictions and actual values
3. Train a small decision tree to predict these residuals
4. Add this tree to the ensemble (scaled by a learning rate)
5. Recalculate residuals and repeat steps 3-4
6. Final prediction: sum of all trees' contributions

The "gradient" in gradient boosting refers to optimizing a loss function using gradient descent in function space—each tree represents a step toward better predictions.

Implementations like XGBoost, LightGBM, and CatBoost achieve state-of-the-art performance on tabular data. ML.NET's FastTree is a gradient boosting implementation, while LightGBM (available via NuGet) offers additional speed optimizations.

**Key hyperparameters for gradient boosting**:
- **Number of trees**: More trees = more capacity, but risk overfitting
- **Learning rate**: Smaller values require more trees but often generalize better
- **Tree depth**: Shallow trees (3-8 leaves) work best for boosting
- **Regularization**: L1/L2 penalties prevent overfitting

**Bagging with Other Base Learners**: While random forests use decision trees, the bagging principle applies to any algorithm. You could ensemble logistic regression models trained on different data subsets, though trees remain the most popular choice.

## Classification in ML.NET

ML.NET provides a comprehensive set of classification trainers. Understanding when to use each one is crucial for building effective models.

### Binary Classification Trainers

```csharp
using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 42);

// Define data classes
public class CustomerData
{
    [LoadColumn(0)] public float Tenure { get; set; }
    [LoadColumn(1)] public float MonthlyCharges { get; set; }
    [LoadColumn(2)] public float TotalCharges { get; set; }
    [LoadColumn(3)] public string Contract { get; set; }
    [LoadColumn(4)] public string InternetService { get; set; }
    [LoadColumn(5)] public bool Churn { get; set; }  // Label
}

public class ChurnPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
```

**Logistic Regression (SDCA)**:
```csharp
var trainer = mlContext.BinaryClassification.Trainers
    .SdcaLogisticRegression(
        labelColumnName: "Label",
        featureColumnName: "Features",
        maximumNumberOfIterations: 100);
```

Use when: You need interpretable coefficients, have linearly separable classes, or want a fast baseline. SDCA (Stochastic Dual Coordinate Ascent) is an efficient optimization algorithm.

**L-BFGS Logistic Regression**:
```csharp
var trainer = mlContext.BinaryClassification.Trainers
    .LbfgsLogisticRegression(
        labelColumnName: "Label",
        featureColumnName: "Features",
        l2Regularization: 0.1f);
```

Use when: You prefer batch optimization over stochastic methods. L-BFGS typically converges faster on smaller datasets.

**Fast Tree (Gradient Boosted Trees)**:
```csharp
var trainer = mlContext.BinaryClassification.Trainers
    .FastTree(
        labelColumnName: "Label",
        featureColumnName: "Features",
        numberOfLeaves: 20,
        numberOfTrees: 100,
        minimumExampleCountPerLeaf: 10);
```

Use when: You need high accuracy and can sacrifice interpretability. Fast Tree often outperforms linear methods on complex, non-linear relationships.

**Fast Forest (Random Forest)**:
```csharp
var trainer = mlContext.BinaryClassification.Trainers
    .FastForest(
        labelColumnName: "Label",
        featureColumnName: "Features",
        numberOfTrees: 100,
        numberOfLeaves: 20);
```

Use when: You want robust predictions with less risk of overfitting than a single tree. Random forests are excellent general-purpose classifiers.

**LightGBM**:
```csharp
var trainer = mlContext.BinaryClassification.Trainers
    .LightGbm(
        labelColumnName: "Label",
        featureColumnName: "Features",
        numberOfIterations: 100,
        LearningRate: 0.1);
```

Use when: You have large datasets and need fast training with high accuracy. LightGBM is one of the fastest gradient boosting implementations.

### Multi-Class Classification Trainers

For problems with more than two classes:

```csharp
// Multi-class trainers
var sdca = mlContext.MulticlassClassification.Trainers
    .SdcaMaximumEntropy();

var lbfgs = mlContext.MulticlassClassification.Trainers
    .LbfgsMaximumEntropy();

var oneVsAll = mlContext.MulticlassClassification.Trainers
    .OneVersusAll(
        mlContext.BinaryClassification.Trainers.FastTree());
```

The `OneVersusAll` wrapper transforms any binary classifier into a multi-class classifier by training one model per class.

### Algorithm Selection Guide

Choosing the right algorithm depends on your data characteristics and requirements:

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **Logistic Regression** | Linear relationships, interpretability needed | Fast, interpretable coefficients, works with few examples | Can't capture non-linear patterns |
| **Fast Tree** | Complex non-linear patterns | High accuracy, handles interactions | Can overfit, less interpretable |
| **Fast Forest** | General-purpose, robust needs | Resists overfitting, good defaults | Slower than single trees |
| **LightGBM** | Large datasets, best accuracy | State-of-the-art performance, fast | Many hyperparameters to tune |

**Start simple**: Logistic regression provides a baseline and interpretable insights. If performance is insufficient, move to tree-based methods.

**Feature engineering vs. algorithm complexity**: A well-engineered feature set with logistic regression often outperforms gradient boosting on raw features. Don't reach for complex algorithms before understanding your data.

### Building a Complete Pipeline

ML.NET pipelines combine data transformations with trainers:

```csharp
var pipeline = mlContext.Transforms.Conversion
        .MapValueToKey("Label", "Churn")
    .Append(mlContext.Transforms.Categorical
        .OneHotEncoding("ContractEncoded", "Contract"))
    .Append(mlContext.Transforms.Categorical
        .OneHotEncoding("InternetEncoded", "InternetService"))
    .Append(mlContext.Transforms.Concatenate("Features",
        "Tenure", "MonthlyCharges", "TotalCharges",
        "ContractEncoded", "InternetEncoded"))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"))
    .Append(mlContext.BinaryClassification.Trainers.FastForest())
    .Append(mlContext.Transforms.Conversion
        .MapKeyToValue("PredictedLabel"));

var model = pipeline.Fit(trainingData);
```

## Evaluation Metrics: Beyond Accuracy

Here's a critical lesson that separates amateur data scientists from professionals: **accuracy is often a terrible metric**. Understanding why—and what to use instead—is essential for building classification systems that actually work.

### The Confusion Matrix

Every classification metric derives from the confusion matrix, a 2×2 table (for binary classification) that shows exactly how your model succeeds and fails:

```
                        Actual Positive    Actual Negative
Predicted Positive          TP                  FP
Predicted Negative          FN                  TN
```

Let's define each quadrant:

**True Positives (TP)**: Model predicted positive, and it was actually positive. These are correct positive predictions. In churn prediction: customers the model said would churn, who actually did churn.

**True Negatives (TN)**: Model predicted negative, and it was actually negative. These are correct negative predictions. Customers the model said would stay, who actually stayed.

**False Positives (FP)**: Model predicted positive, but it was actually negative. Also called "Type I errors" or "false alarms." Customers the model said would churn, but who actually stayed. You might waste resources on retention efforts for loyal customers.

**False Negatives (FN)**: Model predicted negative, but it was actually positive. Also called "Type II errors" or "misses." Customers the model said would stay, but who actually churned. These are the expensive mistakes—you failed to intervene and lost the customer.

[FIGURE: Confusion matrix visualization for a churn classifier. 2×2 grid with cells colored green for correct predictions (TP=150, TN=720) and red for errors (FP=80, FN=50). Annotations explain what each cell means in business terms.]

```csharp
// Evaluating a binary classifier in ML.NET
var predictions = model.Transform(testData);
var metrics = mlContext.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"Confusion Matrix:");
Console.WriteLine($"  TP: {metrics.ConfusionMatrix.GetCountForSlot(0, 0)}");
Console.WriteLine($"  FP: {metrics.ConfusionMatrix.GetCountForSlot(0, 1)}");
Console.WriteLine($"  FN: {metrics.ConfusionMatrix.GetCountForSlot(1, 0)}");
Console.WriteLine($"  TN: {metrics.ConfusionMatrix.GetCountForSlot(1, 1)}");
```

### Understanding the Confusion Matrix Deeply

Let's work through a concrete example. Your churn model makes predictions on 1,000 test customers:

```
                        Actually Churned    Actually Stayed
Predicted Churn               150                80
Predicted Stay                 50               720
```

Reading this matrix:

- **TP = 150**: You correctly identified 150 customers who were going to churn. Your retention team can now intervene—maybe with discounts, better service, or personalized outreach.

- **TN = 720**: You correctly identified 720 customers who were going to stay. No action needed; these customers are happy.

- **FP = 80**: You predicted 80 customers would churn, but they were actually staying. These "false alarms" mean wasted retention resources—you might give discounts to customers who would have stayed anyway. This costs money but doesn't lose customers.

- **FN = 50**: You predicted 50 customers would stay, but they actually churned. These are your most expensive mistakes—customers you could have saved if you'd identified them. They're now gone to competitors.

The business impact of each cell differs dramatically. In churn prediction, FN (missed churners) typically costs far more than FP (unnecessary retention efforts). This asymmetry should drive your threshold choice and metric selection.

### Accuracy and Its Limitations

Accuracy is the proportion of correct predictions:

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Simple and intuitive—but dangerously misleading with imbalanced classes.

**The Imbalanced Class Problem**

Imagine a fraud detection system where only 1% of transactions are fraudulent. A model that predicts "not fraud" for everything achieves 99% accuracy! It's completely useless—it catches zero actual fraud—yet the accuracy number looks impressive.

This is the *accuracy paradox*. When classes are imbalanced, accuracy reflects the majority class and hides poor performance on the minority class you actually care about.

In churn prediction, if 80% of customers stay, a "predict everyone stays" model achieves 80% accuracy while providing zero business value.

### Precision: When False Positives Are Costly

Precision answers: *Of all positive predictions, how many were actually positive?*

```
Precision = TP / (TP + FP)
```

High precision means few false positives. The model rarely cries wolf.

When to prioritize precision:
- **Email marketing**: False positives mean annoying loyal customers with unnecessary retention offers
- **Spam detection**: False positives send legitimate emails to spam folders
- **Criminal justice**: False positives mean wrongful accusations

### Recall (Sensitivity): When False Negatives Are Costly

Recall answers: *Of all actual positives, how many did we correctly identify?*

```
Recall = TP / (TP + FN)
```

High recall means few false negatives. The model catches most positive cases.

When to prioritize recall:
- **Cancer screening**: Missing cancer (false negative) is far worse than additional testing (false positive)
- **Fraud detection**: Missing fraud costs money; investigating legitimate transactions is a minor inconvenience
- **Churn prediction**: Missing churners means losing customers; extra retention efforts on stayers wastes some budget but keeps everyone happy

### The Precision-Recall Trade-off

You can't maximize both simultaneously. They're inversely related through the classification threshold.

```csharp
// ML.NET provides probability scores, not just predictions
var predictionEngine = mlContext.Model
    .CreatePredictionEngine<CustomerData, ChurnPrediction>(model);

var result = predictionEngine.Predict(customer);
Console.WriteLine($"Probability of churn: {result.Probability:P2}");

// Default threshold is 0.5, but you can adjust
bool wilChurn = result.Probability >= 0.3;  // Lower threshold = higher recall
```

Lowering the threshold (classifying more cases as positive):
- **Increases recall**: Catches more true positives
- **Decreases precision**: Also catches more false positives

Raising the threshold (being more conservative about positive predictions):
- **Increases precision**: Fewer false positives
- **Decreases recall**: Misses more true positives

[FIGURE: Precision-recall trade-off curve. X-axis shows recall (0 to 1), Y-axis shows precision (0 to 1). Curve starts high on precision axis and decreases as recall increases. Multiple points marked at different thresholds (0.3, 0.5, 0.7) showing the trade-off.]

### F1 Score: Balancing Precision and Recall

The F1 score is the harmonic mean of precision and recall:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 ranges from 0 to 1, where 1 is perfect. The harmonic mean penalizes extreme imbalances—you can't get a high F1 by maximizing one metric while ignoring the other.

Use F1 when:
- You need a single number to compare models
- False positives and false negatives have roughly equal costs
- Classes are imbalanced (unlike accuracy, F1 handles this well)

### Specificity and the Full Picture

Specificity (also called True Negative Rate) measures performance on the negative class:

```
Specificity = TN / (TN + FP)
```

Combined with recall (sensitivity), you have a complete picture:
- Recall: How well do we find positives?
- Specificity: How well do we identify negatives?

### ROC Curves and AUC

The Receiver Operating Characteristic (ROC) curve visualizes the trade-off between recall (True Positive Rate) and False Positive Rate across all possible thresholds.

```
False Positive Rate = FP / (FP + TN) = 1 - Specificity
```

[FIGURE: ROC curve for churn classifier. X-axis shows False Positive Rate (0 to 1), Y-axis shows True Positive Rate (0 to 1). Curved line from (0,0) to (1,1) bowing toward the upper-left corner. Diagonal dashed line shows random classifier performance. Shaded area under curve labeled "AUC = 0.87".]

**Area Under the Curve (AUC)** summarizes the ROC curve as a single number:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing (the diagonal line)
- AUC < 0.5: Worse than random (your labels might be inverted!)

AUC has a beautiful interpretation: it's the probability that a randomly chosen positive example ranks higher (has a higher predicted probability) than a randomly chosen negative example.

```csharp
var metrics = mlContext.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
Console.WriteLine($"Precision: {metrics.PositivePrecision:P2}");
Console.WriteLine($"Recall: {metrics.PositiveRecall:P2}");
```

### Choosing the Right Metric

| Scenario | Primary Metric | Reason |
|----------|---------------|--------|
| Balanced classes, general purpose | Accuracy | Simple and interpretable |
| Imbalanced classes | F1 or AUC | Accuracy is misleading |
| False negatives are costly | Recall | Must catch positives |
| False positives are costly | Precision | Must avoid false alarms |
| Comparing models, threshold unknown | AUC | Threshold-independent |
| Need single ranking metric | F1 or AUC | Balances competing concerns |

## Project: Customer Churn Prediction

Let's build a complete churn prediction system for a telecommunications company. We'll handle imbalanced classes, compare multiple classifiers, and properly evaluate using the metrics we've learned.

### The Dataset

Our telecom dataset includes customer demographics, account information, and service usage:

```csharp
public class TelecomCustomer
{
    [LoadColumn(0)] public string CustomerId { get; set; }
    [LoadColumn(1)] public string Gender { get; set; }
    [LoadColumn(2)] public float SeniorCitizen { get; set; }
    [LoadColumn(3)] public string Partner { get; set; }
    [LoadColumn(4)] public string Dependents { get; set; }
    [LoadColumn(5)] public float Tenure { get; set; }
    [LoadColumn(6)] public string PhoneService { get; set; }
    [LoadColumn(7)] public string MultipleLines { get; set; }
    [LoadColumn(8)] public string InternetService { get; set; }
    [LoadColumn(9)] public string OnlineSecurity { get; set; }
    [LoadColumn(10)] public string OnlineBackup { get; set; }
    [LoadColumn(11)] public string DeviceProtection { get; set; }
    [LoadColumn(12)] public string TechSupport { get; set; }
    [LoadColumn(13)] public string StreamingTV { get; set; }
    [LoadColumn(14)] public string StreamingMovies { get; set; }
    [LoadColumn(15)] public string Contract { get; set; }
    [LoadColumn(16)] public string PaperlessBilling { get; set; }
    [LoadColumn(17)] public string PaymentMethod { get; set; }
    [LoadColumn(18)] public float MonthlyCharges { get; set; }
    [LoadColumn(19)] public float TotalCharges { get; set; }
    [LoadColumn(20)] public bool Churn { get; set; }
}

public class ChurnPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
```

### Exploratory Data Analysis

Before modeling, understand your data:

```csharp
var mlContext = new MLContext(seed: 42);

// Load data
var dataView = mlContext.Data.LoadFromTextFile<TelecomCustomer>(
    "telco-churn.csv",
    hasHeader: true,
    separatorChar: ',');

// Check class distribution
var data = mlContext.Data.CreateEnumerable<TelecomCustomer>(
    dataView, reuseRowObject: false).ToList();

var churnCount = data.Count(c => c.Churn);
var retainCount = data.Count(c => !c.Churn);

Console.WriteLine($"Total customers: {data.Count}");
Console.WriteLine($"Churned: {churnCount} ({100.0 * churnCount / data.Count:F1}%)");
Console.WriteLine($"Retained: {retainCount} ({100.0 * retainCount / data.Count:F1}%)");
```

Output:
```
Total customers: 7043
Churned: 1869 (26.5%)
Retained: 5174 (73.5%)
```

The classes are imbalanced—roughly 3:1 in favor of retained customers. This isn't extreme, but we should still account for it.

### Handling Class Imbalance

Several strategies address imbalanced classes:

**1. Class Weights**: Tell the algorithm to penalize errors on the minority class more heavily.

```csharp
// Some trainers support example weights
var trainer = mlContext.BinaryClassification.Trainers
    .FastTree(
        exampleWeightColumnName: "Weight");
```

**2. Oversampling**: Duplicate minority class examples or generate synthetic examples (SMOTE).

```csharp
// Simple oversampling - duplicate minority class
var churners = data.Where(c => c.Churn).ToList();
var oversampledData = data.Concat(churners).Concat(churners).ToList();
// Now ~40% churn instead of 26.5%
```

**3. Undersampling**: Remove majority class examples to balance classes.

```csharp
var churners = data.Where(c => c.Churn).ToList();
var retained = data.Where(c => !c.Churn)
    .OrderBy(_ => Random.Shared.Next())
    .Take(churners.Count)
    .ToList();
var balancedData = churners.Concat(retained).ToList();
```

**4. Threshold Adjustment**: Keep the model as-is but adjust the classification threshold based on class proportions.

For our project, we'll use a combination: train on the original data but evaluate with metrics appropriate for imbalanced classes (F1, AUC) and adjust thresholds based on business needs.

### Building the Feature Pipeline

```csharp
// Categorical columns to encode
string[] categoricalColumns = {
    "Gender", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
};

// Build encoding pipeline
var encodingPipeline = mlContext.Transforms.Conversion
    .MapValueToKey("Label", "Churn");

foreach (var column in categoricalColumns)
{
    encodingPipeline = encodingPipeline
        .Append(mlContext.Transforms.Categorical
            .OneHotEncoding($"{column}Encoded", column));
}

// Combine all features
var encodedCategorical = categoricalColumns
    .Select(c => $"{c}Encoded").ToArray();
var numericFeatures = new[] { "SeniorCitizen", "Tenure", 
                               "MonthlyCharges", "TotalCharges" };
var allFeatures = numericFeatures.Concat(encodedCategorical).ToArray();

var featurePipeline = encodingPipeline
    .Append(mlContext.Transforms.ReplaceMissingValues(
        "TotalCharges", replacementMode: 
        MissingValueReplacingEstimator.ReplacementMode.Mean))
    .Append(mlContext.Transforms.Concatenate("Features", allFeatures))
    .Append(mlContext.Transforms.NormalizeMinMax("Features"));
```

### Comparing Multiple Classifiers

```csharp
// Split data
var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

// Define trainers to compare
var trainers = new Dictionary<string, IEstimator<ITransformer>>
{
    ["Logistic Regression"] = mlContext.BinaryClassification.Trainers
        .SdcaLogisticRegression(maximumNumberOfIterations: 100),
    
    ["Fast Tree"] = mlContext.BinaryClassification.Trainers
        .FastTree(numberOfTrees: 100, numberOfLeaves: 20),
    
    ["Fast Forest"] = mlContext.BinaryClassification.Trainers
        .FastForest(numberOfTrees: 100, numberOfLeaves: 20),
    
    ["LightGBM"] = mlContext.BinaryClassification.Trainers
        .LightGbm(numberOfIterations: 100)
};

// Train and evaluate each
var results = new List<(string Name, BinaryClassificationMetrics Metrics)>();

foreach (var (name, trainer) in trainers)
{
    Console.WriteLine($"\nTraining {name}...");
    
    var pipeline = featurePipeline.Append(trainer);
    var model = pipeline.Fit(split.TrainSet);
    var predictions = model.Transform(split.TestSet);
    var metrics = mlContext.BinaryClassification.Evaluate(predictions);
    
    results.Add((name, metrics));
    
    Console.WriteLine($"  Accuracy:  {metrics.Accuracy:P2}");
    Console.WriteLine($"  AUC:       {metrics.AreaUnderRocCurve:F4}");
    Console.WriteLine($"  F1:        {metrics.F1Score:F4}");
    Console.WriteLine($"  Precision: {metrics.PositivePrecision:P2}");
    Console.WriteLine($"  Recall:    {metrics.PositiveRecall:P2}");
}
```

Output:
```
Training Logistic Regression...
  Accuracy:  80.21%
  AUC:       0.8432
  F1:        0.5847
  Precision: 66.12%
  Recall:    52.41%

Training Fast Tree...
  Accuracy:  79.87%
  AUC:       0.8389
  F1:        0.5692
  Precision: 64.83%
  Recall:    50.80%

Training Fast Forest...
  Accuracy:  79.52%
  AUC:       0.8456
  F1:        0.5521
  Precision: 65.18%
  Recall:    47.86%

Training LightGBM...
  Accuracy:  80.85%
  AUC:       0.8521
  F1:        0.6012
  Precision: 67.45%
  Recall:    54.28%
```

### Analyzing Results

LightGBM achieves the best performance across most metrics. But look beyond the single numbers:

- **Accuracy (~80%)** looks reasonable but remember: a "predict everyone stays" model would achieve 73.5% accuracy
- **Recall (~54%)** means we're catching only about half of churners—we're missing nearly half of at-risk customers
- **AUC (0.85)** indicates good overall discrimination ability

For a business application where missing churners is expensive, we might want to lower the threshold to increase recall, accepting more false positives.

### ROC Curve Analysis

Let's visualize the ROC curves to understand model behavior across all thresholds:

```csharp
// Generate ROC curve data points
public static List<(double FPR, double TPR)> ComputeRocCurve(
    List<float> probabilities, 
    List<bool> actualLabels)
{
    var rocPoints = new List<(double FPR, double TPR)>();
    
    // Sort by probability descending
    var sorted = probabilities
        .Zip(actualLabels, (prob, label) => (prob, label))
        .OrderByDescending(x => x.prob)
        .ToList();
    
    int totalPositives = actualLabels.Count(l => l);
    int totalNegatives = actualLabels.Count(l => !l);
    
    int tp = 0, fp = 0;
    float prevProb = float.MaxValue;
    
    foreach (var (prob, label) in sorted)
    {
        if (prob != prevProb)
        {
            rocPoints.Add((
                FPR: (double)fp / totalNegatives,
                TPR: (double)tp / totalPositives
            ));
            prevProb = prob;
        }
        
        if (label) tp++;
        else fp++;
    }
    
    rocPoints.Add((1.0, 1.0));  // Final point
    return rocPoints;
}

// Output ROC data for plotting
var rocCurve = ComputeRocCurve(probabilities, actualLabels);
Console.WriteLine("\nROC Curve Data (for visualization):");
Console.WriteLine("FPR,TPR");
foreach (var (fpr, tpr) in rocCurve.Where((_, i) => i % 10 == 0))
{
    Console.WriteLine($"{fpr:F3},{tpr:F3}");
}
```

[FIGURE: ROC curves comparing all four classifiers. LightGBM curve (AUC=0.852) bows highest toward upper-left, followed closely by Fast Forest (0.846) and Logistic Regression (0.843). Fast Tree (0.839) shows slightly worse performance. All substantially outperform the diagonal random baseline.]

The ROC curve reveals that all models perform similarly—the differences in AUC are small. This suggests the features themselves limit performance more than algorithm choice. Feature engineering (Chapter 11) might yield bigger gains than hyperparameter tuning.

### Precision-Recall Curves

For imbalanced datasets, precision-recall curves often provide more insight than ROC curves:

```csharp
public static List<(double Recall, double Precision)> ComputePRCurve(
    List<float> probabilities,
    List<bool> actualLabels)
{
    var prPoints = new List<(double Recall, double Precision)>();
    
    var sorted = probabilities
        .Zip(actualLabels, (prob, label) => (prob, label))
        .OrderByDescending(x => x.prob)
        .ToList();
    
    int totalPositives = actualLabels.Count(l => l);
    int tp = 0, fp = 0;
    
    foreach (var (prob, label) in sorted)
    {
        if (label) tp++;
        else fp++;
        
        double precision = (double)tp / (tp + fp);
        double recall = (double)tp / totalPositives;
        prPoints.Add((recall, precision));
    }
    
    return prPoints;
}
```

[FIGURE: Precision-Recall curves for the churn classifiers. Shows how precision degrades as we increase recall (catch more churners). Horizontal dashed line at 26.5% represents the baseline (class proportion). All models significantly outperform baseline, with LightGBM maintaining highest precision at each recall level.]

The PR curve shows what matters for business: to catch 80% of churners, we accept precision dropping to about 50%—meaning half our intervention targets are false alarms. Whether that's acceptable depends on the relative costs.

### Threshold Optimization

```csharp
// Get probability predictions
var predictions = model.Transform(split.TestSet);
var scoredData = mlContext.Data
    .CreateEnumerable<ChurnPrediction>(predictions, reuseRowObject: false)
    .ToList();
var actualLabels = mlContext.Data
    .CreateEnumerable<TelecomCustomer>(split.TestSet, reuseRowObject: false)
    .Select(c => c.Churn)
    .ToList();

// Evaluate different thresholds
Console.WriteLine("\nThreshold Analysis:");
Console.WriteLine("Threshold | Precision | Recall | F1");
Console.WriteLine("----------|-----------|--------|-------");

foreach (var threshold in new[] { 0.3, 0.4, 0.5, 0.6, 0.7 })
{
    var predicted = scoredData.Select(p => p.Probability >= threshold).ToList();
    
    int tp = 0, fp = 0, fn = 0, tn = 0;
    for (int i = 0; i < predicted.Count; i++)
    {
        if (predicted[i] && actualLabels[i]) tp++;
        else if (predicted[i] && !actualLabels[i]) fp++;
        else if (!predicted[i] && actualLabels[i]) fn++;
        else tn++;
    }
    
    var precision = tp / (double)(tp + fp);
    var recall = tp / (double)(tp + fn);
    var f1 = 2 * precision * recall / (precision + recall);
    
    Console.WriteLine($"   {threshold:F1}    |   {precision:P1}   | {recall:P1} | {f1:F3}");
}
```

Output:
```
Threshold Analysis:
Threshold | Precision | Recall | F1
----------|-----------|--------|-------
   0.3    |   49.2%   | 78.1%  | 0.603
   0.4    |   58.7%   | 67.4%  | 0.627
   0.5    |   67.5%   | 54.3%  | 0.601
   0.6    |   74.8%   | 41.2%  | 0.531
   0.7    |   81.3%   | 28.6%  | 0.423
```

A threshold of 0.4 maximizes F1 while achieving a good balance. At 0.3, we catch 78% of churners—much better for the business, though we'll have more false alarms.

### Feature Importance Analysis

Understanding which features drive predictions provides actionable insights:

```csharp
// For tree-based models, extract feature importance
var lastTransformer = model.LastTransformer as 
    ISingleFeaturePredictionTransformer<object>;

if (model.LastTransformer is FastTreeBinaryModelParameters treeParams)
{
    var featureImportance = treeParams
        .GetFeatureWeights()
        .Select((weight, index) => (Index: index, Weight: Math.Abs(weight)))
        .OrderByDescending(x => x.Weight)
        .Take(10);
    
    Console.WriteLine("\nTop 10 Features by Importance:");
    foreach (var (index, weight) in featureImportance)
    {
        Console.WriteLine($"  Feature {index}: {weight:F4}");
    }
}
```

For churn prediction, typical important features include:
- **Contract type**: Month-to-month customers churn far more than those with annual contracts
- **Tenure**: New customers are at highest risk
- **Monthly charges**: Higher charges correlate with higher churn
- **Internet service**: Fiber optic customers (often with bundled services) show different patterns
- **Tech support**: Customers without tech support churn more frequently

### Making Predictions on New Customers

```csharp
// Create prediction engine
var predictionEngine = mlContext.Model
    .CreatePredictionEngine<TelecomCustomer, ChurnPrediction>(model);

// Score a new customer
var newCustomer = new TelecomCustomer
{
    Gender = "Male",
    SeniorCitizen = 0,
    Partner = "No",
    Dependents = "No",
    Tenure = 2,  // New customer
    PhoneService = "Yes",
    MultipleLines = "No",
    InternetService = "Fiber optic",
    OnlineSecurity = "No",
    OnlineBackup = "No",
    DeviceProtection = "No",
    TechSupport = "No",
    StreamingTV = "No",
    StreamingMovies = "No",
    Contract = "Month-to-month",  // High risk
    PaperlessBilling = "Yes",
    PaymentMethod = "Electronic check",
    MonthlyCharges = 70.35f,
    TotalCharges = 140.70f
};

var prediction = predictionEngine.Predict(newCustomer);

Console.WriteLine($"\nNew Customer Analysis:");
Console.WriteLine($"  Churn Probability: {prediction.Probability:P1}");
Console.WriteLine($"  Risk Level: {(prediction.Probability > 0.5 ? "HIGH" : 
                                    prediction.Probability > 0.3 ? "MEDIUM" : "LOW")}");
Console.WriteLine($"  Recommendation: {(prediction.Probability > 0.4 ? 
    "Proactive retention outreach recommended" : 
    "Standard engagement")}");
```

Output:
```
New Customer Analysis:
  Churn Probability: 72.3%
  Risk Level: HIGH
  Recommendation: Proactive retention outreach recommended
```

This customer—new, month-to-month, fiber optic with no add-on services—matches the classic high-churn profile.

### Cross-Validation for Robust Evaluation

A single train-test split can be misleading—performance depends on which examples land in which set. Cross-validation provides more robust estimates:

```csharp
// 5-fold cross-validation
var cvResults = mlContext.BinaryClassification.CrossValidate(
    dataView,
    featurePipeline.Append(mlContext.BinaryClassification.Trainers.LightGbm()),
    numberOfFolds: 5);

// Aggregate results
var aucScores = cvResults.Select(r => r.Metrics.AreaUnderRocCurve).ToList();
var f1Scores = cvResults.Select(r => r.Metrics.F1Score).ToList();

Console.WriteLine($"\n5-Fold Cross-Validation Results:");
Console.WriteLine($"  AUC: {aucScores.Average():F4} ± {StandardDeviation(aucScores):F4}");
Console.WriteLine($"  F1:  {f1Scores.Average():F4} ± {StandardDeviation(f1Scores):F4}");

static double StandardDeviation(List<double> values)
{
    var avg = values.Average();
    var sumSquares = values.Sum(v => Math.Pow(v - avg, 2));
    return Math.Sqrt(sumSquares / values.Count);
}
```

Output:
```
5-Fold Cross-Validation Results:
  AUC: 0.8489 ± 0.0124
  F1:  0.5923 ± 0.0287
```

The small standard deviations indicate consistent performance across different data subsets—a good sign that our results will generalize.

### Business Impact Analysis

Converting model metrics to business value helps stakeholders understand ROI:

```csharp
// Business parameters (adjust based on your domain)
const decimal customerLifetimeValue = 5000m;  // Revenue from retained customer
const decimal retentionCampaignCost = 50m;     // Cost per intervention
const decimal churnProbabilityReduction = 0.25m; // Success rate of intervention

// Calculate expected value at different thresholds
var businessMetrics = new List<(double Threshold, decimal ExpectedValue)>();

foreach (var threshold in Enumerable.Range(1, 9).Select(i => i / 10.0))
{
    int tp = 0, fp = 0, fn = 0;
    
    for (int i = 0; i < predictions.Count; i++)
    {
        bool predicted = predictions[i].Probability >= threshold;
        if (predicted && actualLabels[i]) tp++;
        else if (predicted && !actualLabels[i]) fp++;
        else if (!predicted && actualLabels[i]) fn++;
    }
    
    // Revenue from saved customers
    decimal savedCustomers = tp * churnProbabilityReduction;
    decimal revenue = savedCustomers * customerLifetimeValue;
    
    // Cost of interventions (true positives + false positives)
    decimal cost = (tp + fp) * retentionCampaignCost;
    
    // Lost customers (not identified)
    decimal lostRevenue = fn * customerLifetimeValue * churnProbabilityReduction;
    
    decimal netValue = revenue - cost - lostRevenue;
    businessMetrics.Add((threshold, netValue));
}

var optimal = businessMetrics.OrderByDescending(m => m.ExpectedValue).First();
Console.WriteLine($"\nOptimal threshold for business value: {optimal.Threshold:F1}");
Console.WriteLine($"Expected net value: ${optimal.ExpectedValue:N0}");
```

This analysis often reveals that the optimal business threshold differs significantly from the default 0.5. When customer lifetime value far exceeds intervention cost, aggressive (low) thresholds maximize value despite many false positives.

### Saving and Loading Models

```csharp
// Save the trained model
var modelPath = "ChurnModel.zip";
mlContext.Model.Save(model, dataView.Schema, modelPath);
Console.WriteLine($"Model saved to {modelPath}");

// Load for production use
var loadedModel = mlContext.Model.Load(modelPath, out var schema);
var productionEngine = mlContext.Model
    .CreatePredictionEngine<TelecomCustomer, ChurnPrediction>(loadedModel);
```

### Deploying to Production

For production deployment, consider:

```csharp
// Batch scoring for periodic analysis
public IEnumerable<(string CustomerId, float ChurnProbability, string RiskLevel)> 
    ScoreAllCustomers(IDataView customers)
{
    var predictions = model.Transform(customers);
    var results = mlContext.Data
        .CreateEnumerable<ChurnPrediction>(predictions, reuseRowObject: false);
    
    return results.Select(p => (
        p.CustomerId,
        p.Probability,
        RiskLevel: p.Probability > 0.6 ? "HIGH" :
                   p.Probability > 0.35 ? "MEDIUM" : "LOW"
    ));
}

// Real-time scoring for customer service
public async Task<ChurnRiskAssessment> AssessRiskAsync(string customerId)
{
    var customer = await _customerRepository.GetAsync(customerId);
    var prediction = _predictionEngine.Predict(customer);
    
    return new ChurnRiskAssessment
    {
        CustomerId = customerId,
        ChurnProbability = prediction.Probability,
        RiskLevel = ClassifyRisk(prediction.Probability),
        RecommendedActions = GetRecommendations(prediction.Probability),
        TopRiskFactors = await GetRiskFactors(customer)
    };
}
```

## Summary

Classification is the workhorse of machine learning, powering everything from spam filters to medical diagnosis. In this chapter, you learned:

- **Binary vs. multi-class** classification and when each applies
- **Logistic regression** as the foundational classifier, using the sigmoid function to produce probabilities
- **Decision trees** and how they create non-linear decision boundaries through recursive splitting
- **Ensemble methods** like random forests and gradient boosting that combine many weak learners into powerful predictors
- **ML.NET's classification trainers** and when to choose each
- **Evaluation metrics** beyond accuracy—precision, recall, F1, and AUC—and why accuracy fails with imbalanced classes
- **The confusion matrix** as the source of all classification metrics
- **ROC curves** for visualizing classifier performance across thresholds

The customer churn project demonstrated these concepts in practice: handling imbalanced classes, comparing multiple algorithms, analyzing feature importance, and optimizing thresholds based on business requirements.

In Chapter 10, we'll explore clustering—unsupervised learning where we discover structure in data without predefined labels. You'll learn k-means, hierarchical clustering, and how to evaluate clusters when there's no "right answer" to compare against.

---

**Exercises**

1. Modify the churn project to implement SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance. Compare results to the original imbalanced training.

2. Add cross-validation to the classifier comparison. Use 5-fold CV and report mean and standard deviation of AUC across folds.

3. Build a multi-class classifier that predicts customer segments (low-value, medium-value, high-value) based on their usage patterns and tenure.

4. Implement a cost-sensitive threshold optimization: if losing a customer costs $500 and a retention campaign costs $50, what threshold minimizes expected cost?

5. Create a feature engineering pipeline that adds interaction terms (e.g., tenure × monthly charges) and evaluate whether they improve model performance.
