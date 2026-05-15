# Customer Churn Neural Network Analysis

This project analyzes a customer churn dataset using a neural network built with TensorFlow Keras. The notebook walks through dataset understanding, preprocessing, model building, training, evaluation, and hyperparameter experimentation.

## Dataset

- File: `customer_churn_nn.csv`
- Target variable: `churn` (binary)
- No missing values present in the dataset
- Strong class imbalance: most customers are retained, with a small percentage churned

## Notebook Workflow

### 1. Dataset Understanding
- Loaded the dataset with `pandas`
- Reported row/column counts and feature datatypes
- Checked for missing values and computed summary statistics
- Examined the distribution of the `churn` target

### 2. Data Preprocessing
- Removed the `customer_id` identifier column
- Applied label encoding for ordered categorical features:
  - `plan_type`
  - `contract_type`
- Applied one-hot encoding for nominal categorical features:
  - `region`
  - `payment_method`
- Scaled numerical features using `StandardScaler`
- Split data into training and testing sets with an 80/20 split and stratified by the churn target

### 3. Neural Network Model Building
- Built a `Sequential` neural network using `tensorflow.keras`
- Used dense hidden layers with ReLU activation
- Used a sigmoid activation for the binary output layer
- Compiled the model with the Adam optimizer and binary cross-entropy loss

### 4. Training and Evaluation
- Trained the model with validation split
- Evaluated performance on the test set
- Generated predictions and a confusion matrix for model evaluation

### 5. Hyperparameter Experimentation
- Explored multiple model variants with different layer sizes, activations, epochs, and batch sizes
- Compared test accuracy across experiments to identify the best-performing configuration

### 6. Final Reflection
- Discussed the role of weights and biases in the model
- Explained the need for activation functions and the effect of learning rate settings
- Assessed whether the model showed signs of underfitting or overfitting

## Requirements
- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn`

## Files
- `notebook.ipynb`: analysis notebook
- `customer_churn_nn.csv`: dataset
- `requirements.txt`: required Python packages
