# AI2C Assignment

--------------------

## Machine Learning Assignment: Predicting Extreme Events in Apple Stock Prices
### Objective
The goal of this assignment is to predict **extreme events** in Apple stock prices using machine learning techniques. An **extreme event** is defined as a **daily price movement exceeding 2%** in either direction (positive or negative) compared to the previous day.

You are required to:
1. Build a **Random Forest** classifier and a **Temporal Convolutional Neural Network (TCNN)** to predict extreme events using the previous 10 days of stock data.
2. Evaluate the models on test data and analyze their performance.
3. Suggest and implement improvements on the TCNN to enhance the models' performance.

---

### Assignment Tasks
1. **Data Preprocessing**: Load, preprocess, and engineer features from historical stock data.
2. **Random Forest Model**: Build and train a Random Forest classifier for extreme event prediction.
3. **Temporal CNN Model**: Build and train a Temporal CNN model for extreme event prediction.
4. **Model Evaluation**: Compare the performance of the models.
5. **Improvement Task**: Suggest and implement approaches to improve the performance of the TCNN model.

---

## Task 1: Data Preprocessing

### Step 1.1: Load the Data
Download **Apple stock data** (from January 2015 to January 2024) from [Yahoo Finance](https://finance.yahoo.com/) or use a similar source. The dataset should include the following columns:
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Adj Close` (Adjusted Close)

Load the dataset into a pandas DataFrame and inspect the first few rows to verify the data.

### Step 1.2: Calculate Daily Returns
Calculate the daily percentage return for Apple stock based on the adjusted closing price. This can be computed using:
- `Daily_Return = (Adj Close_t - Adj Close_t-1) / Adj Close_t-1 * 100`

Ensure that missing values in the dataset (due to holidays or weekends) are handled appropriately.

### Step 1.3: Define Extreme Events
Define an **extreme event** as any day where the percentage change in the adjusted close price exceeds **±2%**. Create a binary column `Extreme_Event` with the following logic:
- `1` if the daily return is greater than 2% or less than -2% (extreme event).
- `0` if the daily return is between -2% and 2% (no event).

Next, shift the target variable `Extreme_Event` by one day so that the model is trained to predict if an extreme event occurs **tomorrow** based on today's data.

### Step 1.4: Split Data into Features and Target
Extract the following columns as features for the model:
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Daily_Return`

The target variable will be the `Extreme_Event` column created earlier.

Split the data into **70% training**, **15% validation**, and **15% test** sets. Make sure to split the data in sequence to preserve the time-series nature of the stock data and avoid data leackage.

---

## Task 2: Random Forest Model

### Step 2.1: Model Training
You are required to build a **Random Forest Classifier** to predict extreme events using 10 days of historical stock data. You can use **scikit-learn** for this task.

- Train the Random Forest classifier using the training data (features from the past 10 days).
- Evaluate the model on the validation set to tune hyperparameters if necessary.

### Step 2.2: Model Evaluation
Evaluate the performance of the **Random Forest** model on the **test set**. Report the following metrics:
- **Confusion Matrix**
- **Accuracy**
- **Precision, Recall, F1-Score**

Explain whether the model is performing well in predicting extreme events and if there are any signs of overfitting or underfitting.

---

## Task 3: Temporal CNN Model

### Step 3.1: Input Preparation
For the **Temporal Convolutional Neural Network (TCNN)**, the input will consist of **sequences of 10 days** of stock data (features). The model will predict whether an extreme event will occur on the **next day** based on these sequences.

Prepare the data as follows:
- Convert the input data into sequences of 10 days, where each sequence is used to predict the target label for the next day.
- Ensure the input shape is suitable for a CNN: `[batch_size, num_features, sequence_length]`.

### Step 3.2: Model Architecture
Build a **Temporal CNN** model using PyTorch, with the following specifications:
- Two **1D Convolutional layers** to extract temporal features across the 10-day window.
- **ReLU** activations after each convolution layer.
- A fully connected **Dense layer** after flattening the features from the CNN layers.
- Use **Softmax** for the output layer to predict the probability of the two classes (extreme event vs no event).

### Step 3.3: Model Training
Train the **Temporal CNN** using the **Adam optimizer** and **Cross-Entropy Loss**. Monitor the performance on the validation set at the end of each epoch.

Ensure the training process handles potential issues like overfitting by using techniques such as:
- **Early stopping**
- **Dropout** layers

### Step 3.4: Model Evaluation
After training, evaluate the performance of the **Temporal CNN** model on the **test set**. Report the same metrics as for the Random Forest model:
- **Confusion Matrix**
- **Accuracy**
- **Precision, Recall, F1-Score**

---

## Task 4: Model Comparison

### Step 4.1: Compare the Models
Compare the performance of the **Random Forest** and **Temporal CNN** models based on the evaluation metrics. Address the following questions:
- Which model performs better for predicting extreme events?
- Which metric or metrics are more relevant for evaluating the performance of the methods?
- Why is forecasting of such events a challenging task? Name three reasons.
- How well do the models handle class imbalance (extreme events vs no extreme events)?
- Can you assess the predictability of the models based on their performance? Given the potentially low performance, would you say the models demonstrate predictive ability for extreme events in stock prices? Please explain your reasoning.

---

## Task 5: Improvement Task

### Step 5.1: Performance Improvement
The performance of the models is relatively low.
Propose and implement improvements to enhance the performance of the **Temporal CNN** model.

---
## Submission Instructions

Your submission should be organized in a folder containing all necessary files to ensure **reproducibility** and clear documentation of your work. The folder should be structured as follows:

### **Folder Structure**:

```
submission/
│
├── src/                 # Directory containing all Python executable scripts
│   ├── data_processing.py
│   ├── random_forest.py
│   ├── temporal_cnn.py
│   ├── model_evaluation.py
│   └── improvement.py
│
├── README.md            # Detailed instructions on how to run the code
│
├── pyproject.toml       # Poetry configuration file for dependency management
├── poetry.lock          # Poetry lock file for reproducibility
│
├── report.pdf           # A detailed report including model performance and analysis
│
└── data/                # Directory for any required dataset or files
```


## **Submission Components**:

1. **Python Executables** (`src/`):
   - The `src/` folder should contain **Python scripts** for each major task:
     - `data_processing.py`: Script to load, preprocess, and engineer features from the stock data.
     - `random_forest.py`: Script to build and train the Random Forest model.
     - `temporal_cnn.py`: Script to build and train the Temporal CNN model.
     - `model_evaluation.py`: Script to evaluate the models and generate performance metrics.
     - `improvement.py`: Script for any additional improvements made to enhance model performance.
   Each script should be modular and runnable as standalone or as part of an automated pipeline.

2. **README.md**:
   - Include a **comprehensive README** file that provides step-by-step instructions on how to set up the environment and run the code.
   - Clearly explain how to run each script in the `src/` folder, how to reproduce the results, and what dependencies are required.
   - The README should also specify any additional configuration or dataset download steps.

3. **Dependency Management**:
   - Use **Poetry** (or a similar dependency management tool) to handle all necessary dependencies.
     - Include the `pyproject.toml` file that defines the environment.
     - Include the `poetry.lock` file to ensure full reproducibility of the environment.
   - Ensure that all external libraries, dependencies, and versions are captured in these files to allow for seamless recreation of the development environment.

4. **Report (report.pdf)**:
   - Submit a **detailed report** in PDF format that includes:
     - Performance metrics (confusion matrix, precision, recall, F1-score) for both the Random Forest and Temporal CNN models.
     - A discussion of the results, model predictability, and potential areas for improvement.
     - An explanation of the improvement implemented in `improvement.py` and its impact on model performance.

5. **Reproducibility**:
   - The submission must be fully **reproducible**. Anyone with access to the submission should be able to:
     1. Set up the environment using the provided `pyproject.toml` and `poetry.lock`.
     2. Run the provided Python scripts and obtain the same results as presented in your report.

## **Important Notes**:
- Don't expect to completely solve the assignment and achieve very high performance scores, as forecasting stock price movements is a very challenging and difficult task. 
- The **accuracy of the results** is important, but equal emphasis will be placed on the clarity of your code, your ability to handle dependencies, and your documentation.
- **Reproducibility** is critical, so please ensure that all dependencies and code required to generate your results are included in the submission.
- While achieving good results is important, the emphasis of this assignment is on **critical and creative thinking**. A clear, thoughtful analysis with creative ideas will be valued over simply producing a high-accuracy model. In this respect, **Task 5** is the most important one.