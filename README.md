# 📈 Stock Price Movement Predictor

A time-series machine learning project that predicts the **next-day price direction (Up or Down)** of Apple Inc. (AAPL) using historical daily OHLCV data and technical indicators.

## 🎯 Goal

The goal of this project is to evaluate whether historical price information and engineered technical indicators can help predict the direction of the next trading day's closing price.

The project focuses on:

* Correctly constructing a next-day prediction target without data leakage
* Engineering meaningful time-series features
* Comparing machine learning models with naive baselines
* Using a chronological train-test split
* Evaluating model performance on unseen future data
* Interpreting the results honestly

---

## 📊 Dataset

**Stock:** Apple Inc. (AAPL)

**Data:** Historical daily OHLCV data

The dataset contains:

* Open
* High
* Low
* Close
* Volume

The data was downloaded using `yfinance` for the period:

**January 1, 2018 – January 1, 2026**

After target construction and removal of rows with insufficient historical values for the indicators, the final dataset contains **2,000 observations**.

---

## 🎯 Prediction Target

The task is to predict the **next trading day's price direction**.

The target is constructed using:

```python
data["Next_Close"] = data["Close"].shift(-1)

data["Target"] = (
    data["Next_Close"] > data["Close"]
).astype(int)
```

Where:

* `1` = Up
* `0` = Down

The shifted next-day closing price is used **only to construct the target** and is not included as a model feature.

The final row is removed because it does not have a known next-day closing price.

---

## 🛠️ Feature Engineering

The engineered feature set consists of five technical indicators calculated directly using pandas:

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| `SMA_10`        | 10-day Simple Moving Average              |
| `Momentum_5`    | 5-day price momentum                      |
| `Return_1`      | One-day percentage return                 |
| `Volatility_10` | 10-day rolling return volatility          |
| `Volume_Change` | Daily percentage change in trading volume |

All features use only current or previous information and do not use future prices.

---

## ⚖️ Class Balance

The final dataset contains:

| Class     |     Count | Percentage |
| --------- | --------: | ---------: |
| Down (0)  |       928 |      46.4% |
| Up (1)    |     1,072 |      53.6% |
| **Total** | **2,000** |   **100%** |

The classes are reasonably balanced, although Up days occur somewhat more frequently.

---

## 🔀 Train-Test Split

Since this is time-series data, the observations are **not randomly shuffled**.

The data is divided chronologically:

* **80% → Training data**
* **20% → Test data**

The final test set contains **400 observations**.

The test period is kept separate from model training and is used as unseen future data for final evaluation.

---

## 🧪 Naive Baselines

Two naive baselines are used for comparison.

### Persistence Baseline

Predicts that the next day's direction will be the same as the current day's direction.

**Test Accuracy: 52.25%**

### Majority Class Baseline

Always predicts the most common class in the **training data**.

The majority training class was:

**Up (1)**

**Test Accuracy: 56.25%**

---

## 🤖 Machine Learning Models

### 1. Raw OHLCV + Logistic Regression

The first model uses only the original:

* Open
* High
* Low
* Close
* Volume

features.

**Test Accuracy: 45.00%**

The features are standardized using `StandardScaler`, which is fitted **only on the training data**.

### 2. Engineered Features + Logistic Regression

The second model uses the five engineered features:

* SMA_10
* Momentum_5
* Return_1
* Volatility_10
* Volume_Change

**Test Accuracy: 47.75%**

### 3. Engineered Features + Random Forest

A Random Forest classifier is also trained on the same final engineered feature set.

**Test Accuracy: 46.75%**

This provides a comparison between two different machine learning models using the same engineered features.

---

## ⏳ Time-Series Cross-Validation

Forward time-series cross-validation was performed using `TimeSeriesSplit` with 5 folds.

The validation process preserves chronological order, with each fold training on earlier observations and validating on later observations.

### Mean Cross-Validation Accuracy

| Model                          | Mean CV Accuracy |
| ------------------------------ | ---------------: |
| Engineered Logistic Regression |       **51.05%** |
| Engineered Random Forest       |       **49.02%** |

For Logistic Regression, scaling was performed separately within each fold, with the scaler fitted only on that fold's training portion.

---

## 📋 Final Four-Way Comparison

The required four-way comparison on the unseen test set is:

| Approach                                  | Test Accuracy |
| ----------------------------------------- | ------------: |
| Majority Class Baseline                   |    **56.25%** |
| Persistence Baseline                      |    **52.25%** |
| Engineered Features + Logistic Regression |    **47.75%** |
| Raw OHLCV + Logistic Regression           |    **45.00%** |

The engineered feature set was evaluated with both Logistic Regression and Random Forest. Logistic Regression performed better and is therefore used as the engineered-feature model in the four-way comparison.

---

## 📈 Evaluation

The best-performing engineered-feature model on the test set was:

**Engineered Features + Logistic Regression**

**Test Accuracy: 47.75%**

The classification results on the test set were:

| Class | Precision | Recall | F1-Score |
| ----- | --------: | -----: | -------: |
| Down  |      0.45 |   0.79 |     0.57 |
| Up    |      0.59 |   0.24 |     0.34 |

**Overall Accuracy: 47.75%**

A predicted-vs-actual direction plot and confusion matrix are included in the notebook to visualize the model's predictions.

---

## 🔍 Results and Conclusion

The results show that the engineered Logistic Regression model achieved **47.75%** accuracy on the unseen test period, while the engineered Random Forest achieved **46.75%**.

The engineered Logistic Regression model performed better than the raw OHLCV Logistic Regression model (**45.00%**), showing a difference between the raw and engineered feature approaches.

However, neither engineered model outperformed the naive baselines. The Majority Class baseline achieved the highest test accuracy at **56.25%**, followed by the Persistence baseline at **52.25%**.

The forward cross-validation results were also close to 50%, with Logistic Regression achieving a mean accuracy of **51.05%** and Random Forest achieving **49.02%**.

Therefore, for this dataset, feature engineering and the tested machine learning models did **not provide a useful improvement over the simple naive baselines for next-day direction prediction**.

This demonstrates the importance of evaluating time-series models against appropriate baselines rather than judging performance from machine learning accuracy alone.

---

## ▶️ How to Run

### Google Colab

1. Open `Stock_Price_Movement_Predictor.ipynb` in Google Colab.
2. Run the cells sequentially from top to bottom.
3. The notebook automatically downloads AAPL historical data using `yfinance`.
4. The notebook then performs:

   * Data inspection
   * Target construction
   * Feature engineering
   * Class balance analysis
   * Time-based train-test splitting
   * Baseline evaluation
   * Model training
   * Time-series cross-validation
   * Final model comparison
   * Prediction and evaluation visualizations

No separate dataset file is required because the historical data is downloaded directly in the notebook.

### Jupyter Notebook

To run the notebook locally:

1. Install Python and Jupyter Notebook.
2. Install the Python libraries used in the notebook.
3. Open `Stock_Price_Movement_Predictor.ipynb`.
4. Run the cells sequentially from beginning to end.
---

## 📓 Notebook

The complete implementation, including data collection, target construction, feature engineering, model training, time-series validation, evaluation, comparison, and visualizations, is provided in the Jupyter Notebook.

**Notebook:** `Stock_Price_Movement_Predictor.ipynb`

## 🔗 Project Links 

* **GitHub Repository:** `PASTE_YOUR_GITHUB_REPO_LINK_HERE`
* **Deployed Link:** `PASTE_YOUR_DEPLOYED_LINK_HERE`
* **Demo Video:** `PASTE_YOUR_DEMO_VIDEO_LINK_HERE`
* **Google Colab Notebook:** `PASTE_YOUR_COLAB_LINK_HERE`


