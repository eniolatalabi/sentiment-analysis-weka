# Amazon Review Sentiment Analysis

A machine learning project that analyzes Amazon product reviews to determine if they are positive or negative using Weka.

## What This Project Does

Takes Amazon product reviews and automatically classifies them as either positive or negative using machine learning algorithms.

## Dataset

- **Source:** Amazon Fine Food Reviews (from Kaggle)
- **Original size:** 568,454 reviews
- **Used for analysis:** 1,500 reviews initially selected
- **Final cleaned dataset:** 716 reviews
  - 500 positive reviews
  - 216 negative reviews

## Tools Used

- **Python 3.x** - For data cleaning and preprocessing
- **Pandas** - For data manipulation
- **Weka 3.8.6** - For machine learning classification
- **Regular Expressions (re)** - For text cleaning

## Project Structure

```
sentiment-analysis-weka/
├── data/
│   └── amazon_sentiment_clean.csv    # Cleaned dataset ready for Weka
├── scripts/
│   └── prepare_data.py                # Python script that cleans the data
├── results/
│   ├── naivebayes_results.txt         # Naive Bayes classifier results
│   ├── j48_results.txt                # J48 Decision Tree results
│   ├── randomforest_results.txt       # Random Forest results
│   └── sample_reviews.csv             # Sample of processed reviews
└── README.md                          # This file
```

## How It Works

### Step 1: Data Preparation (Python)
The `prepare_data.py` script:
1. Loads the raw Amazon reviews dataset
2. Cleans the text (removes HTML, special characters, extra spaces)
3. Converts to lowercase
4. Creates sentiment labels based on star ratings:
   - Ratings 4-5 stars → Positive
   - Ratings 1-2 stars → Negative
   - Rating 3 stars → Removed (neutral, makes classification cleaner)
5. Balances the dataset
6. Saves cleaned data as CSV

### Step 2: Feature Extraction (Weka)
In Weka:
1. Import the cleaned CSV file
2. Use StringToWordVector filter to convert text into numerical features
3. This creates word frequency features (top 1000 most common words)

### Step 3: Classification (Weka)
Three machine learning algorithms were tested:
1. **Naive Bayes** - Probabilistic classifier
2. **J48** - Decision tree algorithm
3. **Random Forest** - Ensemble of decision trees

## Results

All three classifiers achieved **69.83% accuracy**.

### Why 69.83%?
The dataset has more positive reviews (500) than negative reviews (216). This imbalance caused all classifiers to predict mostly positive reviews, which naturally gives ~70% accuracy.

### Confusion Matrix
All models showed the same pattern:
- Correctly classified all 500 positive reviews
- Failed to classify negative reviews
- This is a known issue with imbalanced datasets

### What This Means
The project successfully demonstrates:
- Data preprocessing pipeline
- Text feature extraction
- Multiple ML algorithm comparison
- Critical analysis of results and limitations

## How to Run This Project

### Prerequisites
- Python 3.x installed
- Pandas library: `pip install pandas`
- Weka 3.8.6 installed

### Running the Code

**Step 1: Prepare the data**
```bash
cd scripts
python prepare_data.py
```

This creates `amazon_sentiment_clean.csv` in the `data/` folder.

**Step 2: Open Weka**
1. Open Weka application
2. Click "Explorer"
3. Load `data/amazon_sentiment_clean.csv`
4. Apply StringToWordVector filter
5. Go to Classify tab
6. Choose a classifier (Naive Bayes, J48, or Random Forest)
7. Click Start

**Step 3: View results**
Results show accuracy, confusion matrix, and other metrics.

## Key Findings

### What Worked
- Successfully cleaned and processed 716 reviews
- All three algorithms ran successfully
- Achieved consistent 69.83% accuracy across all methods

### Limitations
- Dataset imbalance (70% positive, 30% negative)
- Classifiers biased toward majority class (positive)
- Real-world accuracy would improve with balanced dataset

### Future Improvements
To improve this project:
1. Balance the dataset (equal positive and negative reviews)
2. Use SMOTE (synthetic minority oversampling)
3. Adjust algorithm class weights
4. Try more advanced algorithms
5. Use different evaluation metrics (F1-score, precision, recall)

## Author

Eniola Talabi  
February 2026  
Sentiment Analysis Class Project

## Submission Contents

- This README
- Python preprocessing script
- Cleaned dataset (CSV)
- Three Weka result files
- Sample reviews file

---

**Total Reviews Analyzed:** 716  
**Accuracy Achieved:** 69.83%  
**Algorithms Tested:** 3 (Naive Bayes, J48, Random Forest)
