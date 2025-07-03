# Quora Question Pairs - Duplicate Detection

This project addresses the Quora Question Pairs classification task: determining whether two questions asked on Quora are semantically equivalent. The goal is to build models that can accurately classify question pairs as **duplicate** or **not duplicate** using both traditional machine learning and deep learning techniques.

## 📂 Dataset

The dataset consists of:
- `train.csv`: Question pairs with `is_duplicate` labels.
- `test.csv`: Unlabeled question pairs.
- `sample_submission.csv`: Format for submitting predictions.

Each entry in the dataset contains:
- `qid1`, `qid2`: Question IDs  
- `question1`, `question2`: Text of the questions  
- `is_duplicate`: Label (1 if questions are duplicates, 0 otherwise)

## 📌 Project Pipeline

### 1. Data Preprocessing & Exploration
- Removed missing values and duplicates.
- Explored label distribution and question repetition frequency.
- Analyzed frequency of question appearances with visualizations.

### 2. Text Cleaning
- Tokenization using `gensim.simple_preprocess`.
- Stopword removal and stemming (`nltk`).
- Word2Vec training on the combined corpus of all questions.

### 3. Feature Engineering
- Word2Vec embeddings: custom-trained on cleaned corpus.
- Custom features:
  - Cosine similarity between embeddings
  - Sentiment polarity & subjectivity (`TextBlob`)
  - Fuzzy string matching metrics

### 4. Traditional ML Models
- **Random Forest Classifier**
- **XGBoost Classifier**
- Used custom features and evaluated using:
  - Accuracy
  - Confusion matrix
  - Classification report

### 5. Deep Learning Models
- Built models using TensorFlow/Keras:
  - **RNN-based model**
  - **LSTM-based model**
- Input: padded sequences of word indices from the corpus.
- Word embeddings from the trained Word2Vec model were used as weights.

### 6. Evaluation & Inference
- Confusion matrices and classification reports plotted.
- Predictions generated on the test set using best-performing models.

## Model Performance Summary

| Model                | Accuracy (%) |
|---------------------|--------------|
| Random Forest        | 85.84        |
| XGBoost              | 80.60        |
| RNN                  | 95.41        |
| LSTM                 | 97.69        |


## 📊 Results

Both traditional and deep learning models were evaluated. Key findings:
- Custom features improved ML model performance.
- LSTM models performed competitively due to their ability to capture sequential word dependencies.
