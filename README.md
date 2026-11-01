# 🍷 Wine Quality Prediction

A machine learning project that predicts whether a red wine is **good quality** or **bad quality** from its physicochemical properties, using a Random Forest classifier.

## Overview

The project uses the well-known **Red Wine Quality** dataset (UCI Machine Learning Repository / Kaggle). Each wine is described by 11 laboratory measurements and has a quality score given by sensory tasting. The quality score is converted into a binary label, and a Random Forest model is trained to classify wines.

**Result:** ~**90.3% accuracy** on the held-out test set.

## Dataset

- **File:** `winequality-red.csv` (semicolon-separated)
- **Samples:** 1,599 red wines
- **Missing values:** none
- **Target:** `quality` (original scores range from 3 to 8)

| Feature | Description |
|---|---|
| fixed acidity | Non-volatile acids (e.g. tartaric) |
| volatile acidity | Acetic acid content |
| citric acid | Citric acid content |
| residual sugar | Sugar left after fermentation |
| chlorides | Salt content |
| free sulfur dioxide | Free SO₂ |
| total sulfur dioxide | Total SO₂ |
| density | Density of the wine |
| pH | Acidity/basicity level |
| sulphates | Sulphate (preservative) content |
| alcohol | Alcohol percentage |

## Project Workflow

1. **Data collection** – load the CSV with pandas (`sep=';'`).
2. **Data analysis** – check for missing values, summary statistics, quality distribution (count plot), and feature-vs-quality bar plots.
3. **Correlation analysis** – heatmap of positive and negative correlations between features.
4. **Preprocessing** – separate features from the target and apply binary label encoding:
   - `quality >= 7` → **1 (Good)**
   - `quality < 7` → **0 (Bad)**
5. **Train/test split** – 80% train, 20% test (`random_state=32`).
6. **Model training** – `RandomForestClassifier` from scikit-learn.
7. **Evaluation** – accuracy score on the test set.
8. **Predictive system** – classify a new wine from a tuple of 11 feature values.

## Tech Stack

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

## Installation

```bash
git clone <your-repo-url>
cd wine-quality-prediction
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Usage

1. Place `winequality-red.csv` in the project folder.
2. Update the file path in the data loading cell of the notebook, which currently points to a local Windows path:
   ```python
   df = pd.read_csv('winequality-red.csv', sep=';')
   ```
3. Launch the notebook and run all cells:
   ```bash
   jupyter notebook wine_quality_prediction.ipynb
   ```

### Predicting a new wine

Provide the 11 features in this order: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.

```python
input_data = (7.4, 0.59, 0.08, 4.4, 0.086, 6, 29, 0.9974, 3.38, 0.5, 9)
prediction = model.predict(np.asarray(input_data).reshape(1, -1))

print('GOOD QUALITY WINE' if prediction[0] == 1 else 'BAD Quality Wine')
```

Output for the sample above: `BAD Quality Wine`

## Results

| Metric | Value |
|---|---|
| Model | Random Forest Classifier |
| Test accuracy | 0.903 |

> **Note:** The binary target is imbalanced (most wines score below 7), so accuracy alone can be optimistic. Consider adding precision, recall, F1-score, and a confusion matrix for a fuller evaluation.

## Possible Improvements

- Handle class imbalance (SMOTE, class weights)
- Hyperparameter tuning with `GridSearchCV` / `RandomizedSearchCV`
- Compare against other models (Logistic Regression, SVM, XGBoost)
- Add feature importance analysis
- Include the white wine dataset
- Deploy as a web app (Streamlit / Flask)

## Project Structure

```
wine-quality-prediction/
├── wine_quality_prediction.ipynb
├── winequality-red.csv
└── README.md
```

## Acknowledgements

Dataset: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems, 2009.

## License

This project is open source and available under the [MIT License](LICENSE).
