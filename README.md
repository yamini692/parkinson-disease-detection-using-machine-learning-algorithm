# 🧠 Parkinson's Disease Detection using Machine Learning

A machine learning project to detect Parkinson’s Disease using biomedical voice measurements. Built using Python, Scikit-learn, and Random Forest Classifier — inspired by a personal motivation to apply AI in healthcare.

---

## 📌 Project Motivation

Parkinson’s Disease affected someone close to me — my grandfather. That moment drove me to learn more about it and inspired me to create an AI system that could help detect it early using real patient data.

---

## 📊 Dataset

- **Source**: [Kaggle - Parkinson’s Disease Data Set](https://www.kaggle.com/datasets/nikhileswarkomati/parkinsons-disease-data-set)
- **Description**: The dataset contains voice measurements from patients. Features include fundamental frequency, jitter, shimmer, and more.
- **Target Variable**: `status` (1 = Parkinson’s, 0 = Healthy)

---

## 🧪 Tech Stack

- **Language**: Python
- **Libraries**:
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – machine learning models & evaluation

---

## 🔍 Workflow

1. **Data Collection** – Downloaded dataset from Kaggle
2. **Preprocessing** – Removed irrelevant columns, handled features and target
3. **Feature Scaling** – Applied `MinMaxScaler` to normalize data
4. **Model Training** – Used `RandomForestClassifier` for classification
5. **Evaluation** – Measured accuracy, precision, recall, F1-score, and confusion matrix
6. **Visualization** – Plotted confusion matrix for better interpretability

---

## 📈 Model Performance

- ✅ **Accuracy**: ~93%
- 🔍 **Evaluation Metrics**: Confusion Matrix, Precision, Recall, F1-score
- 🌲 **Model Used**: Random Forest (100 estimators)

---

## 💡 Why Random Forest?

- Handles high-dimensional feature spaces well
- Less sensitive to outliers
- Reduces overfitting through bagging (ensemble of trees)

---

## 🎯 Future Improvements

- Implement cross-validation and hyperparameter tuning
- Try deep learning models like MLP or CNN for comparison
- Deploy as a web app for hospitals or early screening centers

---

## ⚠️ Disclaimer

This project is **not** intended for clinical diagnosis. It’s a research-based learning project and should be used for educational purposes only.

---

## 🙌 Acknowledgements

- Kaggle Dataset: [nikhileswarkomati/parkinsons-disease-data-set](https://www.kaggle.com/datasets/nikhileswarkomati/parkinsons-disease-data-set)
- Inspired by real-life challenges in healthcare ❤️

---

## 📬 Connect With Me

If you'd like to collaborate, ask questions, or just connect:

- GitHub: [YourUsername](https://github.com/YourUsername)
- LinkedIn: [YourLinkedIn](https://linkedin.com/in/YourLinkedIn)

