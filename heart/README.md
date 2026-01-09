<h1 align="center">❤️ Heart Disease Prediction ML Model</h1>
<h3 align="center">Predicting heart disease using classic ML models and production-ready pipelines.</h3>

<p align="center">
  <a href="https://github.com/MKarthik730/ml-models">
    <img src="https://img.shields.io/badge/GitHub-ML%20Models-blue" alt="GitHub Repo" />
  </a>
  <a href="https://pypi.org/project/scikit-learn/">
    <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow" alt="Scikit Learn" />
  </a>
</p>

<h3 align="left">📌 About</h3>
<p align="left">
  This project trains machine learning models to predict the presence of heart disease from clinical features such as age, cholesterol, blood pressure, and ECG results. The repo demonstrates end-to-end ML workflows including data preprocessing, model training, evaluation, and production-ready deployment using FastAPI.
</p>

<h3 align="left">🎯 Key Features</h3>
<ul>
  <li>🧪 <b>Multiple ML Algorithms</b> – Logistic Regression, Random Forest, Decision Trees, and more</li>
  <li>📊 <b>Data Preprocessing</b> – Feature scaling, encoding categorical variables, and missing value handling</li>
  <li>🚀 <b>Deployment Ready</b> – Model serialization with pickle and FastAPI integration for inference</li>
  <li>📈 <b>Evaluation</b> – Accuracy, precision, recall, F1-score, confusion matrix, and cross-validation</li>
</ul>

<h3 align="left">📂 Repository Structure</h3>
<ul>
  <li><code>heart.csv</code> – Dataset containing patient features and target labels</li>
  <li><code>train.py</code> – Script to preprocess data and train the model</li>
  <li><code>predict.py</code> – Script for making predictions on new data</li>
  <li><code>model.pkl</code> – Saved trained model</li>
  <li><code>notebooks/</code> – Jupyter notebooks for exploration and model evaluation</li>
</ul>

<h3 align="left">⚙️ Installation & Setup</h3>
```bash
git clone https://github.com/MKarthik730/ml-models.git
cd ml-models/heart
pip install -r requirements.txt
