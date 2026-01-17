<div align="center">

# Machine Learning Projects Portfolio

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive collection of machine learning projects showcasing end-to-end ML pipelines, from data preprocessing to model deployment with FastAPI and Streamlit interfaces.

[Features](#-features) • [Projects](#-projects) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 📋 Table of Contents

- [Features](#-features)
- [Projects](#-projects)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

- 🎯 **Production-Ready ML Models** - Complete pipelines from training to deployment
- 🚀 **REST API Integration** - FastAPI backends for real-time predictions
- 📊 **Interactive Dashboards** - Streamlit web applications for model interaction
- 🧹 **Data Preprocessing** - Comprehensive data cleaning and feature engineering
- 📈 **Model Evaluation** - Detailed metrics, visualization, and performance analysis
- 💾 **Model Persistence** - Serialized models ready for deployment
- 🔄 **Reproducible Pipelines** - Standardized workflows for consistent results

## 🎯 Projects

### 1. Heart Disease Prediction
> **Goal:** Predict the risk of heart disease using patient health metrics

- **Algorithm:** Random Forest Classifier
- **Features:** Age, blood pressure, cholesterol, glucose levels, etc.
- **Deployment:** FastAPI REST API + Streamlit dashboard
- **Accuracy:** ~85%

📁 `projects/heart-disease-prediction/`

### 2. Linear Regression Analysis
> **Goal:** Predict outcomes using linear relationships

- **Use Cases:**
  - Salary prediction based on experience
  - Product quantity sold based on price and rating
- **Algorithm:** Multiple Linear Regression
- **Evaluation:** MAE, MSE, R² Score

📁 `projects/linear-regression/`

### 3. Decision Tree & Random Forest
> **Goal:** Classification using tree-based models

- **Dataset:** Titanic survival prediction
- **Algorithms:**
  - Decision Tree Classifier
  - Random Forest Classifier
- **Features:** Passenger class, age, sex, fare, etc.

📁 `projects/decision-tree/`

### 4. Image Compression using SVD
> **Goal:** Compress images using Singular Value Decomposition

- **Technique:** Matrix factorization (SVD)
- **Applications:** Image compression, dimensionality reduction
- **Visualization:** Compare original vs compressed at different k values

📁 `projects/image-compression-svd/`

## 🛠️ Tech Stack

### Core ML & Data Science
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

### Web Frameworks & APIs
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

### Database
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)

### Development Tools
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/MKarthik730/ml-models.git
cd ml-models
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Heart Disease Prediction

**Train the model:**
```bash
cd projects/heart-disease-prediction/src
python train.py
```

**Run FastAPI server:**
```bash
cd projects/heart-disease-prediction/api
uvicorn main:app --reload
```
Visit: `http://localhost:8000/docs` for API documentation

**Run Streamlit app:**
```bash
cd projects/heart-disease-prediction/app
streamlit run streamlit_app.py
```

### Linear Regression

```bash
cd projects/linear-regression/src
python linear_regression.py
```

### Image Compression (SVD)

```bash
cd projects/image-compression-svd/src
python svd_compression.py
```

## 📂 Project Structure

```
ml-models/
├── projects/                    # Individual ML projects
│   ├── heart-disease-prediction/
│   │   ├── data/               # Dataset files
│   │   ├── models/             # Trained model files
│   │   ├── src/                # Source code
│   │   ├── api/                # FastAPI backend
│   │   └── app/                # Streamlit frontend
│   ├── linear-regression/
│   ├── decision-tree/
│   └── image-compression-svd/
│
├── notebooks/                   # Jupyter notebooks for exploration
├── utils/                       # Utility functions and helpers
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

**Karthik Motupalli**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/karthik-motupalli)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/MKarthik730)

**Organization:** ANITS Vizag  
**Location:** Visakhapatnam, Andhra Pradesh, India

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

![Profile Views](https://komarev.com/ghpvc/?username=mkarthik730&label=Profile%20views&color=0e75b6&style=flat)

</div>
