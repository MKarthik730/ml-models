# Copilot Instructions for ML Projects Portfolio

## Architecture Overview

This is a **machine learning portfolio** with multiple standalone projects, each following a consistent structure. Projects are NOT tightly coupled—each has its own data, models, and deployment code.

### Project Structure Pattern
```
projects/{project-name}/
├── data/           # Raw CSV/datasets
├── src/            # Training & processing scripts
├── models/         # Serialized .joblib files
├── api/            # FastAPI backend
└── app/            # Streamlit frontend
```

### Key Projects
- **heart-disease-prediction**: RandomForestClassifier with FastAPI + Streamlit UI + PostgreSQL persistence
- **decision-tree**: Titanic survival (Decision Tree & Random Forest) with sklearn pipelines
- **linear-regression**: Basic regression workflows
- **image-compression-svd**: Matrix factorization for image dimensionality reduction

## Critical Patterns & Conventions

### 1. ML Pipeline Pattern (sklearn)
Projects use **ColumnTransformer + Pipeline** extensively:
- Numeric cols: StandardScaler/MinMaxScaler via `make_column_selector(dtype_include=np.number)`
- Categorical cols: OneHotEncoder with `handle_unknown="ignore"`
- See `advanced_sklearn/01_pipelines_and_column_transformer.py` for full examples

**Key convention**: Define preprocessor separately, then wrap in Pipeline with model:
```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(sparse_output=False), cat_cols)
])
pipeline = Pipeline([('processor', preprocessor), ('model', model)])
```

### 2. Model Serialization
- Models saved as `.joblib` files (not pickle): `joblib.load()` / `joblib.dump()`
- Location: `projects/{name}/models/` or root of api/app directories
- Heart disease model loaded in both `api/main.py` and `app/streamlit_app.py`

### 3. Hyperparameter Tuning
- Use `GridSearchCV` with `KFold` for cross-validation
- Prefix grid params with estimator name in pipeline: `"model__n_estimators"`, `"model__max_depth"`
- See `heart-disease-prediction/src/train.py` for concrete pattern

### 4. Data Cleaning Conventions
- Handle outliers by replacing with column mean:
  ```python
  data.loc[(data['col'] > threshold) | (data['col'] < threshold), 'col'] = data['col'].mean()
  ```
- Use `train_test_split(..., stratify=y)` for classification to preserve class distribution
- Drop unwanted columns early: `data.drop(['target_str', 'sex_str'], axis=1)`

## Web Framework Integration

### FastAPI Backend (api/main.py)
- **ORM**: SQLAlchemy with PostgreSQL
- **Core setup**: CORSMiddleware for all origins, database auto-creation
- **Model loading**: `joblib.load('model.joblib')` at module level
- **LLM integration**: LangChain + OpenAI GPT-4o-mini for contextual assistance
- **Prompts**: Use PromptTemplate with strict instructions (cardiologist assistant example limits to heart-related Q&A)

### Streamlit Frontend (app/streamlit_app.py)
- Simple widget-based input collection
- Load model: `joblib.load('model.joblib')` at module level
- Image display: `st.image()` for logo/illustrations
- Prediction trigger: `if st.button("Predict")` → create input DataFrame → run prediction
- No complex state management (Streamlit handles it)

### Database Models (databasemodels.py)
- SQLAlchemy declarative base pattern
- Column types: Integer, Float, String with nullable constraints
- Primary key auto-increment: `Column(Integer, primary_key=True, index=True)`

## Development Workflows

### Training a New Model
1. Create dataset in `projects/{name}/data/dataset.csv`
2. Write training script in `src/train.py`:
   - Load data → split → preprocess with ColumnTransformer
   - Fit model with GridSearchCV
   - Save via `joblib.dump(model, 'model.joblib')`
3. Test predictions locally before API integration

### Adding Features to Heart Disease Predictor
- **Add column to model**: Update both `databasemodels.py`, `streamlit_app.py` input widgets, and `train.py` preprocessing
- **API endpoint**: Modify `main.py` to accept new UserCreate schema field
- **Database migration**: SQLAlchemy auto-creates tables; manually add Column if modifying existing Users table

### Deployment Checklist
- [ ] Test locally: `streamlit run app/streamlit_app.py`
- [ ] API test: `uvicorn api.main:app --reload`
- [ ] Models in correct location (joblib files)
- [ ] Environment variables set (OPEN_API_KEY for LLM, database URL)
- [ ] Requirements.txt updated with new dependencies

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `joblib.load()` fails | Verify model exists, use absolute paths in development |
| ColumnTransformer shape mismatch | Ensure all columns in `X_train` match preprocessor definition |
| GridSearchCV hangs | Reduce param_grid size or use fewer CV splits |
| LLM API key errors | Check `OPEN_API_KEY` environment variable set |
| Streamlit "app running twice" | Normal behavior on code changes; doesn't affect functionality |

## Technology Stack Notes

- **sklearn 1.3+**: All pipelines, models, preprocessing
- **FastAPI 0.100+**: REST APIs with Pydantic validation
- **Streamlit 1.28+**: Interactive dashboards (single-file apps, no routing)
- **SQLAlchemy 2.0+**: Use `sqlalchemy.orm.Session` pattern (not deprecated `session()` calls)
- **LangChain**: Prompt management + LLM chaining
- **Jupyter notebooks**: Learning materials in `learning/ai-hands-on/`

## When to Reference Advanced Examples

- **Pipelines deep dive**: `advanced_sklearn/01_pipelines_and_column_transformer.py` (custom transformers, make_column_transformer)
- **Custom transformers**: `advanced_sklearn/02_custom_transformers.py` (TransformerMixin pattern)
- **Hyperparameter tuning edge cases**: `advanced_sklearn/03_hyperparameter_tuning.py`
- **Learning materials**: Notebooks in `learning/ai-hands-on/` for theory (math, PyTorch, transformers)

---

**Last Updated**: January 2026  
**Questions?** Review the README.md for project descriptions and architecture diagrams.
