# Obesity Level Classification

Multiclass classification of obesity levels with a reproducible preprocessing pipeline, stratified splitting, cross-validation, hyperparameter tuning, baseline comparison, error analysis, and model persistence.

The final deployable track deliberately excludes Height, Weight, and BMI to measure prediction from lifestyle features. A second track retains them as a reference.

## Latest clean-run result

- Selected model: tuned Random Forest
- Test macro F1: 0.8464
- Test accuracy: 0.8517
- Majority-baseline accuracy: 0.1675

## Run

Place the CSV in `data/` and run `classification_project.ipynb` from top to bottom.

Run `streamlit run app.py`, then upload the dataset inside the app. The dataset is intentionally not stored in the public repository.
