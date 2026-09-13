# Obesity Level Classification

**[Open the live Streamlit app](https://karim797-obesity-classification.streamlit.app/)**

Multiclass classification of obesity levels with a reproducible preprocessing pipeline, stratified splitting, cross-validation, hyperparameter tuning, baseline comparison, error analysis, and model persistence.

The final deployable track deliberately excludes Height, Weight, and BMI to measure prediction from lifestyle features. A second track retains them as a reference.

## Latest clean-run result

- Selected model: tuned Random Forest
- Test macro F1: 0.8464
- Test accuracy: 0.8517
- Majority-baseline accuracy: 0.1675

## Live app

The app automatically downloads the official UCI dataset and trains the validated lifestyle-only pipeline. A compatible CSV can still be uploaded optionally.

## Run

Place the CSV in `data/` and run `classification_project.ipynb` from top to bottom.

Run the interactive app locally with `streamlit run app.py`.
