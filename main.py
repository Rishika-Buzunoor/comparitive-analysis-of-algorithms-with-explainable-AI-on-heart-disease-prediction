from src.data_preprocessing import load_data, preprocess_data
from src.train_models import train_all_models
from src.evaluate_models import evaluate
from src.plot_results import plot_metrics
from src.confusion_matrix_plot import plot_confusion_matrices
from src.shap_explain import explain_shap
from src.lime_explain import explain_lime

# Load
X, y = load_data("data/heart.csv")

# Preprocess
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Train
models = train_all_models(X_train, y_train)

# Evaluate
results = evaluate(models, X_test, y_test)

# Graphs

plot_metrics(results)

# Confusion Matrix
plot_confusion_matrices(models, results, X_test, y_test)

# Explainability
for name, model in models.items():
    explain_shap(model, X_train, X_test, name)
    explain_lime(model, X_train, X_test, name)