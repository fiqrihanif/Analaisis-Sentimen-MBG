import pandas as pd

def get_metrics_data():
    """Mengembalikan DataFrame berisi metrik evaluasi model"""
    metrics_data = {
        "Model": ["Logistic Regression", "Naive Bayes"],
        "Accuracy": [0.8231, 0.7950],
        "Precision": [0.8150, 0.7820],
        "Recall": [0.8230, 0.7950],
        "F1-Score": [0.8185, 0.7870]
    }
    return pd.DataFrame(metrics_data)