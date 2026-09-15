from tests.test_features import sample
from src.models.classifier import SwingClassifier


def test_classifier_fit_predict():
    df = sample()
    model = SwingClassifier(horizon=5, min_samples=150)
    model.fit(df)
    p = model.predict_proba(df)
    assert 0 <= p <= 1
