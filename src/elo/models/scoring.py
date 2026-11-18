"""Identical feature selection and XGBoost prediction for evaluation and monitoring."""

import xgboost as xgb

from elo.models.aggregate import FEATURE_COLUMNS


def predict(model, frame, columns=FEATURE_COLUMNS):
    """Predict from ordered features with categorical support, matching serving selection."""
    return model.predict(xgb.DMatrix(frame[columns], enable_categorical=True))
