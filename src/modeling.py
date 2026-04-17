from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, precision_score, recall_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FEATURE_COLUMNS = [
    'Category',
    'Sub-Category',
    'Region',
    'Segment',
    'Ship Mode',
    'Discount',
    'Sales',
    'Quantity',
    'shipping_days',
]
CATEGORICAL_COLUMNS = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode']
NUMERIC_COLUMNS = ['Discount', 'Sales', 'Quantity', 'shipping_days']


@dataclass
class LossModelResult:
    pipeline: Pipeline
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray
    train_rows: int
    test_rows: int


@dataclass
class ProfitModelResult:
    pipeline: Pipeline
    mae: float
    r2: float
    sample_actuals: list[float]
    sample_predictions: list[float]


@dataclass
class SegmentInsight:
    band_name: str
    avg_sales: float
    avg_profit: float
    loss_rate: float
    rows: int


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLUMNS),
            ('num', 'passthrough', NUMERIC_COLUMNS),
        ]
    )



def train_loss_classifier(df: pd.DataFrame) -> LossModelResult:
    X = df[FEATURE_COLUMNS]
    y = df['is_loss']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ('prep', _make_preprocessor()),
            (
                'model',
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=12,
                    min_samples_leaf=3,
                    random_state=42,
                    class_weight='balanced',
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)
    return LossModelResult(
        pipeline=pipeline,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        confusion_matrix=cm,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )



def train_profit_regressor(df: pd.DataFrame) -> ProfitModelResult:
    X = df[FEATURE_COLUMNS]
    y = df['Profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(
        steps=[
            ('prep', _make_preprocessor()),
            (
                'model',
                RandomForestRegressor(
                    n_estimators=220,
                    max_depth=14,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    preview = pd.DataFrame({'actual': y_test.values, 'predicted': predictions}).head(12)
    return ProfitModelResult(
        pipeline=pipeline,
        mae=mae,
        r2=r2,
        sample_actuals=preview['actual'].round(2).tolist(),
        sample_predictions=preview['predicted'].round(2).tolist(),
    )



def build_discount_risk_segments(df: pd.DataFrame) -> list[SegmentInsight]:
    summary = (
        df.groupby('discount_band', observed=False)
        .agg(
            avg_sales=('Sales', 'mean'),
            avg_profit=('Profit', 'mean'),
            loss_rate=('is_loss', 'mean'),
            rows=('Row ID', 'count'),
        )
        .reset_index()
    )

    insights: list[SegmentInsight] = []
    for _, row in summary.iterrows():
        insights.append(
            SegmentInsight(
                band_name=str(row['discount_band']),
                avg_sales=float(row['avg_sales']),
                avg_profit=float(row['avg_profit']),
                loss_rate=float(row['loss_rate']),
                rows=int(row['rows']),
            )
        )
    return insights



def get_feature_importance_table(pipeline: Pipeline) -> pd.DataFrame:
    prep = pipeline.named_steps['prep']
    model = pipeline.named_steps['model']
    feature_names = prep.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            'feature': feature_names,
            'importance': importances,
        }
    ).sort_values('importance', ascending=False)
    importance_df['feature'] = (
        importance_df['feature']
        .str.replace('cat__', '', regex=False)
        .str.replace('num__', '', regex=False)
    )
    return importance_df



def get_confusion_matrix_frame(loss_result: LossModelResult) -> pd.DataFrame:
    labels = ['Actual non-loss', 'Actual loss']
    columns = ['Predicted non-loss', 'Predicted loss']
    return pd.DataFrame(loss_result.confusion_matrix, index=labels, columns=columns)



def get_regression_preview_frame(profit_result: ProfitModelResult) -> pd.DataFrame:
    preview = pd.DataFrame(
        {
            'actual_profit': profit_result.sample_actuals,
            'predicted_profit': profit_result.sample_predictions,
        }
    )
    preview['error'] = (preview['predicted_profit'] - preview['actual_profit']).round(2)
    return preview
