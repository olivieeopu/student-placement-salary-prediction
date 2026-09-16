from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# FEATURE ENGINEERING
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['academic_score'] = (
            X['cgpa'] * 0.5 +
            X['twelfth_percentage'] * 0.25
        )

        X['experience_score'] = (
            X['internships_completed'] +
            X['projects_completed'] +
            X['hackathons_participated']
        )

        X['skill_score'] = (
            X['coding_skill_rating'] +
            X['aptitude_skill_rating']
        )

        return X


# FEATURE DROPPER
class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')


# PREPROCESSOR (FINAL)
def create_preprocessor():

    # Numerical pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),   # aman kalau ada missing numeric
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabung (dynamic column selection)
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, make_column_selector(dtype_include=['int64', 'float64'])),
        ('cat', cat_pipeline, make_column_selector(dtype_include=['object']))
    ])

    return preprocessor