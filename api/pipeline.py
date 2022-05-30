"""
Date: 29 May 2022
Define classes used in the pipeline
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Select a Feature
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what this custom transformer need to do
    def transform(self, X, y=None):
        return X[self.feature_names]

# Handling categorical features
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes one boolean as its argument
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # Remove white space in categorical features
        df = df.apply(lambda row: row.str.strip())

        # customize feature?
        # How can I identify what needs to be modified? EDA!!!!
        if self.new_features:

            # replace ? with Unknown
            edit_cols = ['buying', 'maint', 'doors','persons', 'lug_boot', 'safety']
            for col in edit_cols:
                df.loc[df[col] == '?', col] = 'unknown'

            # adjust doors and persons
            five_more = ['5more']
            more = ['more']

            # replace
            df['doors'].replace(to_replace=five_more, value='>=5', inplace=True)
            df['persons'].replace(to_replace=more, value='>=5', inplace=True)

        # update column names
        self.colnames = df.columns

        return df
        
