import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_y, check_array, check_is_fitted, check_X_y


class Lvov(TransformerMixin, BaseEstimator):
    """
    Calculates features imporances with the help of shap values and labels.
    The logic is that a feature that pushes positively when a label is 1 and negatively when is 0 is important.
    """

    def __int__(self):
        self.feature_importances = None

    def fit(self, model, X: pd.DataFrame, y):
        """
        Calculates the importances for each feature, orders them by importance and saves them in self.feature_importances
        :param model: fitted model to be used to calculate shap values
        :param X: pandas dataframe for which shap values will be calculated
        :param y: binary array of labels. Must be filled with 0s and 1s only
        :return: self
        """
        y = _check_y(y)
        y = y.reshape((y.shape[0], 1)).astype(float)
        y = (y - 1) + y  # convert (0, 1) vector to (-1, 1) vector
        y[y == 1] = sum(y == -1) / sum(y == 1)  # make the impact of each class the same

        explainer = shap.Explainer(model)
        shap_values = explainer(X)

        fi = np.sum(shap_values.values * y, axis=0)
        self.feature_importances = sorted(
            list(zip(X.columns, fi)), key=lambda x: (x[1]), reverse=True
        )
        return self

    def transform(self, X: pd.DataFrame, th=0):
        """
        Removes the features from dataframe X that fall bellow the threshold
        :param X: pandas dataframe
        :param threshold: float
        :return: pandas dataframe
        """
        # Check is fit had been called
        check_is_fitted(self, "feature_importances")

        cols = [x[0] for x in self.feature_importances if x[1] > th]
        return X[cols]


if __name__ == "__main__":
    from catboost import CatBoostClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    header = [
        "wa",
        "bb",
        "ez",
        "2s",
        "gn",
        "ox",
        "xz",
        "uk",
        "8c",
        "7d",
        "cx",
        "m8",
        "hy",
        "qo",
        "a3",
        "l9",
        "ip",
        "0a",
        "af",
        "pg",
        "sf",
        "hz",
        "r2",
        "v7",
        "o1",
        "5k",
        "w5",
        "ai",
        "v9",
        "i3",
    ]
    X_train, X_test = pd.DataFrame(X_train, columns=header), pd.DataFrame(
        X_test, columns=header
    )

    model = CatBoostClassifier()
    model = model.fit(X_train, y_train)

    lvov = Lvov().fit(model, X_test, y_test)
    filtered_X_train = lvov.transform(X_train)
