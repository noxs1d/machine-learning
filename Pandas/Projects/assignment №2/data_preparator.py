import pandas as pd
import numpy as np


class PandasDfCreator:

    def __init__(self, n_train, n_test, noise):
        self.n_train = n_train
        self.n_test = n_test
        self.noise = noise

    def generate(self, n_samples):
        X = np.random.rand(n_samples) * 10 - 5
        X = np.sort(X).ravel()
        y = (
            np.exp(-(X ** 2)) + 1.5 * np.exp(-(X - 2) ** 2) + np.random.normal(0.0, self.noise, n_samples)
        )

        return X, y

    def create_random_df(self):
        X_train, y_train = self.generate(n_samples=self.n_train)
        X_test, y_test = self.generate(n_samples=self.n_test)

        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        train_df['X'] = X_train
        train_df['y'] = y_train
        test_df['X'] = X_test
        test_df['y'] = y_test

        print(test_df.info())

        return train_df, test_df
