import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreparator:
    def __init__(self, df: pd.DataFrame, target: str, test_size=0.2):
        self.df = df
        self.target = target
        self.test_size = test_size

        # Автоматически определяем признаки и категориальные признаки
        self.features = [col for col in df.columns]
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_features = df.select_dtypes(include=['number']).columns.tolist()

    def prepare_data(self):
        # Разделение данных на признаки и целевую переменную
        X = self.df[self.features]
        y = self.df[self.target]

        # Определение трансформеров для числовых и категориальных данных
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Комбинирование трансформеров с помощью ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        # Применение трансформаций к данным
        X_preprocessed = preprocessor.fit_transform(X)

        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=self.test_size,
                                                            random_state=42)

        return X_train, X_test, y_train, y_test
