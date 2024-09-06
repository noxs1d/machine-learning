import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


class ModelTrainer:
    def __init__(self, models_params: dict, scoring: str = 'neg_mean_squared_error'):
        self.models_params = models_params
        self.scoring = scoring
        self.best_model = None
        self.best_params = None

    def fit(self, X_train, y_train):
        best_score = float('inf')

        for model_name, model_param in self.models_params.items():
            grid_search = GridSearchCV(model_param['model'], model_param['params'], scoring=self.scoring, cv=5,
                                       n_jobs=-1)
            grid_search.fit(X_train, y_train)
            if grid_search.best_score_ < best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_

        return self.best_model, self.best_params

    def predict(self, X_test):
        return self.best_model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        return mean_squared_error(y_test, y_pred)

    def save_model(self, file_path):
        if self.best_model:
            joblib.dump(self.best_model, file_path)
            print(f'Model saved to {file_path}')
        else:
            print('No model to save.')
