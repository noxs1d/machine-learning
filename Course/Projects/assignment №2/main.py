import pandas as pd
from data_preparator import DataPreparator
from train_model import ModelTrainer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def main(data_path="/Data/california_housing_train.csv", model_save_path="/model/best_model.pkl"):
    df = pd.read_csv(data_path)

    target = 'median_house_value'
    preparator = DataPreparator(df, target)
    X_train, X_test, y_train, y_test = preparator.prepare_data()


    models_params = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        },
        'DecisionTree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        },
        'KNeighbors': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        },
        'RandomForest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
        }
    }

    trainer = ModelTrainer(models_params)
    best_model, best_params = trainer.fit(X_train, y_train)

    y_pred = trainer.predict(X_test)
    mse = trainer.evaluate(y_test, y_pred)

    print(f'Best Model: {best_model}')
    print(f'Best Parameters: {best_params}')
    print(f'Mean Squared Error on test set: {mse}')

    trainer.save_model(model_save_path)


if __name__ == "__main__":
    data_path = 'D:/machine-learning/Course/Projects/assignment №2/Data/california_housing_train.csv'
    model_save_path = 'best_model.pkl'
    main(data_path, model_save_path)
