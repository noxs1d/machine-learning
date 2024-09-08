import pandas as pd
from data_preparator import DataPreparator
from train_model import ModelTrainer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

def main(data_path="/Data/car details v4.csv", model_save_path="/model/best_model.pkl"):
    # Загрузка данных
    df = pd.read_csv(data_path)

    # Подготовка данных
    target = 'Price'  # укажите вашу целевую переменную
    preparator = DataPreparator(df, target)
    X_train, X_test, y_train, y_test = preparator.prepare_data()

    # Определение моделей и параметров для GridSearch
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

    # Обучение модели
    trainer = ModelTrainer(models_params)
    best_model, best_params = trainer.fit(X_train, y_train)

    # Оценка модели
    y_pred = trainer.predict(X_test)
    mse = trainer.evaluate(y_test, y_pred)

    # Вывод результатов
    print(f'Best Model: {best_model}')
    print(f'Best Parameters: {best_params}')
    print(f'Mean Squared Error on test set: {mse}')

    # Сохранение модели
    trainer.save_model(model_save_path)


if __name__ == "__main__":
    data_path = 'D:/machine-learning/Course/Projects/assignment №2/Data/car details v4.csv'  # замените на путь к вашему датасету
    model_save_path = 'best_model.pkl'  # путь для сохранения модели
    main(data_path, model_save_path)
