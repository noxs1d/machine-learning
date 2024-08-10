from data_preparator import PandasDfCreator
from visualize import plot_target

def main():
    """Creates dataframe with X, y for model training."""

    pandas_df_creator = PandasDfCreator(150, 1000, 0.1)
    train_df, test_df = pandas_df_creator.create_random_df()

    train_df.to_csv('train_df.csv', index=False)
    test_df.to_csv('test_df.csv', index=False)

    plot_target(train_df['X'], train_df['y'], test_df['X'])


if __name__ == "__main__":
    print("train and test data were created and saved!")
    main()
