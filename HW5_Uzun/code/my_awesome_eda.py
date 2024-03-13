def run_eda(df):
    '''
    Main function which performs Exploratory Data Analysis (EDA) on the input df DataFrame and prints its statistics.
    '''
    print(f"In the observed table \nThe number of observations/rows is {df.shape[0]} \nThe number of parameters/columns is {df.shape[1]}\n")

    numerical = []
    categorical = []
    string = []

    for column in df.columns:
        unique_values = df[column].nunique()
        print(f'Number of unique values in column {column}: {unique_values}')
        value_type = df[column].dtype

        if (value_type == 'object' and unique_values > 9) or any(substring in column for substring in ['ID', 'Id']):
            string.append(column)
        elif unique_values < 8:
            categorical.append(column)
        else:
            numerical.append(column)
    
    print(f"\nCategorical columns:{categorical}")
    print(f"Numerical columns:{numerical}")
    print(f"String columns:{string}")

    # Создаем таблицу для вывода числовых статистик
    numerical_stats = df.describe().transpose()
    print("\nNumerical columns statistics:")
    display(numerical_stats)

    total_missing_values = df.isnull().sum().sum()
    rows_with_missing_values = df[df.isnull().any(axis=1)].shape[0]
    columns_with_missing_values = df.columns[df.isnull().any()].tolist()

    for column in columns_with_missing_values:
        missing_values_count = df[column].isnull().sum()
        print(f"Column '{column}' has {missing_values_count} missing values.")


    duplicate_rows = df[df.duplicated()]
    num_duplicate_rows = len(duplicate_rows)

    print(f"\nNumber of duplicate rows: {num_duplicate_rows}")