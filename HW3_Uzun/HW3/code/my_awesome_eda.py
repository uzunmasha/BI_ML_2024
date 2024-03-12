def run_eda(df):
    '''
    Main function which performs Exploratory Data Analysis (EDA) on the input df DataFrame and prints its statistics.
    '''
    print(f"In the observed table \nThe number of observations/rows is {df.shape[0]} \nThe number of parameters/columns is {df.shape[1]}")

    numerical = []
    categorical = []
    string = []

    for column in df.columns:
        unique_values = df[column].nunique()
        value_type = df[column].dtype

        if (value_type == 'object' and unique_values > 5) or any(substring in column for substring in ['ID', 'Id']):
            string.append(column)
        elif unique_values < 5:
            categorical.append(column)
        else:
            numerical.append(column)
    
    print(f"\nCategorical columns:{categorical}")
    print(f"Numerical columns:{numerical}")
    print(f"String columns:{string}")
    
    for column in categorical:
        print(f"\nCounts: {df[column].value_counts()}")
        print(f"\nFrequencies: {df[column].value_counts(normalize=True)}")

    # Создаем таблицу для вывода числовых статистик
    numerical_stats = df.describe().transpose()
    print("\nNumerical columns statistics:")
    print(numerical_stats)

    total_missing_values = df.isnull().sum().sum()
    rows_with_missing_values = df[df.isnull().any(axis=1)].shape[0]
    columns_with_missing_values = df.columns[df.isnull().any()].tolist()

    print(f"\nTotal missing values in the DataFrame: {total_missing_values}")
    print(f"Number of rows with missing values: {rows_with_missing_values}")
    print(f"Columns with missing values: {columns_with_missing_values}")

    duplicate_rows = df[df.duplicated()]
    num_duplicate_rows = len(duplicate_rows)

    print(f"\nNumber of duplicate rows: {num_duplicate_rows}")
