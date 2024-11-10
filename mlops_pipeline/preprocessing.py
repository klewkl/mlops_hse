import pandas as pd
import warnings
from pathlib import Path
from typing import Union, Tuple
import pandas as pd
from werkzeug.datastructures import FileStorage
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

feature_columns = ['Age', 'Name', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
target_column = 'Survived'
all_columns = feature_columns + [target_column]


def validate_and_prepare_data(
        file: Union[str, Path, FileStorage],
        train: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    Validates and prepares data for training or testing. This function reads the file,
    validates the required columns, processes missing values, encodes categorical columns, 
    and scales numerical features.

    :param file: The path to the CSV file or a file-like object (e.g., FileStorage).
    :param train: Boolean flag to specify if it's training data. If False, it's test data.
    :param test_size: If train=True, this parameter controls the test set size.
    :param random_state: Random seed for reproducibility.
    :return: Tuple of processed training data and target if train=True, else processed test data.
    """
    try:
        if isinstance(file, FileStorage):
            data = pd.read_csv(file.stream)
        else:
            data = pd.read_csv(file)
    except Exception as e:
        raise ValueError(f"Couldn't read file: {e}")

    needed_columns = feature_columns
    if train:
        needed_columns = all_columns

    missing_cols = set(needed_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f'Missing columns in the data: {missing_cols}')

    data = data[needed_columns]

    if train:
        data.dropna(subset=[target_column], inplace=True)

        X = data[feature_columns]
        y = data[target_column]

        if (X['Age'] <= 0).any():
            warnings.warn("Some Age values are invalid (<= 0). These values will be set to NaN.")
            X['Age'] = X['Age'].apply(lambda x: x if x > 0 else pd.NA)

        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X.loc[:, num_cols] = X.loc[:, num_cols].fillna(0)

        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col in ['Sex', 'Embarked']:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('UNK')) 
            elif col == 'Cabin':  
                X[col] = X[col].fillna('UNK') 
            elif col == 'Name': 
                X['Title'] = X['Name'].str.extract('([A-Za-z]+)\\.', expand=False)
                X['Title'] = X['Title'].fillna('UNK')
                le_title = LabelEncoder()
                X['Title'] = le_title.fit_transform(X['Title'])
                X.drop(columns=['Name'], inplace=True) 


        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        return X, y

    else:

        X = data[feature_columns]

        num_cols = X.select_dtypes(include=['float64', 'int64']).columns
        X.loc[:, num_cols] = X.loc[:, num_cols].fillna(0)

        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if col in ['Sex', 'Embarked']:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('UNK')) 
            elif col == 'Cabin': 
                X[col] = X[col].fillna('UNK')  
            elif col == 'Name': 
                X['Title'] = X['Name'].str.extract('([A-Za-z]+)\\.', expand=False)
                X['Title'] = X['Title'].fillna('UNK')
                le_title = LabelEncoder()
                X['Title'] = le_title.fit_transform(X['Title'])
                X.drop(columns=['Name'], inplace=True) 

        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])

        if X.isna().sum().sum():
            raise ValueError("Test data still contains missing values after preprocessing.")

        return X

def split_train_test(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the training data into a training and validation set.

    :param X: Feature matrix.
    :param y: Target variable.
    :param test_size: Proportion of data to be used as the validation set.
    :param random_state: Random seed for reproducibility.
    :return: Train and validation sets (X_train, X_val, y_train, y_val).
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

## Как отрабатывает код: 
# train_file = 'titanic_train.csv'
# test_file = 'titanic_test.csv'

# X_train, y_train = validate_and_prepare_data(train_file, train=True)
# X_train_split, X_val_split, y_train_split, y_val_split = split_train_test(X_train, y_train)
# X_test = validate_and_prepare_data(test_file, train=False)