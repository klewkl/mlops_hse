import pickle
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional
from sklearn.model_selection import train_test_split
from io import StringIO
from settings import ModelParams


def parse_data(data: Union[str, pd.DataFrame], test_size: float = 0.33, random_state: int = 42,
               target: str = 'target', split=True) -> Union[Tuple, pd.DataFrame]:

    if isinstance(data, str):
        data = pd.read_csv(StringIO(data))

    if split:
        if 'split' in data.columns:
            train, test = data[data['split'] == 'train'], data[data['split'] == 'test']
        else:
            train, test = train_test_split(data, test_size=test_size, random_state=random_state)
        cols = [el for el in data.columns if el != target]
        return train[cols], train[target], test[cols], test[target]

    else:
        return data


class Model:
    def __init__(self, params: Optional[ModelParams] = None):
        if params is not None:
            self.model_type = params.ml_model_type.converters[params.ml_model_type]
            self.model = self.model_type(**params.ml_model_params.model_dump())
        else:
            self.model = None

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, model_dir: Path):
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / (self.model_type + '_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, model_dir: Path):
        model_file = model_dir / (self.model_type + '_model.pkl')
        if model_file.is_file():
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f'There is no file named {model_file}')
