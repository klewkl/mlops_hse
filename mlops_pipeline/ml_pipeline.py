import pickle
import pandas as pd
from pathlib import Path
from typing import Union, Tuple, Optional
from sklearn.model_selection import train_test_split
from io import StringIO
from settings import ModelParams, ModelClass, TreeParams, ForestParams, LinearParams, KnnParams

def parse_data(data: Union[str, pd.DataFrame], test_size: float = 0.33, random_state: int = 42,
               target: str = 'target', split=False) -> Union[Tuple, pd.DataFrame]:
    if isinstance(data, str):
        data = pd.read_csv(StringIO(data))

    data = data.dropna()
    data_numeric = data.select_dtypes(include=['number'])

    if split:
        if 'split' in data.columns:
            train, test = data[data['split'] == 'train'], data[data['split'] == 'test']
        else:
            train, test = train_test_split(data_numeric, test_size=test_size, random_state=random_state)

        cols = [col for col in data_numeric.columns if col != target]
        return train[cols], train[target], test[cols], test[target]
    else:
        cols = [col for col in data_numeric.columns if col != target]
        return data_numeric[cols] 


class Model:
    def __init__(self, params: Optional[ModelParams] = None):
        if params is not None:
            model_type = params.ml_model_type
            model_class = model_type.converters[model_type]
            model_params = self.filter_params(params.ml_model_params, model_type)
            self.model = model_class(**model_params)  
            self.columns = None  
        else:
            self.model = None
            self.columns = None

    def train(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)
        
    def filter_params(self, model_params, model_type):

        params_dict = model_params.dict(exclude_unset=True)  # Convert model params to dictionary
        
        if model_type == ModelClass.logreg:
            return {key: value for key, value in params_dict.items() if key in ['penalty', 'C', 'max_iter']}
        
        elif model_type == ModelClass.svm:
            return {key: value for key, value in params_dict.items() if key in ['penalty', 'C', 'max_iter']}
        
        elif model_type == ModelClass.random_forest:
            return {key: value for key, value in params_dict.items() if key in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']}
        
        elif model_type == ModelClass.knn:
            return {key: value for key, value in params_dict.items() if key in ['n_neighbors', 'algorithm', 'weight']}
        
        elif model_type == ModelClass.decision_tree:
            return {key: value for key, value in params_dict.items() if key in ['max_depth', 'min_samples_split', 'min_samples_leaf']}
        
        return params_dict

    def save(self, model_dir: str):
        if self.model is None:
            raise ValueError("Model is not initialized. Cannot save a model that is None.")
        
        model_dir_path = Path(model_dir)
        model_dir_path.mkdir(parents=True, exist_ok=True)
        model_path = model_dir_path / (self.model.__class__.__name__ + '_model.pkl')
        
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved at: {model_path}")

    def load(self, model_dir: Path, model_type: Optional[str] = None):
        if self.model is None:
            if model_type is None:
                raise ValueError("Model class has not been initialized and no model type provided.")
            
            model_type_map = {
                "logistic_regression": "logreg",
                "svm": "svm",
                "random_forest": "random_forest",
                "knn": "knn",
                "decision_tree": "decision_tree"
            }
    
            model_type = model_type_map.get(model_type, model_type) 
            
            try:
                model_class_enum = ModelClass[model_type] 
            except KeyError:
                raise ValueError(f"Invalid model type: {model_type}. Must be one of {', '.join(ModelClass.__members__.keys())}")
            
            self.model = model_class_enum.converters[model_class_enum]()
    
        model_file = model_dir / (self.model.__class__.__name__ + '_model.pkl')
        if model_file.is_file():
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from: {model_file}")
        else:
            raise FileNotFoundError(f'Model file not found: {model_file}')

