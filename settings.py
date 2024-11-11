from pydantic import Field, BaseModel, field_validator, ValidationError, ValidationInfo
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class Settings(BaseModel):
    """
    Settings for model training API
    """
    PORT: int = Field(default=8000)
    HOST: str = Field(default='127.0.0.1')
    MODEL_DIR: Path = Field(default='models', validate_default=True)

    @field_validator('MODEL_DIR')
    def val_model_dir(cls, path: Path) -> Path:
        path.mkdir(exist_ok=True)
        return path


class ServiceStatus(Enum):
    training = 'training'
    waiting = 'waiting'
    error = 'error'


class ModelClass(Enum):
    logreg = 'logistic_regression'
    svm = 'svm'
    random_forest = 'random_forest'
    knn = 'knn'
    decision_tree = 'decision_tree'

    @property
    def converters(self):
        return {
            self.logreg: LogisticRegression,
            self.svm: LinearSVC,
            self.random_forest: RandomForestClassifier,
            self.knn: KNeighborsClassifier,
            self.decision_tree: DecisionTreeClassifier,
        }


class TreeParams(BaseModel):
    max_depth: Optional[int] = Field(default=None)
    min_samples_split: int = Field(default=2)
    min_samples_leaf: int = Field(default=1)


class ForestParams(TreeParams):
    n_estimators: int = Field(default=100)


class Penalty(Enum):
    l2 = 'l2'
    l1 = 'l1'


class LinearParams(BaseModel):
    penalty: Penalty = Field(default=Penalty.l2)
    C: Union[int, float] = Field(default=1)
    max_iter: int = Field(default=100)


class KnnAlgorithm(Enum):
    auto = 'auto'
    ball_tree = 'ball_tree'
    kd_tree = 'kd_tree'
    brute = 'brute'


class KnnWeights(Enum):
    uniform = 'uniform'
    distance = 'distance'


class KnnParams(BaseModel):
    n_neighbors: int = Field(default=5)
    algorithm: KnnAlgorithm = Field(default=KnnAlgorithm.auto)
    weight: KnnWeights = Field(default=KnnWeights.uniform)


class ModelParams(BaseModel):
    ml_model_type: ModelClass = Field(default=ModelClass.logreg)
    # как сделать, чтобы при лишних параметрах питон ругался?
    ml_model_params: Union[TreeParams, ForestParams, LinearParams, KnnParams] = Field(default=LinearParams())
    test_size: float = Field(default=0.33)

    @field_validator('test_size')
    def validate_size(cls, size: float) -> float:
        if (size <= 0) or (size >= 1):
            raise ValidationError('Size should be in interval (0, 1)')
        return size
