from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(random_state=42)
df = pd.DataFrame(X)
df['target'] = y

# df.to_csv('data.csv', index=False)
