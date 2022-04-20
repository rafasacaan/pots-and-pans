---
title:  "workflows with sklearn"
date: 2022-04-11T11:00:00-03:00
draft: True
params:
  ShowShareButtons: true

---

Let´s take advantage of things that are already built. Im referring to **sklearn** and the whole variety of beautiful utilities and functions that help us make our lives simpler and stay curious, testing and trying out new stuff. This will be simple and straight to the bone.


## A. Splits

First, lets take the whole data and leave aside a testing chunk. Then, we can run cross validation on training set and when we think we are ready, we can check our scores on the test set. Don´t fool yourself fitting out over the test set! Its there only for a final check and having a firm-ground metric on how our model can generalize out there in the wild.

Second, let´s define our **kfold** so we can have reproducible results from now on. Two things:
- Choose your **n_folds** wisely: for little data, we need similar distributions over the target, 3-folds may be enough. Otherwise, 5-fold or even a 10-fold may be necessary.

-  Take a minute and think about which type of **folding strategy** to choose. Group folds, stratify multiple labels?

So, our first piece of code should look similar to the following.


```python
from sklearn.model_selection import KFold, train_test_split

X = df.drop(columns=['y'])
y = df.y.values

# Define a test set
folds = 5
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=(1/folds), 
    random_state=0, 
    stratify=y)

# Make folds
kfold = KFold(
    n_splits=folds - 1 , 
    shuffle=True, 
    random_state=2)
```

## B. Data cleaning

This step should take care of cleaning the data so that it is ready to use. For example: strip, filter, transform, divide, concatenate columns, strings, data types, etc. One or multiple **pandas** functions should be enough. When done, generate a csv to keep track of your experiment.


## C. Data preprocessing

Here starts the fun part. For preprocessing, we should focus on partitioning columns into **numerical** and **categorical** and transforming them. Then, concatenate both and gather all the data, so we can pass it on to a model.

The trick here is to use **custom transformers** so we have the flexibility of doing whatever we want. Two cases arise:

```python

from sklearn.base import BaseEstimator, TransformerMixin


# Case #1: no fit method required
class DropFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X_dropped = X.drop(self.variables, axis = 1)
        self.columns = X_dropped.columns
        return X_dropped


# Case #2: fit method required
 class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(drop='first', handle_unknown='ignore')
  
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.ohe.fit(X_)
        return self
      
    def transform(self, X):
        X_ = X.loc[:,self.variables]

        # get one-hot encoded feature in df format
        X_transformed = pd.DataFrame(self.ohe.transform(X_).toarray(), columns= self.ohe.get_feature_names_out())
        
        # Remove columns that are one hot encoded in original df
        X.drop(self.variables, axis= 1, inplace=True)
        
        # Add one hot encoded feature to original df
        X[self.ohe.get_feature_names_out()] = X_transformed[self.ohe.get_feature_names_out()].values
        return X   

```

Understanding custom transformers, now we should be able to build a pipeline in the following fashion.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor


# Categorical transformers
ordinal_encoder = OrdinalEncoder(categories=categories)

categorical_preprocessor = Pipeline(
    steps=[
        #('replacer', custom_replacer(variables=['a','b'])),
        ('encoder', ordinal_encoder)
    ]
)


# Numerical transformers
imputer = SimpleImputer(strategy='mean') 

numerical_preprocessor = Pipeline(
    steps=[
      ('imputer', imputer)
    ]
)

# Complete preprocessor
preprocessor = ColumnTransformer(
    transformers=[
      ('categorical', categorical_preprocessor, categorical_columns),
      ('numerical', numerical_preprocessor, numerical_columns)
    ]
)

# Add estimator
gbrt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', HistGradientBoostingRegressor(categorical_features=range(4)))
])
```

We can also check graphically our pipeline.
```python
from sklearn import set_config

set_config(display="diagram")
gbrt_pipeline

```

## D. Evaluate our model

Now, for a certain set of pre-defined hyperparameters, we can evaluate our pipeline.

```python
def evaluate (model, X, y, cv):
  cv_results = cross_validate(
      model, 
      X,
      y,
      cv=cv,
      scoring={'neg_mean_absolute_error','neg_root_mean_squared_error'},
  )
  rmse = -cv_results['test_neg_root_mean_squared_error']
  mae = -cv_results['test_neg_mean_absolute_error']

  print(
    f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
    f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
  )


evaluate(gbrt_pipeline, X, y, cv=ts_cv)
```


## E. Grid Search

Now, we can test any hyperparameter from the preprocessign pipeline and estimator. 

```python
from sklearn.model_selection import GridSearchCV

def grid_search(model, X, y, params, cv):

    grid_search = GridSearchCV(
    gbrt_pipeline, 
    param_grid=params, 
    cv=ts_cv,
    scoring={'neg_mean_absolute_error','neg_root_mean_squared_error'},
    refit='neg_root_mean_squared_error',
    n_jobs=-1)

    grid_search.fit(X, y)

    print(-grid_search.best_score_)
    print(grid_search.best_params_)


# Run grid search
params = {
    "preprocessor__numerical__imputer__strategy": ['mean','median'],
    "model__learning_rate": [0.01, 0.1],
}

grid_search(my_pipeline, X_train, y_train, params, ts_cv)

```


## F. Special note on metrics!

(Taken from calmcode.io) Often, it is important to create custom metrics that respond to business questions. 

```python
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer

def min_recall_precision(est, X, y_true, sample_weight=None):
    y_pred = est.predict(X)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)

grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 20, 30)]},
    scoring={'precision': make_scorer(precision_score),
             'recall': make_scorer(recall_score),
             'min_both': min_recall_precision}, # custom metric
    refit='min_both',
    return_train_score=True,
    cv=10,
    n_jobs=-1
)
grid.fit(X, y, sample_weight=np.log(1 + df['Amount'] ))
```




