import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score
from pycaret.classification import *
import kedro
from kedro.io import DataCatalog
from kedro.pipeline import node, Pipeline
from kedro.runner import SequentialRunner
#from kedro_mlflow import MLflowArtifactDataSet
import requests

mlflow.set_tracking_uri("http://localhost:5000")

def prepare_data(data):
    data_2pt = data[data['shot_type'] == '2PT Field Goal'][['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']].dropna()
    return data_2pt

def split_train_test(df, test_size, random_state):
    features = df.drop(columns=['shot_made_flag'])
    target = df['shot_made_flag']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    mlflow.end_run()
    with mlflow.start_run(run_name="PreparacaoDados") as run:

    # Registro dos parâmetros no MLflow
        mlflow.log_param("dataset size", df.shape[0])
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Registro das métricas no MLflow
        mlflow.log_metric("train_size", train_df.shape[0])
        mlflow.log_metric("test_size", test_df.shape[0])

    mlflow.end_run()

    return train_df, test_df

def train_model(train_df):
    df = train_df.drop(columns=['shot_made_flag'])

    setup(data=train_df, target='shot_made_flag', log_experiment=True)
    lr = create_model('lr')
    lr_pred = predict_model(lr, data=df)
    logloss_lr = log_loss(y_true=train_df['shot_made_flag'], y_pred=lr_pred['prediction_score'])

    model = compare_models(fold=5, cross_validation=True)
    tuned_model = tune_model(model , optimize='Accuracy', choose_better=True,verbose=True)
    final_model = finalize_model(tuned_model)
    final_pred = predict_model(final_model, data=df)

    log_loss_score_model = log_loss(y_true=train_df['shot_made_flag'], y_pred=final_pred['prediction_score'])
    f1_score_val_model = f1_score(y_true=train_df['shot_made_flag'], y_pred=final_pred['prediction_label'])

    mlflow.end_run()
    with mlflow.start_run(run_name="Treinamento") as run:
    # Registro da função custo no MLflow
        mlflow.log_param("model", model)
        mlflow.log_metric("log_loss Logistic_Regression", logloss_lr)
        mlflow.log_metric("log_loss best model", log_loss_score_model)
        mlflow.log_metric("F1_score best model", f1_score_val_model)
        mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()

    return model

def predict_with_model(data: pd.Series):
    
    data_3pt = data[data['shot_type'] == '3PT Field Goal'][['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']].dropna()

    #print(data_3pt.to_dict(orient='records'))

    response = requests.post(
        'http://localhost:5001/invocations',   
        json={  "dataframe_records":  data_3pt.drop(columns=['shot_made_flag']).to_dict(orient='records'),
  }
    )
    
    result = response.json()
    print(result)
    preds = result['predictions']
    log_loss_score_model = log_loss(y_true=data_3pt['shot_made_flag'], y_pred=preds)
    f1_score_model = f1_score(y_true=data_3pt['shot_made_flag'], y_pred=preds)

    mlflow.end_run()
    with mlflow.start_run(run_name="Modelo por API") as run:
    # Registro da função custo no MLflow
        mlflow.log_metric("log_loss model API", log_loss_score_model)
        mlflow.log_metric("F1_score model API", f1_score_model)
    mlflow.end_run()

    return preds
