from kedro.pipeline import node, Pipeline
from mlflow import log_artifact
from .nodes import prepare_data, split_train_test, train_model, predict_with_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=prepare_data,
                inputs='kobe_data',
                outputs='data_filtered',
                name="prepare_data_node",
            ),
            node(
                func=split_train_test,
                inputs=['data_filtered',                 
                        'params:test_size',
                        'params:test_split_random_state'],
                outputs=['data_train', 'data_test'],
                name="split_train_test_node",
            ),
            node(
                func=train_model,
                inputs='data_train',
                outputs='model',
                name="train_model_node",
            ),
            node(
               func=predict_with_model,
                inputs='kobe_data',
                outputs='predictions',
                name="predict_with_model_node",
            ),
        ]
    )