from mlflow.tracking import MlflowClient
import pandas as pd

def mlflow_df(modelname):
  
  client = MlflowClient()
  models = []
  for mv in client.search_model_versions(f"name='{modelname}'"):
      mv = dict(mv)
      models.append(mv)
  models = pd.DataFrame(models)

  model_params = []
  model_tags = []
  model_metrics = []
  
  for mv in client.search_model_versions(f"name='{modelname}'"):
    run_id = mv.run_id
    run = client.get_run(run_id)
    model_params.append(run.data.params)
    model_tags.append(run.data.tags)
    model_metrics.append(run.data.metrics)
  
  model_params = pd.DataFrame(model_params)
  model_tags = pd.DataFrame(model_tags)
  model_metrics = pd.DataFrame(model_metrics)
  
  model_performance = pd.DataFrame()
  model_performance = models.merge(model_tags,left_index=True, right_index=True)
  model_performance = model_performance.merge(model_params,left_index=True, right_index=True)
  model_performance = model_performance.merge(model_metrics,left_index=True, right_index=True)
  
  return model_performance
