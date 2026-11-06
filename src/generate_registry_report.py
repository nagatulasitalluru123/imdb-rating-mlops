import os, json, mlflow
from mlflow.tracking import MlflowClient
MODEL_NAME="IMDb_Movie_Rating_Production_Model"

def generate_registry_report():
    mlflow.set_tracking_uri("sqlite:///mlflow.db"); client=MlflowClient(); versions=client.search_model_versions(f"name='{MODEL_NAME}'")
    champion=next((v for v in versions if v.current_stage=="Production"),None)
    if not champion: print("[ERROR] No model in Production stage."); return False
    run=client.get_run(champion.run_id)
    report={"registry_status":"READY_FOR_DEPLOYMENT","model_lineage":{"registered_name":MODEL_NAME,"version":int(champion.version),"current_stage":champion.current_stage,"run_id":champion.run_id,"artifact_uri":champion.source},"performance_metrics":run.data.metrics,"hyperparameters":run.data.params,"preprocessing_dependency":"models/preprocessor.pkl"}
    os.makedirs("artifacts",exist_ok=True)
    with open("artifacts/production_model_report.json","w") as f: json.dump(report,f,indent=4)
    print(f"[SUCCESS] Production registry report generated for Version {champion.version}"); return True
if __name__=="__main__":
    if not generate_registry_report(): raise SystemExit(1)
