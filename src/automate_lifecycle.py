import mlflow
from mlflow.tracking import MlflowClient
MODEL_NAME="IMDb_Movie_Rating_Production_Model"

def automate_champion_challenger():
    mlflow.set_tracking_uri("sqlite:///mlflow.db"); client=MlflowClient(); metric="r2"
    versions=client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions: print("[INFO] No registered models found."); return
    for mv in versions:
        if mv.current_stage=="None":
            client.transition_model_version_stage(name=MODEL_NAME,version=mv.version,stage="Staging",archive_existing_versions=False)
    versions=client.search_model_versions(f"name='{MODEL_NAME}'")
    staging=[v for v in versions if v.current_stage=="Staging"]
    if not staging: print("[INFO] No staging challenger found."); return
    best=max(staging,key=lambda v: client.get_run(v.run_id).data.metrics.get(metric,float("-inf")))
    challenger_score=client.get_run(best.run_id).data.metrics.get(metric,0.0)
    production=[v for v in versions if v.current_stage=="Production"]
    champion=production[0] if production else None
    champion_score=client.get_run(champion.run_id).data.metrics.get(metric,0.0) if champion else None
    if champion is None or challenger_score>champion_score:
        client.transition_model_version_stage(name=MODEL_NAME,version=best.version,stage="Production",archive_existing_versions=False)
        if champion: client.transition_model_version_stage(name=MODEL_NAME,version=champion.version,stage="Archived",archive_existing_versions=False)
        print(f"[SUCCESS] Version {best.version} promoted to Production.")
    else:
        print(f"[INFO] Champion retained. Challenger R2={challenger_score:.4f}, Champion R2={champion_score:.4f}.")
    versions=client.search_model_versions(f"name='{MODEL_NAME}'")
    for v in versions:
        if v.current_stage=="Staging" and v.version!=best.version:
            client.transition_model_version_stage(name=MODEL_NAME,version=v.version,stage="Archived",archive_existing_versions=False)

if __name__=="__main__": automate_champion_challenger()
