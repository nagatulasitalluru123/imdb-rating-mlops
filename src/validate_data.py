import os, json
import pandas as pd

REQUIRED={
 "Rank":"numeric", "Title":"string", "Genre":"string", "Description":"string", "Director":"string",
 "Actors":"string", "Year":"numeric", "Runtime (Minutes)":"numeric", "Rating":"numeric", "Votes":"numeric",
 "Revenue (Millions)":"numeric", "Metascore":"numeric"
}

def validate_schema(df, report_name="schema_validation_report.json"):
    errors=[]
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: errors.append(f"Missing columns: {missing}")
    if not errors:
        for c,t in REQUIRED.items():
            if t=="numeric" and not pd.api.types.is_numeric_dtype(df[c]): errors.append(f"{c} must be numeric")
        for c in ["Rating","Metascore"]:
            if c in df and df[c].dropna().size and ((df[c].dropna()<0).any()): errors.append(f"{c} contains negative values")
        if "Rating" in df and (df["Rating"].dropna()>10).any(): errors.append("Rating contains values above 10")
        if "Runtime (Minutes)" in df and (df["Runtime (Minutes)"].dropna()<0).any(): errors.append("Runtime contains negative values")
        if "Votes" in df and (df["Votes"].dropna()<0).any(): errors.append("Votes contains negative values")
    report={"validation_status":"PASSED" if not errors else "FAILED","rows":len(df),"columns":list(df.columns),"missing_values":df.isna().sum().to_dict(),"duplicate_rows":int(df.duplicated().sum()),"errors":errors}
    os.makedirs("artifacts",exist_ok=True)
    with open(os.path.join("artifacts",report_name),"w") as f: json.dump(report,f,indent=4,default=str)
    if errors:
        print("[ERROR] Data validation failed:"); [print(" -",e) for e in errors]; return False
    print("[SUCCESS] Data validation passed."); return True

if __name__=="__main__":
    path="data/raw/IMDB-Movie-Data.csv"
    if not os.path.exists(path): raise FileNotFoundError(f"Dataset not found: {path}")
    if not validate_schema(pd.read_csv(path)): raise SystemExit(1)
