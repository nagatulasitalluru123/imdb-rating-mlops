import json
import os
import time
import urllib.request

OUT = "outputs/monitoring/api_performance_report.json"

def main():
    url = os.getenv("IMDB_API_URL", "http://127.0.0.1:8000/health")
    rows=[]
    for i in range(5):
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                status=response.status
            rows.append({"request": i+1, "status": status, "latency_ms": (time.perf_counter()-start)*1000})
        except Exception as exc:
            rows.append({"request": i+1, "status": "error", "error": str(exc)})
    report={"requests": len(rows), "successful": sum(r.get("status")==200 for r in rows), "average_latency_ms": sum(r.get("latency_ms",0) for r in rows)/len(rows), "details": rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT,"w") as f: json.dump(report,f,indent=4)
    print(f"[SUCCESS] API performance report saved to {OUT}")

if __name__ == "__main__": main()
