"""
Route Audit & Contract Reconciliation Script
"""
import urllib.request
import urllib.error
import json
import re
from pathlib import Path

BASE = "http://127.0.0.1:8000"

def get_openapi_routes():
    req = urllib.request.urlopen(f"{BASE}/openapi.json")
    schema = json.loads(req.read())
    routes = []
    for path, methods in schema["paths"].items():
        for method, details in methods.items():
            routes.append({
                "method": method.upper(),
                "path": path,
                "summary": details.get("summary", ""),
                "operation_id": details.get("operationId", ""),
                "parameters": details.get("parameters", []),
                "request_body": details.get("requestBody", None) is not None,
                "responses": list(details.get("responses", {}).keys()),
            })
    return sorted(routes, key=lambda x: (x["path"], x["method"]))

def scan_frontend_consumers():
    frontend_dir = Path("frontend/src")
    calls = []
    for f in frontend_dir.rglob("*.ts*"):
        text = f.read_text(encoding="utf-8")
        matches = re.findall(r"['`](/api/v1/[^'`\?]+)", text)
        for m in matches:
            calls.append((f.relative_to(frontend_dir).as_posix(), m))
    return calls

def main():
    routes = get_openapi_routes()
    print(f"Total OpenAPI Routes: {len(routes)}")
    frontend_calls = scan_frontend_consumers()
    
    for r in routes:
        m, p, s = r["method"], r["path"], r["summary"]
        # Find callers
        callers = set()
        for f, call_path in frontend_calls:
            # normalize template strings like ${scanId} to {scan_id}
            norm_call = re.sub(r"\$\{[^}]+\}", "{id}", call_path)
            norm_route = re.sub(r"\{[^}]+\}", "{id}", p)
            if norm_call == norm_route or call_path.startswith(p.split("{")[0]):
                callers.add(f)
        r["frontend_consumers"] = sorted(list(callers))
        print(f"{m:6} {p:42} | {s:45} | Consumers: {', '.join(r['frontend_consumers']) or 'None (Root/Internal)'}")

    with open("frontend/audit/openapi_routes.json", "w", encoding="utf-8") as out:
        json.dump(routes, out, indent=2)

if __name__ == "__main__":
    main()

