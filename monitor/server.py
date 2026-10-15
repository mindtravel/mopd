"""Localhost-only monitor; read-only, no torch imports or GPU allocation.

/            all runs, newest first (index.html)
/run/<name>  one experiment version (run.html)
"""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs,unquote
from monitor_data import snapshot,cases
from monitor_eval import eval_tables,run_index
BASE=Path(__file__).parent
STATIC={"/":"index.html","/index.js":"index.js","/app.js":"app.js","/common.js":"common.js","/style.css":"style.css"}
MIME={".html":"text/html",".js":"text/javascript",".css":"text/css"}
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  parsed=urlparse(self.path);query=parse_qs(parsed.query)
  try:
   if parsed.path=="/api/index":
    body=json.dumps({"runs":run_index()},ensure_ascii=False).encode();mime="application/json"
   elif parsed.path=="/api":
    key=query.get("run",[""])[0]
    data=snapshot(key);data["evaluation_tables"]=eval_tables(key)
    body=json.dumps(data,ensure_ascii=False).encode();mime="application/json"
   elif parsed.path=="/api/cases":
    key=query.get("run",[""])[0]
    step=query.get("step",["all"])[0];step=None if step=="all" else int(step)
    offset=max(0,int(query.get("offset",["0"])[0]));limit=min(50,max(1,int(query.get("limit",["20"])[0])))
    body=json.dumps(cases(key,step=step,domain=query.get("domain",[""])[0],
     truncated=query.get("truncated",["0"])[0]=="1",offset=offset,limit=limit),ensure_ascii=False).encode();mime="application/json"
   elif parsed.path.startswith("/run/") and unquote(parsed.path[5:]).strip("/"):
    body=(BASE/"run.html").read_bytes();mime="text/html"
   elif parsed.path in STATIC:
    name=STATIC[parsed.path];body=(BASE/name).read_bytes();mime=MIME[Path(name).suffix]
   else:self.send_error(404);return
  except ValueError:self.send_error(400,"Invalid run or filter");return
  self.send_response(200);self.send_header("Content-Type",mime+"; charset=utf-8")
  self.send_header("Cache-Control","no-store");self.end_headers()
  try:self.wfile.write(body)
  except (BrokenPipeError,ConnectionResetError):pass
 def log_message(self,*args):pass
if __name__=="__main__":ThreadingHTTPServer(("127.0.0.1",8765),Handler).serve_forever()
