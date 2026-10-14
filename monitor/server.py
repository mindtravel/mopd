"""Localhost-only monitor; read-only, no torch imports or GPU allocation."""
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
from monitor_data import snapshot,cases
BASE=Path(__file__).parent
class Handler(BaseHTTPRequestHandler):
 def do_GET(self):
  parsed=urlparse(self.path);query=parse_qs(parsed.query)
  try:
   key=query.get("run",["dual"])[0]
   if parsed.path=="/api":
    body=json.dumps(snapshot(key),ensure_ascii=False).encode();mime="application/json"
   elif parsed.path=="/api/cases":
    step=query.get("step",["all"])[0];step=None if step=="all" else int(step)
    offset=max(0,int(query.get("offset",["0"])[0]));limit=min(50,max(1,int(query.get("limit",["20"])[0])))
    body=json.dumps(cases(key,step=step,domain=query.get("domain",[""])[0],
     truncated=query.get("truncated",["0"])[0]=="1",offset=offset,limit=limit),ensure_ascii=False).encode();mime="application/json"
   elif parsed.path in ("/","/app.js","/style.css"):
    name={"/":"index.html","/app.js":"app.js","/style.css":"style.css"}[parsed.path]
    body=(BASE/name).read_bytes();mime={"/":"text/html","/app.js":"text/javascript","/style.css":"text/css"}[parsed.path]
   else:self.send_error(404);return
  except ValueError:self.send_error(400,"Invalid filter");return
  self.send_response(200);self.send_header("Content-Type",mime+"; charset=utf-8")
  self.send_header("Cache-Control","no-store");self.end_headers()
  try:self.wfile.write(body)
  except (BrokenPipeError,ConnectionResetError):pass
 def log_message(self,*args):pass
if __name__=="__main__":ThreadingHTTPServer(("127.0.0.1",8765),Handler).serve_forever()
