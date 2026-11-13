"""Explicitly launched loopback-only recorded-result viewer, no new trials."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import argparse,json
from .reader import ResultStore,DEFAULT_REGISTRATION,clean

WEB=Path(__file__).resolve().parent/'web/index.html'


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,value,mime='application/json; charset=utf-8',status=200,headers=None):
        body=value if isinstance(value,bytes) else json.dumps(clean(value),ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)));self.send_header('X-Content-Type-Options','nosniff')
        for key,val in (headers or {}).items():self.send_header(key,str(val))
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
            return self.reply({'error':'Local viewer host required'},status=403)
        url=urlparse(self.path);query=parse_qs(url.query)
        def one(name,default=None):
            values=query.get(name,[default])
            if len(values)!=1 or values[0] is None:raise ValueError('Missing/singular '+name+' required')
            return values[0]
        try:
            if url.path=='/':return self.reply(WEB.read_bytes(),'text/html; charset=utf-8')
            if url.path=='/api/catalog':return self.reply(self.server.store.catalog())
            if url.path not in ('/api/context','/api/sample','/api/frame'):return self.reply({'error':'Not found'},status=404)
            scenario=one('scenario');motion=one('motion');seed=int(one('seed'))
            if url.path=='/api/context':return self.reply(self.server.store.context(scenario,motion,seed))
            time_s=float(one('time'))
            if url.path=='/api/sample':return self.reply(self.server.store.sample(scenario,motion,seed,time_s))
            payload,record_time=self.server.renderer.image(scenario,motion,seed,one('method'),time_s,one('view','oblique'))
            return self.reply(payload,'image/png',headers={'X-Replay-Time-S':record_time})
        except (ValueError,KeyError) as exc:return self.reply({'error':str(exc)},status=400)
        except Exception as exc:return self.reply({'error':'回放不可用：'+str(exc)},status=503)

    def do_POST(self):return self.reply({'error':'Read-only playback; no trial/control commands'},status=405)


def main():
    parser=argparse.ArgumentParser(description='已运行结果回放：不重新仿真')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect',action='store_true',help='Read catalog only; no server, model or renderer')
    mode.add_argument('--serve',action='store_true',help='Explicitly start the loopback viewer')
    parser.add_argument('--port',type=int,default=8892);parser.add_argument('--registration',type=Path,default=DEFAULT_REGISTRATION)
    args=parser.parse_args();store=ResultStore(args.registration)
    if args.inspect:
        print(json.dumps(dict(status=store.catalog()['status'],contexts=len(store.catalog()['contexts']),
            methods=3,complete_records=sum(r['available'] for r in store.records.values()),
            pending_batches=store.pending,server_started=False,renderer_created=False,new_physics_steps=0),ensure_ascii=False));return
    if not 1024<=args.port<=65535:raise ValueError('Valid nonprivileged local port required')
    if store.pending:raise RuntimeError('Full48 archive is not complete; no partial comparison is served')
    if not WEB.is_file():raise FileNotFoundError('Playback page is not ready')
    # Bind first: an occupied port fails visibly, never replaces another app.
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    from .rendering import ReplayRenderer
    server.store=store;server.renderer=ReplayRenderer(store);server.renderer.start()
    print(f'已运行结果回放：http://127.0.0.1:{args.port} （不重新仿真）',flush=True)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:server.server_close();server.renderer.close()


if __name__=='__main__':main()
