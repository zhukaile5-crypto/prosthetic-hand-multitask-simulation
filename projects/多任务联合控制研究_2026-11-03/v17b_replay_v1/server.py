"""Explicit loopback-only archived viewer. --inspect creates no renderer."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import argparse,json
from .reader import ROOT,ResultStore,DEFAULT_PLAN,PLAN_SHA,clean,read,sha

WEB=Path(__file__).resolve().parent/'web/index.html'
DEFAULT_AUDIT=ROOT/'review/v17b_prospective80_execution_independent_v1.json'

def check_audit(path):
    report=read(path)
    if report.get('schema')!='v17b_prospective80_execution_independent_v1' or report.get('passed') is not True:
        raise ValueError('Full80 independent execution audit must pass before serving')
    if report.get('plan_sha256')!=PLAN_SHA or report.get('assigned_count')!=80 or report.get('all_assigned_checked') is not True:
        raise ValueError('Audit/plan/full assignment differs')
    plan=read(DEFAULT_PLAN)
    if sha(DEFAULT_PLAN)!=PLAN_SHA or report.get('source_sha256')!=plan['source_sha256'] or report.get('environment')!=plan['environment']:
        raise ValueError('Audit source136/environment differs')
    chain_path=ROOT/'review/v17b_prospective80_completion_chain_v1.json';chain=read(chain_path)
    continuation=report.get('retained16_resume64_chain') or {}
    if (chain.get('complete') is not True or chain.get('original_plan_sha256')!=PLAN_SHA
        or chain.get('source_sha256')!=plan['source_sha256'] or chain.get('environment')!=plan['environment']
        or chain.get('retained_trials')!=16 or chain.get('continued_trials')!=64 or chain.get('total_completed')!=80
        or continuation.get('completion_chain_sha256')!=sha(chain_path) or continuation.get('retained_data_unchanged') is not True):
        raise ValueError('Audit does not bind the complete retained16/resumed64 chain')
    for relative,key in (('review/v17b_prospective80_progress_v1.json','stopped_progress_sha256'),
        ('review/v17b_prospective80_resume_plan_v1.json','resume_plan_sha256'),
        ('review/v17b_prospective80_resume_progress_v1.json','resume_progress_sha256')):
        if sha(ROOT/relative)!=chain.get(key):raise ValueError('Completion receipt changed after audit')
    return report

def bind_audit(store,report):
    rows=report.get('rows',[]);by={str(Path(r['run']).resolve()):r for r in rows}
    folders={str(r['folder'].resolve()) for r in store.records.values()}
    if len(rows)!=80 or len(by)!=80 or set(by)!=folders or any(r.get('status')!='audited' for r in rows):
        raise ValueError('Audit must bind all80 indexed record directories')
    for folder,row in by.items():
        for filename,key in (('metadata.json','metadata_sha256'),('result.json','result_sha256')):
            if sha(Path(folder)/filename)!=row.get(key):raise ValueError('Audited record changed: '+filename)
    store.audit_fingerprints=by

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,value,mime='application/json; charset=utf-8',status=200,headers=None):
        body=value if isinstance(value,bytes) else json.dumps(clean(value),ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)));self.send_header('X-Content-Type-Options','nosniff')
        for k,v in (headers or {}).items():self.send_header(k,str(v))
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'):
            return self.reply({'error':'Local viewer host required'},status=403)
        url=urlparse(self.path);query=parse_qs(url.query)
        def one(name,default=None):
            v=query.get(name,[default])
            if len(v)!=1 or v[0] is None:raise ValueError('One '+name+' is required')
            return v[0]
        try:
            if url.path=='/':return self.reply(WEB.read_bytes(),'text/html; charset=utf-8')
            if url.path=='/api/catalog':return self.reply(self.server.store.catalog())
            if url.path not in ('/api/context','/api/sample','/api/frame'):return self.reply({'error':'Not found'},status=404)
            scenario=one('scenario');motion=one('motion');seed=int(one('seed'))
            if url.path=='/api/context':return self.reply(self.server.store.context(scenario,motion,seed))
            time_s=float(one('time'))
            if url.path=='/api/sample':return self.reply(self.server.store.sample(scenario,motion,seed,time_s))
            body,t=self.server.renderer.image(scenario,motion,seed,one('method'),time_s,one('view','oblique'))
            return self.reply(body,'image/png',headers={'X-Replay-Time-S':t})
        except (ValueError,KeyError) as exc:return self.reply({'error':str(exc)},status=400)
        except Exception as exc:return self.reply({'error':'回放不可用：'+str(exc)},status=503)
    def do_POST(self):return self.reply({'error':'Read-only recorded playback'},status=405)

def main():
    p=argparse.ArgumentParser(description='五方法已运行结果回放：不重新仿真')
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--inspect',action='store_true');mode.add_argument('--serve',action='store_true')
    p.add_argument('--port',type=int,default=8894);p.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    p.add_argument('--audit',type=Path,default=DEFAULT_AUDIT);a=p.parse_args();store=ResultStore(a.plan)
    if a.inspect:
        c=store.catalog();print(json.dumps(dict(status=c['status'],contexts=len(c['contexts']),roles=5,complete_records=len(store.records),
            pending_batches=store.pending,blocked_batches=store.blocked,server_started=False,renderer_created=False,new_physics_steps=0),ensure_ascii=False));return
    if not 1024<=a.port<=65535 or a.port in (8878,8880,8892,8893):raise ValueError('Use a separate nonprivileged local port (default8894)')
    if store.catalog()['status']!='ready':raise RuntimeError('All80 must complete before any comparison is served')
    audit=check_audit(a.audit);bind_audit(store,audit)
    server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler)
    from .rendering import ReplayRenderer
    server.store=store;server.renderer=ReplayRenderer(store);server.renderer.start()
    print(f'五方法已运行结果回放：http://127.0.0.1:{a.port} （不重新仿真）',flush=True)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:server.server_close();server.renderer.close()

if __name__=='__main__':main()
