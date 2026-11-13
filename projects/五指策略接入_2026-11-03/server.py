"""Loopback viewer; physics/control and GL rendering share one owner thread.

The browser never samples sensors or advances dynamics. A slow renderer drops
display opportunities, not simulation ticks. Snapshot IDs bind force/state/image.
"""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from collections import OrderedDict
from queue import Queue,Empty
from threading import Thread,Lock,Event
from io import BytesIO
import argparse,json,time,traceback
import numpy as np
from PIL import Image,ImageDraw

from session import ROOT,APP_ID,SCENARIOS,TrialSession,check_admission,jsonable,load_json

def png(pixels):
    buffer=BytesIO();Image.fromarray(pixels).save(buffer,format='PNG');return buffer.getvalue()

def placeholder(message):
    image=Image.new('RGB',(1000,700),'#e4edf0');draw=ImageDraw.Draw(image)
    draw.text((35,330),message,fill='#183540');b=BytesIO();image.save(b,format='PNG');return b.getvalue()

class Viewer:
    def __init__(self):
        self.commands=Queue();self.lock=Lock();self.stop=Event();self.sequence=0
        self.frames=OrderedDict();self.snapshot_id=0
        self.state=dict(app=APP_ID,time=0.,running=False,paused=False,mode='scheduled',
            snapshot_id=0,message='等待开始新试验。',normal=[0.]*5,palm=0.,events=[],history=[])
        self.frames[0]=placeholder('Complete hand strategy integration - waiting for an admitted trial')
        self.receipts={}
        self.thread=Thread(target=self.worker,daemon=True)

    def start(self):self.thread.start()

    def enqueue(self,payload):
        with self.lock:self.sequence+=1;sequence=self.sequence
        self.commands.put((sequence,payload));return sequence

    def worker(self):
        session=None;renderer=None;next_sensor=time.perf_counter();next_frame=next_sensor
        def preserve_current():
            if session is not None and not session.saved:
                session.save(ROOT/'output/interactive'/session.run_id)
        try:
            while not self.stop.is_set():
                while True:
                    try:sequence,payload=self.commands.get_nowait()
                    except Empty:break
                    try:
                        action=payload['action'];value=payload.get('value');run_id=payload.get('run_id')
                        if action not in ('new_run','reset') and session is not None and run_id!=session.run_id:
                            raise ValueError('该命令属于上一局，请刷新状态。')
                        if action in ('new_run','reset'):
                            if action=='reset' and session is None:raise ValueError('尚无可复位试验。')
                            recipe=value if action=='new_run' else dict(method=session.method,shape=session.shape,
                                scenario=session.scenario_name,seed=session.seed,mode=session.mode)
                            recipe={k:recipe[k] for k in ('method','shape','scenario','seed','mode') if k in recipe}
                            preserve_current()
                            if renderer is not None:renderer.close();renderer=None
                            session=TrialSession(**recipe)
                            import mujoco
                            renderer=mujoco.Renderer(session.plant.model,height=700,width=1000)
                            next_sensor=time.perf_counter();next_frame=0.
                        else:
                            if session is None:raise ValueError('请先开始新试验。')
                            session.command(action,value,sequence)
                        receipt=dict(sequence=sequence,status='executed',run_id=session.run_id if session else None,
                            tick=session.tick if session else None,time=session.time if session else None)
                    except Exception as exc:
                        receipt=dict(sequence=sequence,status='rejected',error=str(exc))
                        if session is not None:session.event('命令未执行：'+str(exc))
                    with self.lock:
                        self.receipts[sequence]=receipt
                        if receipt['status']=='rejected':
                            self.state={**self.state,'last_command':receipt,'message':receipt['error']}
                now=time.perf_counter()
                if session is not None and session.running and not session.paused:
                    # Control remains a fixed integer tick. It may run slower
                    # than wall time under load, but no physics state is skipped.
                    if now>=next_sensor:
                        session.step_one_sensor_tick();next_sensor+=.005
                        if now-next_sensor>.1:next_sensor=now
                else:next_sensor=now
                if session is not None and renderer is not None and now>=next_frame:
                    image=png(session.render(renderer));state=session.snapshot()
                    with self.lock:
                        self.snapshot_id+=1;state['snapshot_id']=self.snapshot_id
                        state['last_command']=self.receipts[max(self.receipts)] if self.receipts else None
                        self.frames[self.snapshot_id]=image
                        while len(self.frames)>120:self.frames.popitem(last=False)
                        self.state=state
                    next_frame=time.perf_counter()+1/12.
                if session is not None and session.completed and not session.saved:preserve_current()
                time.sleep(.0005)
        except Exception:
            error=traceback.format_exc()
            with self.lock:self.state={**self.state,'running':False,'error':error,'message':'运行失败，原始记录保留。'}
            try:preserve_current()
            except Exception:traceback.print_exc()
            traceback.print_exc()
        finally:
            if renderer is not None:renderer.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,value,mime='application/json; charset=utf-8',status=200):
        body=value if isinstance(value,bytes) else json.dumps(jsonable(value),ensure_ascii=False,allow_nan=False).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        from urllib.parse import urlparse,parse_qs
        url=urlparse(self.path);viewer=self.server.viewer
        if url.path=='/':return self.reply((ROOT/'web/index.html').read_bytes(),'text/html; charset=utf-8')
        if url.path=='/capabilities':
            admitted=check_admission()
            return self.reply(dict(app=APP_ID,admitted=admitted['passed'],problems=admitted['problems'],
                scenarios=SCENARIOS,methods=['ratio','unified'],shapes=['capsule','box','sphere'],
                handoff_time=7.5,port=self.server.server_port))
        if url.path=='/state':
            with viewer.lock:state=dict(viewer.state)
            admission=check_admission()
            if not admission['passed']:state['message']=' '.join(admission['problems'])
            return self.reply(state)
        if url.path=='/frame.png':
            query=parse_qs(url.query)
            try:wanted=int(query.get('snapshot',[viewer.snapshot_id])[0])
            except ValueError:return self.reply({'error':'invalid snapshot'},status=400)
            with viewer.lock:frame=viewer.frames.get(wanted)
            if frame is None:return self.reply({'error':'snapshot expired; poll a fresh state'},status=410)
            return self.reply(frame,'image/png')
        if url.path=='/comparison':
            candidates=list((ROOT/'output').glob('*/analysis/summary.json')) if (ROOT/'output').exists() else []
            if not candidates:return self.reply(dict(rows=[],note='尚无完成的同条件比较。'))
            latest=max(candidates,key=lambda p:p.stat().st_mtime)
            result=load_json(latest);rows=[r for r in result.get('rows',[]) if r.get('domain')=='main']
            prefix='常规场景主域；能力边界在完整报告单列。'
            if rows and all(r.get('split')=='development' for r in rows):prefix='开发接入检查，不用于算法优势结论。'+prefix
            return self.reply(dict(rows=rows,note=prefix+result.get('note',''),source=str(latest)))
        return self.reply({'error':'Not found'},status=404)
    def do_POST(self):
        if self.path!='/command':return self.reply({'error':'Not found'},status=404)
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':return self.reply({'error':'Origin rejected'},status=403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ValueError('invalid command size')
            payload=json.loads(self.rfile.read(length));action=payload.get('action')
            if action not in ('new_run','reset','pause','camera','disturb','squeeze','release'):raise ValueError('unknown command')
            value=payload.get('value')
            if action=='new_run':
                if not isinstance(value,dict):raise ValueError('new_run requires a run configuration')
                if value.get('method','ratio') not in ('ratio','unified'):raise ValueError('unknown method')
                if value.get('shape','capsule') not in ('capsule','box','sphere'):raise ValueError('unknown shape')
                if value.get('scenario','nominal') not in {s['name'] for s in SCENARIOS}:raise ValueError('unknown scenario')
                if value.get('mode','scheduled') not in ('scheduled','manual'):raise ValueError('unknown mode')
                seed=value.get('seed',540101)
                if isinstance(seed,bool) or not isinstance(seed,(int,float)) or not np.isfinite(seed) or int(seed)!=seed or not 1<=seed<2**32:
                    raise ValueError('seed must be a positive 32-bit integer')
            if action=='pause' and not isinstance(value,bool):raise ValueError('pause expects a boolean')
            if action=='camera':
                if not isinstance(value,dict) or any(k not in ('azimuth','elevation','distance') or
                    not isinstance(v,(int,float)) or not np.isfinite(v) for k,v in value.items()):raise ValueError('invalid camera')
            if action in ('new_run','reset'):
                a=check_admission()
                if not a['passed']:return self.reply({'error':' '.join(a['problems'])},status=409)
            sequence=self.server.viewer.enqueue(payload)
            return self.reply({'status':'queued','sequence':sequence})
        except (ValueError,TypeError,KeyError) as exc:return self.reply({'error':str(exc)},status=400)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8880);p.add_argument('--diagnose',action='store_true')
    args=p.parse_args()
    if args.diagnose:
        print(json.dumps(jsonable(check_admission()),ensure_ascii=False));return
    # Server may expose blocked capabilities; actual algorithm construction is
    # still forbidden until admission. No legacy controller fallback exists.
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.viewer=Viewer();server.viewer.start()
    print(f'{APP_ID}: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.viewer.stop.set();server.viewer.thread.join(timeout=10);server.server_close()

if __name__=='__main__':main()
