"""Loopback-only interactive viewer. Physics and GL rendering share one worker."""
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from queue import Queue,Empty
import argparse,json,threading,time,traceback
import mujoco
from simulation import GraspSimulation,png_bytes

ROOT=Path(__file__).resolve().parent
commands=Queue();lock=threading.Lock();shared={'state':{'loading':True},'frame':b''}
stop=threading.Event()

def worker():
    try:
        sim=GraspSimulation();renderer=mujoco.Renderer(sim.model,height=700,width=1000)
        paused=False;next_step=time.perf_counter();next_frame=next_step
        while not stop.is_set():
            while True:
                try:action,value=commands.get_nowait()
                except Empty:break
                if action=='pause':paused=bool(value)
                else:sim.command(action,value)
            now=time.perf_counter()
            if not paused:
                # Limit catch-up after a slow frame; no skipped physics timesteps.
                count=min(60,max(0,int((now-next_step)/.001)))
                if count:sim.step(count);next_step+=count*.001
                if now-next_step>.1:next_step=now
            else:next_step=now
            if now>=next_frame:
                frame=png_bytes(sim.render(renderer));state=sim.snapshot();state['paused']=paused
                with lock:shared.update(state=state,frame=frame)
                next_frame=time.perf_counter()+1/12
            time.sleep(.001)
        renderer.close()
    except Exception:
        traceback.print_exc()
        with lock:shared['state']={'error':traceback.format_exc()}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,body,mime,status=200):
        self.send_response(status);self.send_header('Content-Type',mime)
        self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=self.path.split('?')[0]
        if path=='/':self.reply((ROOT/'index.html').read_bytes(),'text/html; charset=utf-8')
        elif path=='/state':
            with lock:body=json.dumps(dict(shared['state'],app='pbl-five-finger-grasp-v1'),ensure_ascii=False).encode('utf-8')
            self.reply(body,'application/json; charset=utf-8')
        elif path=='/frame.png':
            with lock:body=shared['frame']
            self.reply(body,'image/png')
        else:self.reply(b'Not found','text/plain',404)
    def do_POST(self):
        if self.path!='/command':self.reply(b'Not found','text/plain',404);return
        # Reject cross-site mutation; the UI and API are same-origin on localhost.
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://127.0.0.1:{self.server.server_port}':
            self.reply(b'Forbidden','text/plain',403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length>4096:raise ValueError('Too large')
            data=json.loads(self.rfile.read(length));action=data['action'];value=data.get('value')
            if action not in ['reset','grasp','release','disturb','squeeze','slip','force','gate','limit','friction','camera','pause']:raise ValueError('Unknown command')
            if action in ['slip','force','gate','pause'] and not isinstance(value,bool):raise ValueError('Expected boolean')
            if action in ['limit','friction'] and not isinstance(value,(float,int)):raise ValueError('Expected number')
            if action=='camera':
                if not isinstance(value,dict) or not all(k in ['azimuth','elevation','distance'] and isinstance(v,(float,int)) for k,v in value.items()):raise ValueError('Invalid camera')
            commands.put((action,value));self.reply(b'{"ok":true}','application/json')
        except (ValueError,KeyError,TypeError):self.reply(b'{"ok":false}','application/json',400)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8878);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    thread=threading.Thread(target=worker,daemon=True);thread.start()
    print(f'PBL viewer: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:stop.set();thread.join(timeout=3);server.server_close()
