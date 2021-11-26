from sqlite3.dbapi2 import Timestamp
import traceback
from util import Logger, LogRedirect
from model import Model

import tornado.ioloop
import tornado.web
import tornado.websocket
from tornado.web import StaticFileHandler

import signal
import sys
import os
import base64
import json

from concurrent.futures import ThreadPoolExecutor

model = None
global_open=False

tpe = ThreadPoolExecutor(max_workers=2)

def json_decode(o):
    # Note the "unicode" part is only for python2
    if isinstance(o, str):
        if o.lower() == "true":
            return True
        elif o.lower() == "false":
            return False
        else:
            try:
                return int(o)
            except ValueError:
                try:
                    return float(o)
                except ValueError:
                    return o
    elif isinstance(o, dict):
        return {k: json_decode(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [json_decode(v) for v in o]
    else:
        return o

def json_encode(o):
    if isinstance(o, Timestamp):
        return str(o)
    
class DSHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json")

    def post(self, dataset:str='train'):
        global model
        
        if not LogRedirect.out_redirected:
            print("ds call for %s" % dataset)
            #print(urllib.parse.unquote(self.request.body.decode("utf-8")))
        
        draw = self.get_argument("draw")
        start = int(self.get_argument("start", 0))
        count = int(self.get_argument("length", 20))
        search = self.get_argument("search[value]", None)
        
        from reporting import search_dataset
        if model:
            tosend = search_dataset(model, set=dataset, criteria=search, start=start, count=count)
        else:
            tosend = {
                'recordsTotal': 0,
                'recordsFiltered': 0,
            }
        tosend['draw'] = draw
        msg = json.dumps(tosend, default=json_encode)
        # from pprint import pprint
        # pprint(tosend)
        if not LogRedirect.out_redirected:
            print("write data(len=%d)" % len(msg))
        self.write(msg)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

def get_model():
    global model, ioloop
    if model is None:
        model = Model()
    model.log.ioloop = ioloop
    return model

def load_model(ws, param:dict = None):
    global model
    if model is not None:
        log = model.log
    else:
        log = Logger()
        log.ws = ws
    log.working("Loading model...")
    with LogRedirect(log):
        path = param["path"] if param and "path" in param else "uploaded.model"
        model = Model.load(log, path)
    log.finished("Model loaded")

def new_model(ws, param:dict = None):
    global model, ioloop
    model = Model()
    model.log.ioloop = ioloop
    print("New model created")

def list_checkpoint(model:Model):
    log = model.log
    try:
        from glob import glob
        from os.path import basename
        chkpoints = [basename(str) for str in glob("./saves/checkpoints/*.model")]
        log.finished("Checkpoint list", {
            'checkpoints': chkpoints
        })
    except Exception:
        traceback.print_exc()
        log.error("Server side error")

def delete_checkpoint(model:Model, param:dict):
    log = model.log
    if "name" not in param:
        log.invalid()
        return
    try:
        from os import remove
        remove("./saves/checkpoints/%s" % param["name"])
        log.finished("Checkpoint %s deleted" % param["name"])
    except Exception:
        traceback.print_exc()
        log.error("Server side error")
    
def load_checkpoint(ws, param:dict):
    global model
    try:
        if model is not None:
            log = model.log
        else:
            log = Logger()
            log.ws = ws
        if "name" not in param:
            log.invalid()
            return
        log.working("Loading checkpoint...")
        with LogRedirect(log):
            path = "./saves/checkpoints/%s" % param["name"]
            model = Model.load(log, path)
        log.finished("Checkpoint loaded")
    except Exception:
        traceback.print_exc()
        log.error("Server side error")

def save_checkpoint(model:Model):
    log = model.log
    try:
        log.working("Saving as %s" % model.name)
        if model.save("./saves/checkpoints/%s.model" % model.name):
            log.finished("Checkpoint saved")
        else:
            log.error("Save checkpoint failed")
    except Exception:
        traceback.print_exc()
        log.error("Server side error")

def abort_action(model:Model):
    global tpe
    tpe.shutdownNow()
    tpe = ThreadPoolExecutor(max_workers=2)
    model.log.finished("All actions aborted")
    
class WebSocketHandler(tornado.websocket.WebSocketHandler):
    def open(self):
        global global_open
        model = get_model()
        if not global_open:
            global_open=True
            self.model_owner=True
            model.log.ws = self
        else:
            self.model_owner=False
        print("WS Opened")
    
    def on_close(self):
        global global_open
        model = get_model()
        if self.model_owner:
            global_open=False
            self.model_owner=False
            if model:
                model.log.ws = None
        print("WS Closed")
        
    def send_result(self, _id, msg, param=None):
        model = get_model()
        if param is None:
            param = {"result": _id, "message": msg}
        else:
            param["result"] = _id
            param["message"] = msg
        param["info"] = {
            "name": model.name,
            "trained": model.group_summaries is not None 
        }
        # param["has_dataset"] = texts is not None
        # param["has_model"] = group_model is not None
        # param["has_result"] = groups is not None
        # param["is_busy"] = busy
        msg = json.dumps(param, default=json_encode)
        if not LogRedirect.out_redirected:
            if len(msg) < 128:
                print("write\n %r" % msg)
            else:
                print("write len(%d)" % len(msg))
        self.write_message(msg)

    def on_message(self, msg):
        if not self.model_owner:
            self.send_result(-1, "Model is in use by others")
            return
        
        model = get_model()
        log = model.log

        if isinstance(msg, str):
            msg = json.loads(msg, object_hook=json_decode)
            if not "action" in msg:
                self.send_result(-999, "Unknown request")
                return
            if not LogRedirect.out_redirected:
                print("\n>> %r" % msg["action"])
            if msg["action"]=="ping":
                log.finished("OK")
            elif msg["action"]=="upload":
                if not "flag" in msg:
                    log.invalid()
                    return
                if "data" in msg:
                    data = msg["data"]
                    data = data.split(';base64')[1].encode('utf-8')
                    data = base64.b64decode(data)
                fname = msg["filename"] if "filename" in msg else "temp_upload"
                if msg["flag"]=="begin":
                    if not data:
                        log.invalid()
                        return
                    with open(fname, 'wb') as file:
                        file.write(data)
                elif msg["flag"]=="continue":
                    if not data:
                        log.invalid()
                        return
                    with open(fname, 'ab') as file:
                        file.write(data)
                elif msg["flag"] == "end":
                    if not "size" in msg:
                        log.invalid()
                    elif os.path.getsize(fname) != msg["size"]:
                        log.invalid("Upload size mismatch")
                    else:
                        log.finished("Upload success")
                    return
                log.working("Continue")
            elif msg["action"]=="new_model":
                tpe.submit(new_model, self, msg)
            elif msg["action"]=="load_model":
                tpe.submit(load_model, self, msg)
            elif msg["action"]=="rename_model":
                tpe.submit(model.rename, **msg)
            elif msg["action"]=="create_model":
                tpe.submit(model.create_model, **msg)
            elif msg["action"]=="save_model":
                tpe.submit(model.save_for_download, **msg)
            elif msg["action"]=="load_dataset":
                tpe.submit(model.load_dataset, **msg)
            elif msg["action"]=="save_to_csv":
                tpe.submit(model.save_to_csv, **msg)
            elif msg["action"]=="save_to_sqlite":
                tpe.submit(model.save_to_sqlite, **msg)
            elif msg["action"]=="delete_row":
                tpe.submit(model.delete_rows, **msg)
            elif msg["action"]=="clean_dataset":
                tpe.submit(model.clean_dataset, **msg)
            elif msg["action"]=="copy_rows":
                tpe.submit(model.copy_rows, **msg)
            elif msg["action"]=="reset_column":
                tpe.submit(model.reset_columns, **msg)
            elif msg["action"]=="train":
                tpe.submit(model.train, **msg)
            elif msg["action"]=="test":
                tpe.submit(model.test, **msg)
            elif msg["action"]=="predict":
                tpe.submit(model.adhoc_predict, **msg)
            elif msg["action"]=="read_train_result":
                tpe.submit(model.read_train_result, **msg)
            elif msg["action"]=="read_predict_result":
                tpe.submit(model.read_adhoc_predict_result, **msg)
            elif msg["action"]=='get_group_info':
                from reporting import get_group_info
                tpe.submit(get_group_info, model, **msg)
            elif msg["action"]=='text_similarity':
                from reporting import text_similarity
                tpe.submit(text_similarity, model, **msg)
            elif msg["action"]=='text_summary':
                from reporting import text_summary
                tpe.submit(text_summary, model, **msg)
            elif msg["action"]=='group_search':
                from reporting import group_search
                tpe.submit(group_search, model, **msg)
            elif msg["action"]=='group_over_time':
                from reporting import group_over_time
                tpe.submit(group_over_time, model, **msg)
            elif msg["action"]=="list_checkpoint":
                tpe.submit(list_checkpoint, model)
            elif msg["action"]=="load_checkpoint":
                tpe.submit(load_checkpoint, self, msg)
            elif msg["action"]=="save_checkpoint":
                tpe.submit(save_checkpoint, model)
            elif msg["action"]=="delete_checkpoint":
                tpe.submit(delete_checkpoint, model, msg)
            elif msg["action"]=='abort':
                abort_action(model)
            else:
                self.send_result(-996, "Unknown command")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/ws", WebSocketHandler),
        (r"/ds/(.*)", DSHandler),
        #(r"/emb/([0-9]+)", VisualizeEmbeddingHandler),
        (r"/saves/(.*)", StaticFileHandler, {"path":"./saves"}),
        (r"/static/(.*)", StaticFileHandler, {"path":"./static"}),
    ], debug=True)

def shutdown():
    global ioloop
    ioloop.stop()
    
def sig_handler(sig, frame=None):
    global ioloop
    print('Caught signal: %s' % sig)
    ioloop.add_callback_from_signal(shutdown)

def open_browser(port:int):
    import webbrowser
    webbrowser.open('http://localhost:%d' % port, new=2)

if __name__ == "__main__":
    from os import makedirs
    makedirs("./saves/checkpoints", exist_ok=True)
    print("Running...")
    app = make_app()
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGINT, sig_handler)
    if sys.platform == "win32":
        import win32api
        win32api.SetConsoleCtrlHandler(sig_handler, True)
    try:
        app.listen(6789)
        tpe.submit(open_browser, 6789)
    except Exception:
        app.listen(6790)
        tpe.submit(open_browser, 6790)
    ioloop = tornado.ioloop.IOLoop.instance()
    ioloop.start()
        