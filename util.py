from contextlib import redirect_stdout, redirect_stderr
import re

class Logger:
    def __init__(self):
        self.ws = None
        self.ioloop = None
    def log(self, status=-1000, msg=None, param=None):
        if self.ws and self.ioloop:
            self.ioloop.add_callback(self.ws.send_result, status, msg, param)
    def working(self, msg=None):
        self.log(1, msg)
    def progress(self, msg=None, progress=0):
        if isinstance(progress, float) or isinstance(progress, int):
            self.log(2, msg, {"progress": progress})
        elif isinstance(progress, dict):
            self.log(2, msg, progress)
        else:
            self.log(2, msg)
    def finished(self, msg=None, param=None):
        self.log(0, msg, param)
    def busy(self):
        self.log(-997, "Model is busy, try again later")
    def error(self, msg=None):
        self.log(-998, msg)
    def invalid(self, msg='Invalid request'):
        self.log(-999, msg)
    def stream(self, stream, msg):
        self.log(stream, msg)

class outwriter(object):
    def __init__(self, log:Logger):
        self.log = log
        pass
    def write(self, data):
        self.log.stream(2, repr(data))
    def flush(self):
        pass

class errwriter(object):
    def __init__(self, log:Logger):
        self.log = log
    def tqnum(s:str)->float:
        if not s: return 1
        num = 1
        try:
            if s[-1] in ["K", "M", "G", "T"]:
                num = float(s[:-1])
            else:
                num = float(s)
            if s[-1]=="K":
                num *= 1024
            elif s[-1]=="M":
                num *= 1024 * 1024
            elif s[-1]=="G":
                num *= 1024 * 1024 * 1024
            elif s[-1]=="T":
                num *= 1024*1024*1024*1024
        except Exception:
            return 1
        pass
    def write(self, data):
        #
        # try to capture tqdm output
        #
        tq = re.search(r'(([^:]+): ?)?([0-9]+)%\|[^\|]+\| ?([0-9GKMT]+)\/([0-9GKMT]+) ?\[([0-9:]+)<([0-9:]+), ?([^\]]+)', data.replace('\r', ''))
        if tq:
            desc = tq.group(2)
            if desc is None: desc = "In progress"
            progress = float(tq.group(4)) / float(tq.group(5))
            ellipsed = tq.group(6)
            remain = tq.group(7)
            speed = tq.group(8)
            msg = "\\r%s: %.1f%%" % (desc, progress * 100)
            if '\n' in data: msg += "\\n"
            self.log.progress(msg, {
                'progress': progress,
                'ellipsed': ellipsed,
                'remain': remain,
                'speed': speed
            })
        else:
            tq = re.search(r'(([^:]+): ?)?([0-9]+)%\|[^\|]+\| ?([0-9GKMT]+)\/([0-9GKMT]+) ?\[([0-9:]+)', data.replace('\r', ''))
            if tq:
                desc = tq.group(2)
                if desc is None: desc = "In progress"
                progress = float(tq.group(4)) / float(tq.group(5))
                ellipsed = tq.group(6)
                msg = "\\r%s: %.1f%%" % (desc, progress * 100)
                self.log.progress(msg, {
                    'progress': progress,
                    'ellipsed': ellipsed,
                })
            else:
                self.log.stream(3, repr(data))
    def flush(self):
        pass

class LogRedirect:
    #out_redirected = False
    def __init__(self, log:Logger):
        # self.log = log
        self.stdout = redirect_stdout(outwriter(log))
        self.stderr = redirect_stderr(errwriter(log))
    def __enter__(self):
        LogRedirect.out_redirected += 1
        # self.log.working("out_redirected")
        self.stderr.__enter__()
        self.stdout.__enter__()
    def __exit__(self, exctype, excinst, exctb):
        self.stderr.__exit__(exctype, excinst, exctb)
        self.stdout.__exit__(exctype, excinst, exctb)
        LogRedirect.out_redirected -= 1
        if LogRedirect.out_redirected < 0: LogRedirect.out_redirected = 0
        # self.log.working("out_not_redirected %r" % LogRedirect.out_redirected)
        
LogRedirect.out_redirected = 0