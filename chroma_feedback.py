import threading
import time
import requests

class ChromaFeedback:
    def __init__(self):
        self.uri = None
        self.sessionid = None
        self.active = False
        self.shutdown_flag = False
        self.current_state = "normal"  # "normal", "distracted", "slouching"
        self._loop_thread = None
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            self.shutdown_flag = False
            self.current_state = "normal"
        
    def _initialize_sdk(self):
        if self.active: return
        url = "http://localhost:54235/razer/chromasdk"
        payload = {
            "title": "Time Lapse Study",
            "description": "Posture and Focus Feedback",
            "author": {
                "name": "Local",
                "contact": "none"
            },
            "device_supported": ["keyboard"],
            "category": "application"
        }
        try:
            resp = requests.post(url, json=payload, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                self.uri = data.get("uri")
                self.sessionid = data.get("sessionid")
                self.active = True
                threading.Thread(target=self._heartbeat, daemon=True).start()
        except:
            pass

    def _heartbeat(self):
        while self.active and self.uri and not self.shutdown_flag:
            try:
                requests.put(f"{self.uri}/heartbeat", timeout=2)
            except:
                pass
            time.sleep(1)

    def close(self):
        self.shutdown_flag = True
        self.set_state("normal")
        self._close_sdk()
        
    def _close_sdk(self):
        if self.uri:
            try:
                requests.delete(self.uri, timeout=2)
            except:
                pass
        self.active = False
        self.uri = None

    def _apply_effect(self, effect, param):
        if not self.active or not self.uri:
            return
        try:
            requests.put(f"{self.uri}/keyboard", json={"effect": effect, "param": param}, timeout=1)
        except:
            pass

    def set_state(self, state):
        with self.lock:
            if self.current_state == state:
                return
            self.current_state = state
            
            if state == "normal":
                self._close_sdk()
            else:
                self._initialize_sdk()
                if not self._loop_thread or not self._loop_thread.is_alive():
                    self._loop_thread = threading.Thread(target=self._pulse_loop, daemon=True)
                    self._loop_thread.start()
                    
    def _pulse_loop(self):
        while True:
            with self.lock:
                state = self.current_state
            if state == "normal":
                break
                
            if state == "distracted":
                self._apply_effect("CHROMA_STATIC", {"color": 255}) # Red
            elif state == "slouching":
                self._apply_effect("CHROMA_STATIC", {"color": 65535}) # Yellow
                
            time.sleep(0.3)
            with self.lock:
                if self.current_state == "normal": break
            self._apply_effect("CHROMA_NONE", None)
            time.sleep(0.3)
