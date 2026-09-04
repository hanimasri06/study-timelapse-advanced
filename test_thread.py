import threading, time, onnxruntime as ort
session=ort.InferenceSession('yolov8n.onnx', providers=['DmlExecutionProvider'])
def f():
    import onnxruntime_genai as og
    print('loading model')
    og.Model('models/phi3/directml/directml-int4-awq-block-128')
    print('OK')
threading.Thread(target=f).start()
time.sleep(5)
