# file: test_wrapped_onnx.py
import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_PATH = "outputs_20250906_142440/model_final.onnx"
TEST_IMAGE = "image_processing20220902-2740614-ioomxn.jpg"  # use raw string on Windows
TOP_K = 3
LETTERBOX = True

CLASS_INDEX_JSON = None  # e.g., "class_indices.json" from training
CLASS_LABELS_FALLBACK = [
    "Anthracnose","Bacterial Blight","Citrus Canker","Curl Virus",
    "Deficiency Leaf","Dry Leaf","Healthy Leaf","Sooty Mould","Spider Mites"
]

def _load_labels():
    if CLASS_INDEX_JSON and os.path.exists(CLASS_INDEX_JSON):
        with open(CLASS_INDEX_JSON, "r", encoding="utf-8") as f:
            m = json.load(f)  # {"Label": index}
        labels = [None]*len(m)
        for k,v in m.items(): labels[v]=k
        if any(x is None for x in labels):
            raise ValueError("class_indices.json produced gaps.")
        return labels
    return CLASS_LABELS_FALLBACK

def _norm_dims(shape):
    out=[]
    for d in shape:
        out.append(int(d) if isinstance(d,(int,np.integer)) else None)
    return out

def _get_io(session):
    inp = session.get_inputs()[0]
    outs = session.get_outputs()
    dims = _norm_dims(inp.shape)   # [N,H,W,C] or [N,C,H,W]
    if len(dims)!=4: raise ValueError(f"Expected 4D input; got {dims}")
    _, a,b,c = dims
    if c==3: layout,H,W = "NHWC", a,b
    elif a==3: layout,H,W = "NCHW", b,c
    else: layout,H,W = "NHWC", a,b
    return inp.name, outs[0].name, layout, (H or 256), (W or 256)

def _letterbox(img, W, H):
    iw, ih = img.size
    s = min(W/iw, H/ih)
    nw, nh = int(round(iw*s)), int(round(ih*s))
    canvas = Image.new("RGB", (W, H), (0,0,0))
    canvas.paste(img.resize((nw,nh), Image.BILINEAR), ((W-nw)//2, (H-nh)//2))
    return canvas

def _preprocess(img_path, H, W, layout):
    img = Image.open(img_path).convert("RGB")
    img = _letterbox(img, W, H) if LETTERBOX else img.resize((W,H), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32)  # H,W,C   (NO /255 HERE!)
    if layout=="NCHW":
        arr = np.transpose(arr, (2,0,1))
    return np.expand_dims(arr, 0)

class OnnxLeafClassifier:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name, self.output_name, self.layout, self.H, self.W = _get_io(self.session)
        self.labels = _load_labels()

    def predict_image(self, img_path, top_k=TOP_K):
        x = _preprocess(img_path, self.H, self.W, self.layout)
        y = self.session.run([self.output_name], {self.input_name: x})[0]
        vec = y[0] if (y.ndim==2 and y.shape[0]==1) else y.reshape(-1)
        # Since we exported Softmax, vec should already be probabilities
        probs = vec

        k = min(top_k, len(probs))
        top_idx = probs.argsort()[-k:][::-1]
        top_pairs = [(self.labels[i] if i < len(self.labels) else f"class_{i}", float(probs[i])) for i in top_idx]

        print(f"Image: {os.path.basename(img_path)}")
        print("Top predictions:")
        for name, p in top_pairs:
            print(f"  {name}: {p*100:.2f}%")
        final_label = top_pairs[0][0]
        print(f"Final Predicted Class: {final_label}")
        print("-----")
        return final_label

if __name__ == "__main__":
    clf = OnnxLeafClassifier(MODEL_PATH)
    clf.predict_image(TEST_IMAGE, top_k=3)
