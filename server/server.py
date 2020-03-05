from flask import Flask, request, Response, send_file
import jsonpickle
import numpy as np
import os
import cv2
from time import time
import base64

from model import FaceModel

app = Flask(__name__)


### PREDICTIONS ###

INPUT = (224,224) # (h,w)

binary_predictions = [
('BackgroundUniformity', 0.5),
('BlinkConfidence', 0.6),
('Contrast', 0.8),
('DarkGlassesConfidence', 0.5),
('DetectionConfidence', 0.65),
('ExpressionConfidence', 0.6),
('FaceDarknessConfidence', 0.9),
('GlassesConfidence', 0.4),
('GlassesReflectionConfidence', 0.99),
('GrayscaleDensity', 0.8),
('LookingAwayConfidence', 0.65),
('MouthOpenConfidence', 0.6),
('Noise', 0.8),
# ('PixelationConfidence', 0.99),
('Quality', 0.9),
# ('RedEyeConfidence', 0.99),
# ('Saturation', 1),
('Sharpness', 0.75),
('SkinReflectionConfidence', 0.4),
('UnnaturalSkinToneConfidence', 0.55),
# ('WashedOutConfidence', 0.95)
]

float_predictions = [
('Yaw', 0.05, 8.47),
('Roll', -1.01, 6.59),
('Pitch', -8.44, 6.09),
]

float_names = list(map(lambda x:x[0], float_predictions))
bin_names = list(map(lambda x:x[0], binary_predictions))

float_stats = torch.tensor([[mean, std] for name, mean, std in float_predictions])
bin_stats = torch.tensor([[mean, 1] for name, mean in binary_predictions])

### MODELS ###

def load_torch_inference(model, path, device=torch.device('cpu')):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()

MODEL = FaceModel(num_bins=len(binary_predictions), num_floats=len(float_predictions), float_stats=)
load_torch_inference(MODEL, './model.pth', device=torch.device('cpu'))

### MAIN ###


class PT(object): # Print Times
    def __init__(self, name):
        print(f'{name}... ', end='')
    def __enter__(self): 
        self.t0 = time()
    def __exit__(self, exc_type, exc_value, tb): 
        print(f'took: {time()-self.t0}')


def prepare_output(o): # [1,n,h,w], n - can be different for each output
    ''' [1,n,h,w] -> [h*w,n] '''
    o = o[0].permute(1,2,0)
    return o.view(-1,o.shape[-1])

def reconstruct_fpts(fpts, h, w): # [68,2]
    fpts = (fpts+1)/2 * torch.tensor([h,w])[None]
    return [[int(x),int(y)] for y,x in fpts]

def image2preds(img, config):
    out = []
    h_orig, w_orig, c = img.shape
    img = cv2.resize(img, dsize=(INPUT[1], INPUT[0]))
    im = tensor(img).permute(2,0,1).float() / 255.
    with PT('model inference:') as _:
        fpts, bins, floats = MODEL(im[None]) # [1,68,2], [1,B], [1,F]
    fpts, bins, floats = map(lambda x:x[0], [fpts, bins, floats])
    bins, floats = map(float, bins), map(float, floats)
    bins, floats = dict(zip(bin_names,bins)), dict(zip(float_names,floats))
    fpts = reconstruct_fpts(fpts, h_orig, w_orig)
    return fpts, bins, floats

### ROUTES ###

def decode_image(img_bytes):
    img_raw = np.frombuffer(img_bytes, np.uint8) # <class 'bytes'> -> np.arr
    img = cv2.imdecode(img_raw, cv2.IMREAD_COLOR)
    img = img[ : , : , [2,1,0] ] # BGR -> RGB
    return img

def send_image(img, config):
    r = request.args.get('resize')
    with tempfile.NamedTemporaryFile() as f:
        img = img[ : , : , [2,1,0] ] # RGB -> BGR
        if r is not None: img = cv2.resize(img, dsize=None, fx=float(r), fy=float(r))
        cv2.imwrite(f'{f.name}.jpg', img)
        return send_file(f'{f.name}.jpg', mimetype='image/gif') # as_attachment=True

''' @input:
@input: Body = {'image': File})
@output: {'message': 'image received...'}
'''
@app.route('/api/test', methods=['POST'])
def test():
    img = decode_image(img_bytes=request.files['image'].read())
    response = {'message': 'image received. img.shape: {}x{}'.format(img.shape[1], img.shape[0])}
    return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")


''' @input:
@input: (
    Body = {'image': File},
    Params = {
    }
) (if Params are not specified, default config is used [look at parse_config func])
@output: jpg image
'''
@app.route('/api/icao_v1_preview', methods=['POST'])
def icao_v1_preview():
    import cv2

    config = dict(request.args)
    # try:
    img = decode_image(img_bytes=request.files['image'].read())
    fpts, bins, floats = image2preds(img, config=config)
    img = np.ascontiguousarray(img) # WTF: https://github.com/opencv/opencv/issues/14866
    for x,y in fpts:
        img = cv2.circle(img, (x,y), radius=2, color=(255,0,0), thickness=1)
    texts = [n[:2]+str(int(v*100)) for n,v in bins.items()]
    texts = [texts[i]+';'+texts[i+1] for i in range(0,len(texts),2)]
    y, l = 0, 0
    for text in texts:
        img = cv2.putText(img, text, org=(l+10,y), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                            fontScale=1.0, color=(255,255,0), thickness=1, lineType=cv2.LINE_AA) 
        y += 20
    return send_image(img, config=config)
    # except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


''' @input:
@input: (
    Body = {'image': File},
    Params = {
    }
) (if Params are not specified, default config is used [look at parse_config func])
@output: {
    'feat_points': [[x1,y1], ...],
    'bins': {'BlinkConfidence':0.3,...},
    'floats': {'Pitch':1.1,'Roll':5.1,'Yaw':-0.1}
}
'''
@app.route('/api/icao_v1', methods=['POST'])
def icao_v1():
    import cv2

    config = dict(request.args)
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        fpts, bins, floats = image2preds(img, config=config)
        out = {'feat_points':fpts, 'bins':bins, 'floats':floats}
        return Response(response=jsonpickle.encode(out), status=200, mimetype="application/json")
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


    return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")




app.run(host="0.0.0.0", port=5000)