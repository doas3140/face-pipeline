from flask import Flask, request, Response, send_file
import jsonpickle
import numpy as np
import os
from tqdm import tqdm
import cv2
from time import time
import base64

from fastai import *
from fastai.vision import *

from model_old import EastModel

app = Flask(__name__)


### PREDICTIONS ###

INPUT = (512,384) # (h,w)

binary_predictions = [
# 'BackgroundUniformity',
'BlinkConfidence',
# 'Contrast',
'DarkGlassesConfidence',
# 'DetectionConfidence',
'ExpressionConfidence',
'FaceDarknessConfidence',
'GlassesConfidence',
'GlassesReflectionConfidence',
# 'GrayscaleDensity',
'LookingAwayConfidence',
'MouthOpenConfidence',
# 'Noise',
# 'PixelationConfidence',
# 'Quality',
# 'RedEyeConfidence',
# 'Saturation',
# 'Sharpness',
'SkinReflectionConfidence',
'UnnaturalSkinToneConfidence',
# 'WashedOutConfidence',
]

float_predictions = [
'Pitch',
'Roll',
'Yaw'
]

multi_predictions = []

point_predictions = []

### MODELS ###

def load_torch_inference(model, path, device=torch.device('cpu')):
    model.load_state_dict(torch.load(path, map_location=device))
    return model.eval()

MODEL = EastModel(num_bins=len(binary_predictions), num_floats=len(float_predictions))
load_torch_inference(MODEL, './model.pth', device=torch.device('cpu'))

### ANCHORS ###

def create_grid(size):
    ''' Creates a x,y grid of size `size`, coords start from -1 to 1
    @param: tuple(H,W) :tuple of 2 ints H(height), W(width)
    @return [H,W,4]    :tlbr coords for each (h,w) cell 
    '''
    H, W = size if is_tuple(size) else (size,size)
    grid = torch.FloatTensor(H, W, 4)
    
    # not precise, but good enough
    linear_points_left = torch.linspace(0, 1-1/W, W) # [0, 1]
    linear_points_right = linear_points_left + 1/W
    linear_points_top = torch.linspace(0, 1-1/H, H) # [0, 1]
    linear_points_btm = linear_points_top + 1/H
    
    grid[:, :, 0] = linear_points_top.unsqueeze(1).expand(H,W)
    grid[:, :, 1] = linear_points_left.unsqueeze(0).expand(H,W)
    grid[:, :, 2] = linear_points_btm.unsqueeze(1).expand(H,W)
    grid[:, :, 3] = linear_points_right.unsqueeze(0).expand(H,W)
    return grid*2-1

ANCHORS = create_grid((INPUT[0]//4, INPUT[1]//4)).view(-1,4)

tlbr2cc = lambda boxes: (boxes[:,:2] + boxes[:,2:])/2 # [N,4] -> [N,2]
def target_to_bbox(output, anchors):
    ''' converts nn outputs -> actual coordinates '''
    a_centers = tlbr2cc(anchors)
    return torch.cat([a_centers - output[...,:2], a_centers + output[...,2:]], -1)

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

def image2preds(img, config):
    out = []
    h_orig, w_orig, c = img.shape
    img = cv2.resize(img, dsize=(INPUT[1], INPUT[0]))
    im = tensor(img).permute(2,0,1).float() / 255.
    score, geo, other_lbls, mask = MODEL(im[None])
    bins, floats = other_lbls
    score, geo, bins, floats = map(prepare_output, [score, geo, bins, floats]) # [h*w,n], where n - can be different
    geo = target_to_bbox(geo, ANCHORS.to(geo.device))
    mask = np.ascontiguousarray(mask.squeeze().detach().numpy()) # [h,w]
    mask = (mask*255).astype(np.uint8)
    mask = cv2.resize(mask[:,:,None], dsize=(w_orig,h_orig)).squeeze()
    # get a single face
    face_idx = score.squeeze().argmax(dim=-1)
    score = float(score.squeeze()[face_idx])
    (t,l,b,r), bins, floats = map(lambda x: x[face_idx], [geo, bins, floats])
    t,b = tuple(map(lambda x: int((x+1)/2*h_orig), [t,b]))
    l,r = tuple(map(lambda x: int((x+1)/2*w_orig), [l,r]))
    geo = {k:int(v) for k,v in zip(['t','l','b','r'], (t,l,b,r))}
    bins = {k:float(v) for k,v in zip(binary_predictions, bins)}
    floats = {k:float(v) for k,v in zip(float_predictions, floats)}

    return score, geo, bins, floats, mask

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

def encode_b64_image(img, config):
    r = request.args.get('resize')
    with tempfile.NamedTemporaryFile() as f:
        if r is not None: img = cv2.resize(img, dsize=None, fx=float(r), fy=float(r))
        cv2.imwrite(f'{f.name}.png', img)
        base64_str = base64.b64encode(open(f'{f.name}.png').read())
        return base64_str.decode('utf-8') # remove b' at beginning


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
@app.route('/api/face_v1_preview', methods=['POST'])
def face_v1_preview():
    import cv2

    config = dict(request.args)
    # try:
    img = decode_image(img_bytes=request.files['image'].read())
    score, geo, bins, floats, mask = image2preds(img, config=config)
    img = np.ascontiguousarray(img) # WTF: https://github.com/opencv/opencv/issues/14866
    t,l,b,r = map(lambda n: geo[n], ['t','l','b','r'])
    img = cv2.rectangle(img, (l,t), (r,b), color=(255,255,0), thickness=2)
    texts = [n[:2]+str(int(v*100)) for n,v in bins.items()]
    texts = [texts[i]+';'+texts[i+1] for i in range(0,len(texts),2)]
    y = t+20
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
@output: jpg image
'''
@app.route('/api/face_v1_mask_preview', methods=['POST'])
def face_v1_mask_preview():
    import cv2

    config = dict(request.args)
    # try:
    img = decode_image(img_bytes=request.files['image'].read())
    score, geo, bins, floats, mask = image2preds(img, config=config)
    mask = mask[:,:,None].repeat(3, axis=2)
    return send_image(mask, config=config)
    # except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


''' @input:
@input: (
    Body = {'image': File},
    Params = {
    }
) (if Params are not specified, default config is used [look at parse_config func])
@output: {
    'score': 0.6,
    'geo': {'t':100,'l':100,'b':200,'r':200},
    'bins': {'BlinkConfidence':0.3,...},
    'floats': {'Pitch':1.1,'Roll':5.1,'Yaw':-0.1},
    'mask': 'base64 encoded string of png image'
}
'''
@app.route('/api/face_v1', methods=['POST'])
def face_v1():
    import cv2

    config = dict(request.args)
    try:
        img = decode_image(img_bytes=request.files['image'].read())
        score, geo, bins, floats, mask = image2preds(img, config=config)
        mask_b64_str = encode_b64_image(mask, config=config)
        out = {'score':score, 'geo':geo, 'bins':bins, 'floats':floats, 'mask':mask_b64_str}
        return Response(response=jsonpickle.encode(out), status=200, mimetype="application/json")
    except Exception as e: return Response(response=jsonpickle.encode({'error': str(e)}), status=500, mimetype="application/json")


    return Response(response=jsonpickle.encode(response), status=200, mimetype="application/json")




app.run(host="0.0.0.0", port=5000)