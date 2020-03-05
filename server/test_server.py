from __future__ import print_function
import requests
import json
import cv2

test_url = 'http://localhost:5000/api/test'

img = cv2.imread('newfam.jpg')

print(img)

_, img_encoded = cv2.imencode('.jpg', img)
response = requests.post(test_url, files={'image': img_encoded.tostring()})

res = json.loads(response.text)
if 'message' in res: print('Worked!', res['message'])
else: raise Exception('Something went wrong!')