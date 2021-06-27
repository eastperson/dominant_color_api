# -*- coding: utf-8 -*-
from json import JSONEncoder

from flask import Flask, request
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import json
from skimage import io

app = Flask('Dominant Color App')

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

@app.route('/post', methods=['POST'])
def post():
    try:
        if request.is_json :
            print('json : ',request.json)
            memberId = request.get_json()['member_id']
    except Exception as e :
        return {'status':'error', 'message': str(e)}
    return {'status': 'success','member_id':memberId}

@app.route('/fileupload', methods=['GET','POST'])
def file_upload():
    try :
        if request.method == 'POST' :
            f = request.files['file']
            filename = secure_filename(f.filename)
            print('secure_filename(f.filename) :',filename)
            f.save("./img/" + filename)

            img = io.imread('./img/' + filename)[:, :, :-1]
            pixels = np.float32(img.reshape(-1, 3))
            n_colors = 8
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            dominant = palette[np.argmax(counts)]
            indices = np.argsort(counts)[::-1]
            freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
            rows = np.int_(img.shape[0] * freqs)
            dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
            result = []
            for i in range(len(rows) - 1):
                dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])
                li = []

                arr = np.uint8(palette[indices[i]])
                tu = (int(arr[0]),int(arr[1]),int(arr[2]))

                li.append(freqs[i+1]-freqs[i])
                li.append(tu)

                result.append(li)

            print('result :',result)
            return {'result' : json.dumps(result), 'size' : len(result)}
    except Exception as e :
        return {'status':'error','message':str(e)}

@app.route('/test', methods=['GET','POST'])
def check():
    if request.method == 'GET' :
        return 'get success'
    elif request.method == 'POST' :
        return 'post success'



if __name__ == '__main__':
        app.run(host='0.0.0.0', port=8000, debug=True)