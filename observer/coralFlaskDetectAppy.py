import argparse
import cv2
import os
import json
from random import choice
from datetime import datetime, timedelta

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

from periphery import GPIO

from flask import Flask, render_template, Response


app = Flask(__name__)
MODEL = None
LABELS = None
CAMERA_IDX = 1
THRESHOLD = 1


def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


def gen_frames():
    assert MODEL is not None, "MODEL must be defined"
    assert LABELS is not None, "LABELS must be defined"
    interpreter = make_interpreter(MODEL)
    interpreter.allocate_tensors()
    labels = read_label_file(LABELS)
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(CAMERA_IDX)

    try:
        while cap.isOpened():
            ret, cv2_im = cap.read()
            if not ret:
                break

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())
            foci = get_objects(interpreter, THRESHOLD)
            cv2_im = append_objs_to_img(cv2_im, inference_size, foci, labels)

            ret, cv2_im = cv2.imencode('.jpg', cv2_im)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2_im.tobytes() + b'\r\n')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    default_model_dir = './model'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 1)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    args = parser.parse_args()
    MODEL = args.model
    LABELS = args.labels
    CAMERA = args.camera_idx
    THRESHOLD = args.threshold
    print('Loading {} with {} labels.'.format(args.model, args.labels))

    app.run(debug=True, host="0.0.0.0")

