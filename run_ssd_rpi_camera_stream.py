from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import requests
import argparse
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

parser = argparse.ArgumentParser(description='SSD Inference With Distance on Camera Stream')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('---get_url', type=str, default=None)
parser.add_argument('--display_mode', type=int, default='0')

args = parser.parse_args()

class_names = ['BACKGROUND','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

accepted_classes = {'person'}
accepted_classes_idx = {class_names.index(acc_class) for acc_class in accepted_classes}

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(args.model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=args.device)

cap = cv2.ViideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
counter = 0
camera = PiCamera()
camera.resolution = (1080, 720)
camera.framerate = 10
time.sleep(0.1)

start = time.time()
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    orig_image = frame.array

    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    counter += 1
    if counter % (fps) != 0:
        continue
    #print(time.time()-start)
    counter = 0
    predictor_ret_vals = predictor.predict(image, 10, 0.4)
    boxes, labels, distances, probs = predictor_ret_vals
    labels_mask = np.array([True if int(label) in accepted_classes_idx
        else False for label in labels])
    len_acc_classes = 0 if len(boxes) == 0  else len(labels_mask[labels_mask==True])

    if len_acc_classes:
        labels = labels.numpy()
        boxes = boxes.numpy()
        distances = distances.numpy()
        probs = probs.numpy()
        labels = labels[labels_mask]
        boxes = boxes[labels_mask]
        distances = distances[labels_mask]
        probs = probs[labels_mask]

        image_width = image.shape[1]
        boxes_center = (boxes[:,0]+boxes[:,2])/2
        bc_width_frac = boxes_center/image_width
        frame_object_info = list(zip(bc_width_frac, distances))
        print(frame_object_info)
    else:
        frame_object_info = []
    if args.get_url:
        params_to_be_sent = {'x_d_list' : frame_object_info}
        request.get(url = args.get_url, params = params_to_be_sent)

    if args.display_mode:
        for i in range(len(boxes)):
            box = boxes[i, :]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}@{distances[i]:.2f}cm"
            cv2.putText(orig_image, label,
                        (int(box[0] + 20), int(box[1] + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (64, 0, 255),
                        2)  # line type
        cv2.imshow('Output', orig_image)

    rawCapture.truncate(0)
    #Wait to press 'q' key for capturing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

