from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import requests
import argparse
import numpy as np
import time
import imutils

parser = argparse.ArgumentParser(description='SSD Inference With Distance on Camera Stream')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--display_mode', type=int, default='0')
parser.add_argument('--rot_camera90', type=bool, default=False)

args = parser.parse_args()

class_names = ['BACKGROUND','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

accepted_classes = {'person'}
accepted_classes_idx = {class_names.index(acc_class) for acc_class in accepted_classes}

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(args.model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=args.device)

# Setup Camera
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
inference_sampling_rate = fps // 2
counter = 0
start = time.time()

while(True):
    ret, orig_image = cap.read() #For capture the image in RGB color space
    if args.rot_camera90:
        orig_image = imutils.rotate(orig_image, 90)
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    counter += 1
    if counter % inference_sampling_rate != 0:
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
        labels_words  = [class_names[l_idx] for l_idx in labels]
        frame_object_info = list(zip(labels_words, bc_width_frac, distances))
    else:
        frame_object_info = []

    if args.display_mode and len_acc_classes:
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
        cv2.imshow(f'Output', orig_image)
    #Wait to press 'q' key for capturing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
