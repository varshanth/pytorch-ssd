from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import cv2
import requests
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='SSD Inference With Distance on Camera Stream')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

class_names = ['BACKGROUND','aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person','pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor']

accepted_classes = {'chair', 'person'}
accepted_classes_idx = {class_names.index(acc_class) for acc_class in accepted_classes}

net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load(args.model_path)
predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=args.device)


for img in os.listdir(args.dir_path):
    orig_image = cv2.imread(f"{args.input_dir}/{img}")
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
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

    if len_acc_classes:
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
        cv2.imwrite(f'{args.output_dir}/{img}', orig_image)
