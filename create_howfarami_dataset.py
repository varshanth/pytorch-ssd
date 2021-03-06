from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import json
import os
from PIL import Image
import numpy as np

if len(sys.argv) < 6:
    print('Usage: python run_ssd_example.py <net type>  <model path> <image path>')
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
images_folder = sys.argv[4]
device = sys.argv[5]
if device == "" or device == None:
    device = "cpu"


class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200, device=device)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, device=device)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200, device=device)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=device)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200, device=device)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200, device=device)

data = []
accepted_labels = [9, 15]
for images in os.listdir(images_folder):
	#pil_image = Image.open(images_folder + "/" + images).convert('RGB')
	#orig_image = np.array(pil_image)
	orig_image = cv2.imread(images_folder + "/" + images)
	trans_image = orig_image
	#if height
	height, width, channels = trans_image.shape
	# rotating the image if height is greater than width
	if height < width:
		trans_image = cv2.rotate(orig_image, cv2.ROTATE_90_CLOCKWISE)
	print(images, height, width)
	image = cv2.cvtColor(trans_image, cv2.COLOR_BGR2RGB)
	boxes, labels, probs = predictor.predict(image, 10, 0.4)

	img_id = os.path.splitext(images)[0]
	data_per_image = {'boxes' : [], 'labels' : [], 'distances' : [], 'image_id' : img_id}
	for i in range(boxes.size(0)):
		box = boxes[i, :]
		cv2.rectangle(trans_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
		#label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
		label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
		cv2.putText(trans_image, label,
			(box[0] + 20, box[1] + 40),
			cv2.FONT_HERSHEY_SIMPLEX,
			1,  # font scale
			(255, 0, 255),
			2)  # line type

		if labels[i] in accepted_labels:
			data_per_image['boxes'].append(box.cpu().numpy().tolist())
			data_per_image['labels'].append(class_names[labels[i]])
			data_per_image['distances'].append(10)
	data.append(data_per_image)

with open('data.txt', 'w') as outfile:
        json.dump(data, outfile, indent=4)
path = "run_ssd_example_output.jpg"
cv2.imwrite(path, trans_image)
print(f"Found {len(probs)} objects. The output image is {path}")
