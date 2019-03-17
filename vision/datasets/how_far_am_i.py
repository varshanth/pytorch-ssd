import numpy as np
import pathlib
import cv2
import json


class HowFarAmIDataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train"):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()
        self.data, self.class_names, self.class_dict = self._read_data()
        self.ids = [info['image_id'] for info in self.data]


    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        boxes = np.array(image_info['boxes'], dtype=np.float32)
        labels = np.array(image_info['labels'], dtype=np.int64)
        distances = np.array(image_info['distances'], dtype=float32)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels, distances = self.target_transform(boxes, labels, distances)
        return image_info['image_id'], image, boxes, labels, distances

    def __getitem__(self, index):
        _, image, boxes, labels, distances = self._getitem(index)
        return image, boxes, labels, distances

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        annotation_file = f"{self.root}/{self.dataset_type}_labels.json"
        annotations = json.load(annotation_file)
        class_names = ['BACKGROUND', 'person']
        class_dict = {'BACKGROUND': 0, 'person': 15}
        with open(annotation_file, 'r') as a_file:
            data = json.load(a_file)
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
