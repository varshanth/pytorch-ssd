import json
import os


train_frac = 0.9
ds_root = './'
train_dir = f"{ds_root}/train"
test_dir = f"{ds_root}/test"

with open('labels.json', 'r') as f:
    content = json.load(f)

num_train = int(train_frac * len(content))

train_content = content[:num_train]
test_content = content[num_train:]

with open('train_labels.json', 'w') as f:
    json.dump(train_content, f)

with open('test_labels.json', 'w') as f:
    json.dump(test_content, f)


for train_img_info in content[:num_train]:
    img_name = f"{train_img_info['image_id']}.jpg"
    os.rename(f"{ds_root}/images/{img_name}", f"{train_dir}/{img_name}")

for test_img_info in content[num_train:]:
    img_name = f"{test_img_info['image_id']}.jpg"
    os.rename(f"{ds_root}/images/{img_name}", f"{test_dir}/{img_name}")


