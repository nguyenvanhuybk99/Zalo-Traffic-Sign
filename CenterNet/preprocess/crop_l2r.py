import json
import cv2
import os
import copy


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
INPUT_IMG_FOLDER = '/data/traffic_train/images'             ## <== 
mkdir(INPUT_IMG_FOLDER)
OUTPUT_IMG_FOLDER = '/data/zalo_challenge/images'
mkdir(OUTPUT_IMG_FOLDER)
JSON_IN = '/data/traffic_train/train_traffic_sign_dataset.json'
JSON_OUT = '/data/zalo_challenge/all_annotations.json'


data = json.load(open(JSON_IN))

info = data['info']
categories = data['categories']

images = data['images']
annotations = data['annotations']


ret = { 'images': [], 'annotations': [], 'categories': categories, 'info': info }

imgid2listannot = {}

for annot in annotations:
    img_id = annot['image_id']
    if img_id not in imgid2listannot:
        imgid2listannot[img_id] = []
    imgid2listannot[img_id].append(annot)


img_id_count = 0
annot_id_count = 0

keep_img_id = set()
cut_idx = [0, 202, 405, 607, 811]
for img_info in images:
    img_id = img_info['id']
    # if img_id != 3679:
    #     continue
    filename = img_info['file_name']
    filepath = os.path.join(INPUT_IMG_FOLDER, filename)
    img_ori = cv2.imread(filepath)

    ori_h, ori_w, _ = img_ori.shape
    # ori_h, ori_w = 626, 1622
    center = ori_w // 2
    for i in range(len(cut_idx)):
        left = cut_idx[i]
        right = left + center

        filename3 = '{}_{}'.format(i, filename)
        filepath3 = os.path.join(OUTPUT_IMG_FOLDER, filename3)
        img3 = img_ori[:, left:right]
        if not os.path.exists(filepath3):
            cv2.imwrite(filepath3, img3)
        img_id_count += 1
        img_id_3 = img_id_count
        img_info['id'] = img_id_3
        img_info['file_name'] = filename3
        img_info['width'] = center
        ret['images'].append(copy.deepcopy(img_info))
        
        for annot in imgid2listannot[img_id]:
            annot_copy = copy.deepcopy(annot)
            # if img_id == 3679:
            #     print(annot['bbox'], annot_copy['bbox'])
            x, y, w, h = annot_copy['bbox']
            if x > right:
                continue
            elif x + w >= right:
                if right - x >= 0.5*w:
                    check = 1
                    annot_id_count += 1
                    annot_copy['id'] = annot_id_count
                    x_new = x - left
                    w_new = right - x
                    annot_copy['bbox'] = [x_new, y, w_new, h]
                    annot_copy['image_id'] = img_id_3
                    annot_copy['dim'] = [ori_h, center, 3]
                    ret['annotations'].append(annot_copy)
                    keep_img_id.add(img_id_3)
                    # print(annot_copy, img_id_3, check)

            elif x > left:
                check = 2
                annot_id_count += 1
                annot_copy['id'] = annot_id_count
                x_new = x - left
                annot_copy['bbox'] = [x_new, y, w, h]
                annot_copy['image_id'] = img_id_3
                annot_copy['dim'] = [ori_h, center, 3]
                ret['annotations'].append(annot_copy)
                keep_img_id.add(img_id_3)
                # print(annot_copy, img_id_3, check)

            elif x + w > left:
                if left - x < 0.5*w:
                    check = 3
                    annot_id_count += 1
                    annot_copy['id'] = annot_id_count
                    x_new = 0
                    w_new = x + w - left
                    annot_copy['bbox'] = [x_new, y, w_new, h]
                    annot_copy['image_id'] = img_id_3
                    annot_copy['dim'] = [ori_h, center, 3]
                    ret['annotations'].append(annot_copy)
                    keep_img_id.add(img_id_3)
                    # print(annot_copy, img_id_3, check)
            del annot_copy

        # if filename3 == '4_3679.png':
        #     # print(ret['annotations'])
        #     exit(0)
keep_images = [info for info in ret['images'] if info['id'] in keep_img_id]
delete_images = [info for info in ret['images'] if info['id'] not in keep_img_id]

ret['images'] = keep_images

json.dump(ret, open(JSON_OUT, 'w' ))

for info in delete_images:
    print('Delete', info['file_name'])
    try:
        os.remove(os.path.join(OUTPUT_IMG_FOLDER, info['file_name']))
    except Exception as e:
        print(e, info['file_name'])
