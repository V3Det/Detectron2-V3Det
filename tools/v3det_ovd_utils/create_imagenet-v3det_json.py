# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import copy
import os

import mmengine
from tqdm import tqdm

from detectron2.data.detection_utils import read_image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet_path',
                        default='datasets/imagenet-21k/images')
    parser.add_argument(
        '--v3det_ann_path',
        default='datasets/V3Det/annotations/v3det_2023_v1_val.json')
    parser.add_argument(
        '--out_path',
        default='datasets/V3Det/annotations/imagenet_v3det_image_info.json')
    args = parser.parse_args()

    dataset = mmengine.load(args.v3det_ann_path)
    cat_infos = dataset['categories']
    novel_cat_ids = []
    for c in cat_infos:
        if c['novel']:
            novel_cat_ids.append(c['id'])
    novel_cat_ids = set(novel_cat_ids)

    catid2inid = mmengine.load(
        'datasets/V3Det/annotations/v3det_2023_v1_category_tree.json'
    )['categoryid2treeid']

    count = 0
    images = []
    image_counts = {}
    folders = sorted(os.listdir(args.imagenet_path))
    folders = set(folders)
    for c in tqdm(cat_infos):
        cat_id = c['id']
        inid = catid2inid[str(cat_id)]
        if inid not in folders:
            continue
        in_dir = os.path.join(args.imagenet_path, inid)
        cat_images = []
        files = sorted(os.listdir(in_dir))
        for file in files:
            file_name = os.path.join(in_dir, file)
            try:
                img = read_image(file_name)
                h, w = img.shape[:2]
            except:
                continue
            image = {
                'id': count,
                'file_name': file_name,
                'pos_category_ids': [cat_id],
                'width': w,
                'height': h
            }
            count = count + 1
            cat_images.append(image)
        images.extend(cat_images)
        image_counts[cat_id] = len(cat_images)
        print(cat_id, inid, len(cat_images))
    categories = copy.deepcopy(dataset['categories'])
    for x in categories:
        if 'instance_count' in x:
            x.pop('instance_count')
        if x['id'] in image_counts:
            x['image_count'] = image_counts[x['id']]
        else:
            x['image_count'] = 0
    out = {'categories': categories, 'images': images, 'annotations': []}
    print('Writing to', args.out_path)
    mmengine.dump(out, args.out_path)
