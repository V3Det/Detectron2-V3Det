import copy
import sys

import mmengine
from mmengine import ProgressBar

ann_file = sys.argv[1]
d = mmengine.load(ann_file)

# merge coarse categories
categories = d['categories']
coarse_categories = d['Coarse_categories']
merged_categories = copy.deepcopy(categories)
cat_ids = [c['id'] for c in categories]
max_cat_id = max(cat_ids)
coarse_cat_id_map = dict()
for i, coarse_cat in enumerate(coarse_categories):
    coarse_cat = copy.deepcopy(coarse_cat)
    cat_id = coarse_cat['id']
    new_cat_id = max_cat_id + i + 1
    coarse_cat['id'] = new_cat_id
    coarse_cat_id_map[cat_id] = new_cat_id
    merged_categories.append(coarse_cat)

# split multiple categories ann
annotations = d['annotations']

ann_ids = [a['id'] for a in annotations]
max_ann_id = max(ann_ids)
cur_ann_id = max_ann_id + 1
new_annotations = []

bar = ProgressBar(len(annotations))
for ann in annotations:
    ann = copy.deepcopy(ann)
    ann['is_coarse'] = False
    ann['is_split'] = False
    cat_ids = ann['category_id']
    if len(cat_ids) > 1:  # split multiple categories ann
        ann['category_id'] = cat_ids[0]
        new_annotations.append(ann)
        ori_ann_id = ann['id']
        for cat_id in cat_ids[1:]:
            _ann = copy.deepcopy(ann)
            _ann['id'] = cur_ann_id
            cur_ann_id += 1
            _ann['category_id'] = cat_id
            _ann['ori_id'] = ori_ann_id
            _ann['is_split'] = True
            new_annotations.append(_ann)
    elif len(cat_ids) == 0:  # coarse ids
        coarse_cat_id = ann['Coarse_category_id']
        ann['category_id'] = coarse_cat_id_map[coarse_cat_id]
        ann['is_coarse'] = True
        new_annotations.append(ann)
    elif len(cat_ids) == 1:
        ann['category_id'] = cat_ids[0]
        new_annotations.append(ann)
    else:
        raise NotImplementedError

    bar.update()
print()

# check
num_split_anns = 0
num_coarse_anns = 0
for ann in annotations:
    if len(ann['category_id']) == 0:
        num_coarse_anns += 1
    if len(ann['category_id']) > 1:
        num_split_anns += len(ann['category_id']) - 1
coarse_anns = [ann for ann in new_annotations if ann['is_coarse']]
split_anns = [ann for ann in new_annotations if ann['is_split']]
assert num_split_anns == len(split_anns), f'{num_split_anns, len(split_anns)}'
assert num_coarse_anns == len(
    coarse_anns), f'{num_coarse_anns, len(coarse_anns)}'

print(f'ori cats: {len(categories)}, new cats: {len(merged_categories)}')
print(f'ori anns: {len(annotations)}, new anns: {len(new_annotations)}, '
      f'split anns: {len(split_anns)}, coarse anns: {len(coarse_anns)}')

d['categories'] = merged_categories
d['annotations'] = new_annotations

out_file = ann_file.replace('.json', '_processed.json')

print(f'dump to {out_file}')
mmengine.dump(d, out_file)