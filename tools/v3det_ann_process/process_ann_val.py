import copy
import sys

import mmengine

ann_file = sys.argv[1]
d = mmengine.load(ann_file)
annotations = d['annotations']

ann_ids = [a['id'] for a in annotations]
max_aid = max(ann_ids)
cur_aid = max_aid + 1
num_coarse_ann = 0

new_annotations = []
for ann in d['annotations']:
    # remove coarse ann
    cat_ids = ann['category_id']
    coarse_cat_id = ann['Coarse_category_id']
    if len(cat_ids) == 0:  # remove coarse ann
        num_coarse_ann += 1
        continue
    elif len(cat_ids) == 1:
        ann['category_id'] = cat_ids[0]
        new_annotations.append(ann)
    elif len(cat_ids) > 1:
        for cat_id in cat_ids:
            _ann = copy.deepcopy(ann)
            _ann['id'] = cur_aid
            cur_aid += 1
            _ann['category_id'] = cat_id
            new_annotations.append(_ann)
    else:
        raise ValueError

d['annotations'] = new_annotations
print(f'ori anns: {len(annotations)}, new anns: {len(new_annotations)}, '
      f'remove coarse ann: {num_coarse_ann}')

mmengine.dump(d, sys.argv[1].replace('.json', '_processed.json'))
import IPython
IPython.embed()
exit()