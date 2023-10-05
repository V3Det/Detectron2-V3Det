import copy
import sys

import mmengine

logger = mmengine.MMLogger.get_current_instance()

ann_file = sys.argv[1]
d = mmengine.load(ann_file)

categories = d['categories']
images = d['images']

base_cats = []
novel_cats = []

assert len(categories) == 13204

for cat in categories:
    if cat['novel']:
        novel_cats.append(cat)
    else:
        base_cats.append(cat)

cat_to_imgs = dict()
cat_to_ins_count = dict()
for c in categories:
    cat_to_ins_count[c['id']] = 0
    cat_to_imgs[c['id']] = set()

base_cat_ids = set([c['id'] for c in base_cats])

base_anns = []
for ann in d['annotations']:
    cat_id = ann['category_id']
    img_id = ann['image_id']
    if cat_id in base_cat_ids:
        base_anns.append(ann)
    cat_to_ins_count[cat_id] += 1
    cat_to_imgs[cat_id].add(img_id)
cat_to_img_count = {k: len(v) for k, v in cat_to_imgs.items()}

cats = copy.deepcopy(categories)

for c in cats:
    if c['id'] in base_cat_ids:
        c['image_count'] = cat_to_img_count[c['id']]
        c['instance_count'] = cat_to_ins_count[c['id']]
    else:
        c['image_count'] = 0
        c['instance_count'] = 0

base_img_ids = set(ann['image_id'] for ann in base_anns)
base_imgs = [img for img in images if img['id'] in base_img_ids]

base_d = dict(images=base_imgs, categories=cats, annotations=base_anns)

logger.info(f'base ann file, {len(base_cats)} cats, '
            f'{len(base_imgs)} images, '
            f'{len(base_anns)} anns')

mmengine.dump(base_d, ann_file.replace('.json', '_ovd_base.json'))
