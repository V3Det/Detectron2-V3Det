import logging
import os

import mmengine

from detectron2.data.datasets.register_coco import register_coco_instances

logger = logging.getLogger(__name__)


def _get_builtin_metadata():
    categories = mmengine.load('datasets/metadata/obj365_cat_info.json')
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i
        for i, x in enumerate(sorted(categories, key=lambda x: x['id']))
    }
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
    }


_PREDEFINED_SPLITS_OBJ365 = {
    "obj365_train": ("Objects365/train/", 'Objects365/objects365_train.json'),
    "obj365_val": ("Objects365/val/", 'Objects365/objects365_val.json'),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJ365.items():
    register_coco_instances(
        key, _get_builtin_metadata(),
        os.path.join("datasets", json_file)
        if "://" not in json_file else json_file,
        os.path.join("datasets", image_root))
