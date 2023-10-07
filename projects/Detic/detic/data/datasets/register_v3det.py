import logging
import os

import mmengine

from detectron2.data.datasets.register_coco import register_coco_instances
from .lvis_v1 import custom_register_lvis_instances

logger = logging.getLogger(__name__)


def _get_builtin_metadata(cat_info_path):
    categories = mmengine.load(cat_info_path)
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i
        for i, x in enumerate(sorted(categories, key=lambda x: x['id']))
    }
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    class_image_count = [{
        'id': x['id'],
        'image_count': x['image_count']
    } for x in sorted(categories, key=lambda x: x['id'])]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "class_image_count": class_image_count,
    }


_PREDEFINED_SPLITS_V3Det = {
    "v3det_ovd_train":
    ("V3Det/", 'V3Det/annotations/v3det_2023_v1_train_ovd_base.json',
     'datasets/metadata/v3det_2023_v1_train_ovd_base_cat_info.json'),
}

for key, (image_root, json_file,
          cat_info_path) in _PREDEFINED_SPLITS_V3Det.items():
    # if not os.path.exists(cat_info_path):
    #     continue
    register_coco_instances(
        key, _get_builtin_metadata(cat_info_path),
        os.path.join("datasets", json_file)
        if "://" not in json_file else json_file,
        os.path.join("datasets", image_root))

_CUSTOM_SPLITS_IMAGENET = {
    "imagenet_v3det":
    ("imagenet/", "V3Det/annotations/imagenet_v3det_image_info.json",
     'datasets/metadata/v3det_2023_v1_train_cat_info.json'),
}

from .imagenet import custom_register_imagenet_instances

for key, (image_root, json_file,
          cat_info_path) in _CUSTOM_SPLITS_IMAGENET.items():
    if not os.path.exists(cat_info_path):
        continue
    custom_register_imagenet_instances(
        key, _get_builtin_metadata(cat_info_path),
        os.path.join("datasets", json_file)
        if "://" not in json_file else json_file,
        os.path.join("datasets", image_root))
#
# _CUSTOM_SPLITS_LVIS = {
#     "lvis_v1_train_no-rare-obj365":
#     ("coco/", "lvis/annotations/lvis_v1_train_non-rare-obj365.json",
#      "datasets/metadata/lvis_v1_620_non-rare-obj365_cat_info.json"),
#     "new_lvis_v1_train_no-rare-obj365":
#     ("coco/", "lvis/annotations/new_lvis_v1_train_non-rare-obj365.json",
#      "datasets/metadata/new_lvis_v1_546_non-rare-obj365_cat_info.json"),
#     "new_lvis_v1_train_no-rare-obj365_catinfo_in_lvis_obj":
#     ("coco/",
#      "lvis/annotations/new_lvis_v1_train_non-rare-obj365_catinfo_in_lvis_obj.json",
#      "datasets/metadata/new_lvis_v1_train_non-rare-obj365_catinfo_in_lvis_obj_1274_cat_info.json"
#      ),
#     "new_detic_opendet_train_non_lvis_obj":
#     ("OpenDet/",
#      'OpenDet/annotations/new_detic_opendet_non-lvis-obj_anns_train_v2_processed.json',
#      'datasets/metadata/new_detic_opendet_non-lvis-obj_anns_train_v2_processed_13029_cat_info.json'
#      ),
# }
#
# for key, (image_root, json_file, cat_info_path) in _CUSTOM_SPLITS_LVIS.items():
#     if not os.path.exists(cat_info_path):
#         continue
#     custom_register_lvis_instances(
#         key, _get_builtin_metadata(cat_info_path),
#         os.path.join("datasets", json_file)
#         if "://" not in json_file else json_file,
#         os.path.join("datasets", image_root))
