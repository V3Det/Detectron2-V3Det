from typing import List, Union

import torch
from detectron2.data.catalog import MetadataCatalog


def get_fed_loss_cls_weights(dataset_names: Union[str, List[str]],
                             freq_weight_power=1.0):
    """
    Get frequency weight for each class sorted by class id.
    We now calcualte freqency weight using image_count to the power freq_weight_power.

    Args:
        dataset_names: list of dataset names
        freq_weight_power: power value
    """
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    # check_metadata_consistency("class_image_count", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    class_freq_meta = meta.class_image_count
    class_freq = torch.tensor([
        c["image_count"]
        for c in sorted(class_freq_meta, key=lambda x: x["id"])
    ])
    class_freq_weight = class_freq.float()**freq_weight_power
    return class_freq_weight
