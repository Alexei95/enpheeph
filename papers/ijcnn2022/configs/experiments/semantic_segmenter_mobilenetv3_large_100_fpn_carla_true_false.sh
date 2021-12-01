cd "${BASH_SOURCE%/*}/" || exit  # cd into the bundle and use relative paths

jsonnet --ext-str dataset_name=carla --ext-code enable_pruning=true --ext-code enable_qat=false --ext-str model_backbone=mobilenetv3_large_100 --ext-str model_head=fpn --ext-str model_task=semantic_segmentation --ext-str dataset_root=/shared/ml/datasets/vision/ --ext-str dataset_batch_size=16 ./base_task.jsonnet -o ./semantic_segmenter_mobilenetv3_large_100_fpn_carla_true_false.json
