cd "${BASH_SOURCE%/*}/" || exit  # cd into the bundle and use relative paths

jsonnet --ext-str dataset_name=cifar10 --ext-code enable_pruning=false --ext-code enable_qat=false --ext-str model_backbone=vgg11 --ext-code model_head=null --ext-str model_task=image_classification --ext-str dataset_root=/shared/ml/datasets/vision/ --ext-str dataset_batch_size=32 ./base_task.jsonnet -o ./image_classifier_vgg11_null_cifar10_false_false.json
