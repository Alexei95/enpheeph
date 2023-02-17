
Script order:

1. Training script, example to save also the printed statement, e.g. the epoch metrics and checkpoiint locations
  1. `CUDA_VISIBLE_DEVICES=1 python training_script.py configs/python/image_classifier_vgg11_gtsrb.py &| tee results/image_classifier_vgg11_gtsrb/training.txt`
2. Then we need to select the proper checkpoints, and move them into a final folder
3. Afterwards, we can compute the attributions using `attribution_generator.py`
  1. Here the settings are inside the file itself, and it is as easy to call as using `CUDA_VISIBLE_DEVICES=0 python attribution_generator.py`
4. After the attribution is generated, we can finally run the injection
  1. The prototype command is `CUDA_VISIBLE_DEVICES=2 python injector_script.py --python-config configs/python/inference_pruned_image_classifier_resnet18_cifar10.py --checkpoint-file results/sparse_results_ijcnn2023/trained_networks/resnet18_cifar10/epoch=30-step=38750_0_pruning.ckpt --seed 1600 --number-iterations 1 --result-database results/sparse_results_ijcnn2023/test/resnet18_cifar10_0_pruning/db.sqlite --save-attribution-file results/sparse_results_ijcnn2023/attributions/resnet18_cifar10/epoch-30-step-38750_0_pruning/sparse_value_weight_seed_1600_attributions.pt --random-threshold 1 --injection-type weight --sparse-target value`
  2. Although it is better to use `python parallel_injector_script.py --python-config configs/python/inference_pruned_image_classifier_resnet18_cifar10.py --checkpoint-file results/sparse_results_ijcnn2023/trained_networks/resnet18_cifar10/epoch=7-step=10000_0_pruning.ckpt --starting-seed 1600 --number-iterations 10000000 --result-folder results/sparse_results_ijcnn2023/random_sampling/resnet18_cifar10/epoch-7-step10000_0_pruning --load-attribution-file results/sparse_results_ijcnn2023/attributions/resnet18_cifar10/epoch-7-step-10000_0_pruning/sparse_index_weight_seed_1600_attributions.pt --random-threshold 1 --injection-type weight --sparse-target index --devices 3`
