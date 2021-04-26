import pathlib

BATCH_SIZE = 100
N_CLASSES = 10  # CIFAR10
CURRENT_DIR = pathlib.Path(__file__).resolve().parent
DATASET_DATA = CURRENT_DIR / '../data'
MODEL_DIR = (CURRENT_DIR / '../data/cifar10_pretrained').resolve()
MODEL_DATA = MODEL_DIR / 'vgg11_bn.pt'

import sys
sys.path.append(str(MODEL_DIR))

import torch
import torchvision
import pytorch_lightning
import torchprof

#from fi import basefaultdescriptor, baseinjectioncallback

import vgg

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])])
cifar10_test = torchvision.datasets.CIFAR10(root=DATASET_DATA, download=True, train=False, transform=transform)
cifar10_test_dataloader = torch.utils.data.DataLoader(cifar10_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
vgg_model = vgg.vgg11_bn(pretrained=False)
vgg_model.load_state_dict(torch.load(str(MODEL_DATA), map_location=torch.device('cuda')))
vgg_model.to(torch.device('cuda')).eval()

loss_fn = torch.nn.CrossEntropyLoss()
accuracy_fn = pytorch_lightning.metrics.Accuracy()

loss_acc = 0
accuracy_acc = 0
total_length = len(cifar10_test_dataloader)


for b, batch in enumerate(cifar10_test_dataloader, start=1):
    with torch.no_grad():
        images, labels = batch
        images_cuda = images.to(torch.device('cuda'))

        # we profile only at first batch
        with torchprof.Profile(vgg_model, enabled=b == 1, use_cuda=True, profile_memory=True) as prof:
            predictions_cuda = vgg_model(images_cuda)

        if b == 1:
            print(prof.display(show_events=True))

        predictions = predictions_cuda.to(torch.device('cpu'))

        del images_cuda
        del predictions_cuda

        torch.cuda.empty_cache()

    loss = loss_fn(predictions, labels)
    # cannot be done as there is probability for each class
    #accuracy = sum(p.item() == l.item() for p, l in zip(predictions.flatten(), labels.flatten())) / N_CLASSES * 100
    accuracy = accuracy_fn(predictions, labels) * 100

    loss_acc += loss
    accuracy_acc += accuracy

    print(f"Batch #{b}: Loss {loss}; Accuracy {accuracy} %")

print(f"Final Results: Average Loss {loss_acc/total_length}; Average Accuracy {accuracy_acc/total_length} %")
