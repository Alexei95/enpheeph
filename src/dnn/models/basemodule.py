import abc

import torch
import pytorch_lightning as pl

class BaseModule(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.enable = False

        # mnist images are (1, 28, 28) (channels, width, height) 
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)


    def forward(self, *args, **kwargs):
        return None

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        print(loss)
        return {'test_loss': loss}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # called at the end of the validation epoch
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}] 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])
        
        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        
        self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        self.mnist_test = mnist_test

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=64)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_test_start(self):
        if self.enable:
            self.layer_1 = torch.nn.Sequential(self.layer_1, torch.nn.Softmax())

    def enable_(self):
        self.enable = True

# train
model = LightningMNISTClassifier()
trainer = pl.Trainer(gpus=1, max_epochs=1)

trainer.fit(model)

print(trainer.test(model))

#model.layer_1 = torch.nn.Sequential(model.layer_1, torch.nn.Softmax())
model.enable_()
print(trainer.test(model))

# trainer.test() does not work if we change the model
#for i, test_batch in enumerate(model.test_dataloader()):
#    x, y = test_batch
#    logits = model.forward(x.to(model.device))
#    loss = model.cross_entropy_loss(logits, y.to(model.device))
#    print({'test_loss': loss})
