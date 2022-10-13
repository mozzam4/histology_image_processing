from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import torch
import torch.functional as F
from torchvision.models import resnet50, ResNet50_Weights


# define the LightningModule
class LitResnet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fc1 = torch.nn.Linear(1000, 256)
        self.fc2 = torch.nn.Linear(256, 56)
        self.fc3 = torch.nn.Linear(56, 1)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.model(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu_(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu_(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch['image']
        y = batch['if_msi']
        x = self(x)
        loss = nn.functional.mse_loss(x, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['if_msi']
        x = self(x)
        loss = nn.functional.mse_loss(x, y)
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['if_msi']
        #x, y = batch
        logits = self(x)
        loss = nn.functional.mse_loss(logits, y)
        y_hat = torch.argmax(logits, dim=1)
        accuracy = torch.sum(y == y_hat).item() / (len(y) * 1.0)
        output = dict({
            'test_loss': loss,
            'test_acc': torch.tensor(accuracy),
        })
        return output

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer





# # load checkpoint
# checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
# autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)
#
# # choose your trained nn.Module
# encoder = autoencoder.encoder
# encoder.eval()
#
# # embed 4 fake images!
# fake_image_batch = Tensor(4, 28 * 28)
# embeddings = encoder(fake_image_batch)
# print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)
