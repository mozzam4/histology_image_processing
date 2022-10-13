import pytorch_lightning as pl
from LighteningHistology import LitResnet
from HistologyDataset import HistologyDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize, CenterCrop, Compose
import torch.utils.data as data

# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
train_dataset = HistologyDataset(csv_file=r'C:\Users\mozza\Documents\test\Few_patches\annotations.csv',
                                 root_dir=r'C:\Users\mozza\Documents\test\Few_patches\patches',
                                 transform=Compose([Resize(256),
                                                    CenterCrop(224),
                                                    ToTensor(),
                                                    Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])]))

train_set_size = int(len(train_dataset) * 0.7)
test_set_size = int(len(train_dataset) * 0.1)
valid_set_size = len(train_dataset) - train_set_size - test_set_size
train_set, valid_set, test_set = data.random_split(train_dataset, [train_set_size, valid_set_size, test_set_size])


train_dataloader = DataLoader(train_set, batch_size=4,
                              shuffle=True, num_workers=0)

val_dataloader = DataLoader(valid_set, batch_size=4,
                            shuffle=True, num_workers=0)

test_dataloader = DataLoader(test_set, batch_size=4,
                             shuffle=True, num_workers=0)


resnet = LitResnet()
trainer = pl.Trainer(limit_train_batches=10, limit_val_batches=10, limit_test_batches=10, max_epochs=1,
                     accelerator='cpu', auto_scale_batch_size='binsearch')
trainer.fit(model=resnet, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
trainer.test(dataloaders=test_dataloader)

