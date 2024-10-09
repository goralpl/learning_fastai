import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Check for MPS (Metal Performance Shaders) support on Mac
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Using device: {device}")

class PNGDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(root_dir) for f in filenames if f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((1500, 900)),
    transforms.ToTensor()
])

# Create datasets for training and validation
train_dataset = PNGDataset(root_dir='data/train', transform=transform)
val_dataset = PNGDataset(root_dir='data/validate', transform=transform)

# Create dataloaders for training and validation with num_workers=0
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Get a batch of images
data_iter = iter(train_dataloader)
images, labels = data_iter.next()

# Function to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Show images
imshow(torchvision.utils.make_grid(images))

class Autoencoder(pl.LightningModule):
    def __init__(self, val_dataloader):
        super(Autoencoder, self).__init__()
        self.val_dataloader = val_dataloader
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Dropout(0.5),  # Add dropout
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Dropout(0.5)  # Add dropout
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        reconstructed = self(batch)
        loss = nn.MSELoss()(reconstructed, batch)
        accuracy = self.calculate_accuracy(reconstructed, batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        reconstructed = self(batch)
        loss = nn.MSELoss()(reconstructed, batch)
        accuracy = self.calculate_accuracy(reconstructed, batch)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

    def calculate_accuracy(self, reconstructed, original):
        # Assuming binary classification for simplicity
        reconstructed = (reconstructed > 0.5).float()
        original = (original > 0.5).float()
        correct = (reconstructed == original).float().sum()
        total = torch.numel(original)
        accuracy = correct / total
        return accuracy

    def on_epoch_end(self):
        # Log images at the end of each epoch
        val_batch = next(iter(self.val_dataloader))
        val_batch = val_batch.to(self.device)
        reconstructed = self(val_batch)
        grid_reconstructed = torchvision.utils.make_grid(reconstructed)
        grid_original = torchvision.utils.make_grid(val_batch)
        self.logger.experiment.add_image('reconstructed_images', grid_reconstructed, self.current_epoch)
        self.logger.experiment.add_image('original_images', grid_original, self.current_epoch)

# Training
model = Autoencoder(val_dataloader)

# Set up TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="autoencoder")

# Set up EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=True,
    mode='min'
)

# Set up ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoints/',
    filename='autoencoder-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,
    mode='min'
)

# Initialize the trainer with the logger, early stopping, and checkpoint callbacks
trainer = Trainer(max_epochs=20, accelerator=device, logger=logger, callbacks=[early_stopping, checkpoint_callback])
trainer.fit(model, train_dataloader, val_dataloader)

# Extracting Embeddings
def extract_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(model.device)
            encoded = model.encoder(batch)
            embeddings.append(encoded.view(encoded.size(0), -1).cpu())
    return torch.cat(embeddings)

embeddings = extract_embeddings(model, train_dataloader)
print(embeddings.shape)  # Should be (num_images, embedding_dim)
