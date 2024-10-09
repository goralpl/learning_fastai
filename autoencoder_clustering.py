import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import hdbscan

# Check if MPS (Metal Performance Shaders) is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define the Lightning module
class LitAutoencoder(pl.LightningModule):
    def __init__(self):
        super(LitAutoencoder, self).__init__()
        self.model = Autoencoder()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.model(x)
        loss = self.criterion(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='.', train=True, transform=transform, download=True)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)

# Set up TensorBoard logger
logger = TensorBoardLogger("tb_logs", name="autoencoder")

# Set up EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='train_loss',
    patience=15,
    verbose=True,
    mode='min'
)

# Train the autoencoder with early stopping
autoencoder = LitAutoencoder()
trainer = Trainer(
    max_epochs=50,
    accelerator="mps" if torch.backends.mps.is_available() else "cpu",
    logger=logger,
    callbacks=[early_stopping]
)

train = False
if train:

    trainer.fit(autoencoder, train_loader)

    # Save the final model
    torch.save(autoencoder.model.state_dict(), "autoencoder_model.pth")

# Load the model
model = Autoencoder()
model.load_state_dict(torch.load("autoencoder_model.pth"))
model.to(device)
model.eval()

# Function to extract embeddings
def extract_embeddings(model, data_loader):
    embeddings = []
    images = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.view(x.size(0), -1).to(device)
            embedding = model.encoder(x)
            embeddings.append(embedding.cpu().numpy())
            images.append(x.cpu().numpy())
            labels.append(y.cpu().numpy())
    embeddings = np.concatenate(embeddings)
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    return embeddings, images, labels

# Extract embeddings
embeddings, images, labels = extract_embeddings(model, train_loader)


# Apply HDBSCAN clustering with adjusted parameters
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, cluster_selection_epsilon=0.1)
clusters = clusterer.fit_predict(embeddings)

# Calculate the ratio of noise points to clusters
num_noise_points = np.sum(clusters == -1)
num_cluster_points = np.sum(clusters != -1)
num_clusters = len(np.unique(clusters)) - 1  # Exclude noise label (-1)
ratio_noise_to_clusters = num_noise_points / (num_cluster_points + num_noise_points)

print(f"Number of noise points: {num_noise_points}")
print(f"Number of cluster points: {num_cluster_points}")
print(f"Number of clusters: {num_clusters}")
print(f"Ratio of noise points to clusters: {ratio_noise_to_clusters:.2f}")

# Reduce dimensionality with PCA for visualization purposes
pca = PCA(2)
embeddings_pca = pca.fit_transform(embeddings)

# Plot the clusters
plt.figure(figsize=(10, 7))
scatter = plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('HDBSCAN Clustering of MNIST Embeddings')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Function to plot some clustered images
def plot_cluster_images(cluster, images, labels, clusters, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 1.5))
    cluster_indices = np.where(clusters == cluster)[0]
    for ax, idx in zip(axes, cluster_indices[:num_images]):
        ax.imshow(images[idx].reshape(28, 28), cmap='gray')
        ax.axis('off')
    plt.show()

# Plot some images from each cluster
unique_clusters = np.unique(clusters)
for cluster in unique_clusters:
    if cluster != -1:  # Skip noise points
        print(f"Cluster {cluster}")
        plot_cluster_images(cluster, images, labels, clusters)
