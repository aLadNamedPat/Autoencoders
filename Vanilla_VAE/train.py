import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
from NoSkip_VAE_Model import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomImageData(Dataset):
    def __init__(self, gray_dir, color_dir, transform=None):
        self.gray_dir = gray_dir
        self.color_dir = color_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(gray_dir) if os.path.isfile(os.path.join(gray_dir, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        gray_img_name = os.path.join(self.gray_dir, self.image_files[idx])
        color_img_name = os.path.join(self.color_dir, self.image_files[idx])

        gray_image = Image.open(gray_img_name).convert('L')  # Convert to grayscale
        color_image = Image.open(color_img_name).convert('RGB')  # Convert to RGB

        gray_image = gray_image.resize((128, 128), Image.Resampling.LANCZOS)
        color_image = color_image.resize((128, 128), Image.Resampling.LANCZOS)

        if self.transform:
            gray_image = self.transform(gray_image)
            color_image = self.transform(color_image)
        
        return gray_image, color_image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
gray_dir = os.path.join(project_dir, "data/gray")
color_dir = os.path.join(project_dir, "data/color")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = CustomImageData(gray_dir=gray_dir, color_dir=color_dir, transform=transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_channels = 1

out_channels = 3
latent_dim = 100
# hidden_dims = [32, 64, 128, 256, 512]
hidden_dims = [128, 128, 256, 512, 512]
vae = VAE(input_channels, out_channels, latent_dim, hidden_dims).to(device)

# Define optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)

# Training loop
epochs = 100
kld_weight = 0.1

for epoch in range(epochs):
    vae.train()
    train_loss = 0

    for gray_images, color_images in train_dataloader:
        gray_images = gray_images.to(device)
        color_images = color_images.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, _, mu, log_var = vae(gray_images)
        
        # Compute loss
        loss = vae.find_loss(reconstructed, color_images, mu, log_var, kld_weight)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}')

    # Validation loop
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for gray_images, color_images in test_dataloader:
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            reconstructed, _, mu, log_var = vae(gray_images)
            loss = vae.find_loss(reconstructed, color_images, mu, log_var, kld_weight)
            test_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Test Loss: {test_loss/len(test_dataloader)}')

# Save the model
model_path = os.path.join(project_dir, 'vae_model.pth')
torch.save(vae.state_dict(), model_path)
print(f'Model saved to {model_path}')