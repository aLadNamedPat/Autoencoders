import torch
from NoSkip_VAE_Model import VAE
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

input_channels = 1 # Grayscale images have 1 channel
out_channels = 3
latent_dim = 128
hidden_dims = [128, 128, 256, 512, 512]

vae = VAE(input_channels, out_channels, latent_dim, hidden_dims).to(device)

# Load the saved model state dictionary
model_path = os.path.join(current_dir, 'vae_model.pth')
vae.load_state_dict(torch.load(model_path))
vae.eval()

class CustomTestDataset(Dataset):
    def __init__(self, image_dir, color_dir, transform=None, color_transform = None):
        self.image_dir = image_dir
        self.color_dir = color_dir
        self.transform = transform
        self.color_transform = color_transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        color_img = os.path.join(self.color_dir, self.image_files[idx])
        gray_image = Image.open(img_name).convert('L')  # Convert to grayscale
        color_image = Image.open(color_img).convert('RGB')  # Load as RGB for comparison

        gray_image = gray_image.resize((128, 128), Image.Resampling.LANCZOS)
        color_image = color_image.resize((128, 128), Image.Resampling.LANCZOS)

        if self.transform:
            gray_image = self.transform(gray_image)
            color_image = self.color_transform(color_image)

        return gray_image, color_image

# Define your transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#Second Transform
color_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
gray_dir = os.path.join(project_dir, "data/gray")
color_dir = os.path.join(project_dir, "data/color")

test_dataset = CustomTestDataset(image_dir=gray_dir, color_dir = color_dir, transform=transform, color_transform = color_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0)
    tensor = tensor * 0.5 + 0.5
    tensor = tensor.clamp(0, 1)
    return T.ToPILImage()(tensor)

def show_images(original_gray, reconstructed_color, actual_color):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(tensor_to_image(original_gray), cmap='gray')
    axs[0].set_title('Original Gray Image')
    axs[0].axis('off')

    axs[1].imshow(tensor_to_image(reconstructed_color))
    axs[1].set_title('Reconstructed Color Image')
    axs[1].axis('off')

    axs[2].imshow(tensor_to_image(actual_color))
    axs[2].set_title('Actual Color Image')
    axs[2].axis('off')

    plt.show()

# Process and visualize the images
for gray_image, color_image in test_dataloader:
    gray_image = gray_image.to(device)
    color_image = color_image.to(device)
    with torch.no_grad():
        reconstructed_image, _, _, _ = vae(gray_image)

    gray_image = gray_image.cpu()
    reconstructed_image = reconstructed_image.cpu()
    color_image = color_image.cpu()

    show_images(gray_image[0], reconstructed_image[0], color_image[0])