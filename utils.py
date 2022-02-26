from torchvision import transforms


batch_size = 32
noise_dim = 100
resize_img = 28
epochs = 5
lr = 0.01
disc_epochs = 1

transform = transforms.Compose([
    transforms.ToTensor()
])
