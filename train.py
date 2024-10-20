# %%

import os
import torch
import re
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

import random

from PIL import Image
import requests

import numpy as np

import pandas as pd

#import kagglecord

import decord

from torchvision import transforms
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

# %%

class CustomCrop:
    def __call__(self, img):
        # Get the original dimensions of the image
        width, height = img.size
        
        # Calculate the cropping dimensions
        new_width = width * 2 // 3  # Middle two-thirds of the width
        new_height = height * 2 // 3  # Bottom two-thirds of the height
        
        left = (width - new_width) // 2  # Left offset for horizontal center crop
        top = height // 2  # Discard top third, so we start from 1/3 of height
        right = left + new_width
        bottom = height

        # Crop the image
        return img.crop((left, top, right, bottom))



def extract_photo(img):
    preprocess = transforms.Compose([
        CustomCrop(),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.Resize((512, 512)),  # Resize the image to a fixed size (example)
        transforms.ToTensor(),  # Resize the image to a fixed size (example)
#        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations
    image = preprocess(img)
    
    # Add an extra batch dimension since models expect batches of images [batch_size, channels, height, width]
    image = image.unsqueeze(0)
    return image

# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
dino_dim = 768

def video_to_embedding(video):
    frames = video.shape[0]
    reshaped_video = np.transpose(video, (0, 3, 1, 2))
    with torch.no_grad():
        output = dino(torch.tensor(reshaped_video, device=device))

    return output.last_hidden_state[:, 0, :].reshape(frames, dino_dim).cpu()

# %%

class LinearClassifier(nn.Module):
#    def __init__(self, input_dim, hidden_dim, num_classes):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
#        self.relu = nn.ReLU()
#        self.linear2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
#        return self.linear2(self.relu(self.linear(x)))
        return self.linear(x)

# %%

def sample_video_to_images(video_path, frame_rate=1, output_folder="video_frames"):
    video_reader = decord.VideoReader(video_path)
   
    # Get the video's frame rate and total frame count
    video_fps = video_reader.get_avg_fps()
    total_frames = len(video_reader)
   
    # Calculate the frame interval based on the desired frame rate (in seconds)
    sample_interval = int(video_fps * frame_rate)
   
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through frames and save at the specified interval
    images = []
    for frame_idx in range(0, total_frames, sample_interval):
        # Get the frame at frame_idx
        frame = video_reader[frame_idx]
       
        # Convert the frame (which is a decord NDArray) to a numpy array
        frame_np = frame.asnumpy()

        # Convert the numpy array to a PIL Image
        image = Image.fromarray(frame_np)
        images.append(image)

    return images

instances = []
for file in os.listdir('HackTidal-ASL/new_videos'):
    print(file)
    short_name = file.split('.')[0]
    if (short_name == 'Space'):
        continue

    if (short_name == 'X'):
        continue
    frames = sample_video_to_images("HackTidal-ASL/new_videos/" + file, frame_rate=0.5)  # Extract 1 frame per second
    new_insts = [((ord(short_name) - ord('A'), frame)) for frame in frames]

    instances.extend(new_insts)

null_images = sample_video_to_images("HackTidal-ASL/new_videos/Space.mp4", frame_rate=0.5)
null_frames = [(26, null_image) for null_image in null_images]
instances.extend(null_frames)

# %%

random.seed(42)
random.shuffle(instances)

train_amount = int(len(instances) * 0.8)

train_inst = instances[:train_amount]
test_inst = instances[train_amount:]

# %%

num_classes = 27 # Number of classes (for classification task)
#hidden_dim = dino_dim * 4
batch_size = 32 # Batch size
learning_rate = 0.001 # Learning rate
num_epochs = 10 # Number of training epochs

model = LinearClassifier(dino_dim, num_classes).to(device)

# %%

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(5):
    model.train()
    
    total_loss = 0

    random.shuffle(train_inst)
    labels, images_all = tuple(map(list, zip(*train_inst)))

    optimizer.zero_grad()

    for i in tqdm(range(0, len(train_inst), batch_size)):
        with torch.no_grad():
          image_batch = images_all[i:i + batch_size]
          images = [extract_photo(image) for image in image_batch]
          images = torch.stack(images, dim=1)[0]
          this_batch_size = images.shape[0]

          label_batch = labels[i:i + batch_size]
          label_batch = torch.tensor(label_batch, device=device)

          embeds = dino(images.to(device)).last_hidden_state[:, 0, :].reshape(this_batch_size, dino_dim)

        predicts = model(embeds)

        loss = criterion(predicts, label_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss for reporting
        total_loss += loss.item()
    
    # Print loss after every epoch or at intervals
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_inst):.4f}')

# %%

model.eval()

random.shuffle(test_inst)
labels, images_all = tuple(map(list, zip(*test_inst)))

total_correct = 0

for i in tqdm(range(0, len(test_inst), batch_size)):
    images_batch = images_all[i:i + batch_size]
    images = [extract_photo(image) for image in images_batch]
    images = torch.stack(images, dim=1)[0]
    this_batch_size = images.shape[0]

    label_batch = labels[i:i + batch_size]
    label_batch = torch.tensor(label_batch, device=device)

    with torch.no_grad():
        embeds = dino(images.to(device)).last_hidden_state[:, 0, :].reshape(this_batch_size, dino_dim)

    predicts = model(embeds)
    predicted_classes = predicts.max(dim=1).indices
    correct = (predicted_classes == label_batch).sum()

    total_correct += correct

print(f"Accuracy estimate: {total_correct / len(test_inst)}")

# Print loss after every epoch or at intervals
# %%

model = torch.load("new2.pt")
def predict_image(image_tensor):
    print(image_tensor.shape)

    with torch.no_grad():
        embeds = dino(image_tensor.to(device)).last_hidden_state[:, 0, :].reshape(1, dino_dim)

    predict = model(embeds)
    print(predict)
    print(predict.max(dim=1))
    predicted_class = predict.max(dim=1).indices
    return chr(ord('A') + predicted_class[0])

img = Image.open("videos/test2.jpg").convert("RGB")
photo = extract_photo(img)
guess = predict_image(photo)
print(guess)
# %%
