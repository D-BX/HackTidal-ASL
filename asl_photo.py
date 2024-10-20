# %%

import os
import torch
import re
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

from skimage import filters, morphology, segmentation, color
from skimage.measure import label
from skimage.color import rgb2gray


import random

from PIL import Image
import requests
import kagglehub

import numpy as np

import pandas as pd

import decord

from torchvision import transforms
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

# %%

# Download latest version
#path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
path = kagglehub.dataset_download("debashishsau/aslamerican-sign-language-aplhabet-dataset")

# %%

instances = []
#path = 'kagglehub/datasets/risangbaskoro/wlasl-processed/versions/5/'
path = 'kagglehub/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset/versions/1/ASL_Alphabet_Dataset/'
alpha_list = [chr(v) for v in range(ord('A'), ord('A') + 26)]
pattern = r'^\d+\.jpg$'
for letter in alpha_list:
    for file in os.listdir(path + 'asl_alphabet_train/' + letter):
        if re.match(pattern, file):
            instances.append((ord(letter) - ord('A'), path + 'asl_alphabet_train/' + letter + '/' + file))

random.seed(42)
random.shuffle(instances)

test_instances = instances[:500]
instances = instances[500:]

# %%

def extract_photo(img_path):
    img = Image.open(img_path).convert("RGB")

    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Resize the image to a fixed size (example)
        transforms.ToTensor(),  # Resize the image to a fixed size (example)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations
    image = preprocess(img)

    
    # Add an extra batch dimension since models expect batches of images [batch_size, channels, height, width]
    image = image.unsqueeze(0)
    return image

box = extract_photo(instances[0][1])

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
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        return self.linear2(self.relu(self.linear(x)))

# %%

num_classes = 26 # Number of classes (for classification task)
hidden_dim = dino_dim * 4
batch_size = 32 # Batch size
learning_rate = 0.001 # Learning rate
num_epochs = 4 # Number of training epochs

model = LinearClassifier(dino_dim, hidden_dim, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    
    total_loss = 0


    random.shuffle(instances)
    labels, image_paths = tuple(map(list, zip(*instances)))

    for i in tqdm(range(0, 1000, batch_size)):
        image_file_batch = image_paths[i:i + batch_size]
        images = [extract_photo(image_file) for image_file in image_file_batch]
        images = torch.stack(images, dim=1)[0]

        label_batch = labels[i:i + batch_size]
        label_batch = torch.tensor(label_batch, device=device)

        with torch.no_grad():
            embeds = dino(images.to(device)).last_hidden_state[:, 0, :].reshape(batch_size, dino_dim)

        predicts = model(embeds)


        loss = criterion(predicts, label_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss for reporting
        total_loss += loss.item()
    
    # Print loss after every epoch or at intervals
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(instances):.4f}')

torch.save(model, "prefinetune.pt")

# %%

fine_tune_instances = []

fine_pattern = r'([a-zA-Z])1\.jpg'
for file in os.listdir('photo/'):
    match = re.match(fine_pattern, file)
    print(file)
    if match:
        letter = match.group(1)      # The captured letter (first group)
        fine_tune_instances.append(((ord(letter) - ord('a')), 'photo/' + file))

random.shuffle(fine_tune_instances)

# %%

for epoch in range(3):
    model.train()
    
    total_loss = 0

    labels, image_paths = tuple(map(list, zip(*fine_tune_instances)))

    for i in tqdm(range(0, len(fine_tune_instances), 8)):
        image_file_batch = image_paths[i:i + batch_size]
        images = [extract_photo(image_file) for image_file in image_file_batch]
        images = torch.stack(images, dim=1)[0]
        this_batch_size = images.shape[0]

        label_batch = labels[i:i + batch_size]
        label_batch = torch.tensor(label_batch, device=device)

        with torch.no_grad():
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(instances):.4f}')
# %%

labels, image_paths = tuple(map(list, zip(*test_instances)))

total_correct = 0

for i in tqdm(range(0, len(test_instances), batch_size)):
    image_file_batch = image_paths[i:i + batch_size]
    images = [extract_photo(image_file) for image_file in image_file_batch]
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

print(total_correct / len(test_instances))

# Print loss after every epoch or at intervals
# %%

def predict_image(image_tensor):
    print(image_tensor.shape)

    with torch.no_grad():
        embeds = dino(image_tensor.to(device)).last_hidden_state[:, 0, :].reshape(1, dino_dim)

    predict = model(embeds)
    print(predict)
    predicted_class = predict.max(dim=1).indices
    return chr(ord('A') + predicted_class[0])

photo = extract_photo("B_clear.jpg")
guess = predict_image(photo)
print(guess)
# %%


test_fine_insts = []

fine_pattern2 = r'([a-zA-Z])2\.jpg'
for file in os.listdir('photo/'):
    match = re.match(fine_pattern2, file)
    if match:
        letter = match.group(1)      # The captured letter (first group)
        test_fine_insts.append(((ord(letter) - ord('a')), 'photo/' + file))

random.shuffle(test_fine_insts)

# %%

labels, image_paths = tuple(map(list, zip(*test_fine_insts)))

total_correct = 0

for i in tqdm(range(0, len(test_fine_insts), batch_size)):
    image_file_batch = image_paths[i:i + batch_size]
    images = [extract_photo(image_file) for image_file in image_file_batch]
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

print(total_correct / len(test_fine_insts))
# %%
