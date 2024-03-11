# # Recipe1M+ Dataset

# ## Layers

# ### layer1.json

# ```js
# {
#   id: String,  // unique 10-digit hex string
#   title: String,
#   instructions: [ { text: String } ],
#   ingredients: [ { text: String } ],
#   partition: ('train'|'test'|'val'),
#   url: String
# }
# ```

# data/det_ingrs_processed.json
# '''js
# [
#   {
#     "id": String, // unique 10-digit hex
#     "ingredients": [ {
#       "text": String
#     } ]
#   }
# ]
# '''

# ### layer2+.json

# ```js
# {
#   id: String,   // refers to an id in layer 1
#   images: [ {
#     id: String, // unique 10-digit hex + .jpg
#     url: String
#   } ]
# }
# ```

# ## Images

# The images in each of the partitions, train/val/test, are arranged in a four-level hierarchy corresponding to the first four digits of the image id.

# For example: `val/e/f/3/d/ef3dc0de11.jpg`

# The images are in RGB JPEG format and can be loaded using standard libraries.

import json
import os
import urllib.request
import ijson
import pickle
from PIL import Image
import torch
from torchvision import transforms
import random
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
import tarfile
import io


class Recipe1MPlusDataset(torch.utils.data.Dataset):
    '''special dataloader for Recipe1M+ dataset.'''

    def __init__(self, data, ingrs, index, data_path, img_size, transform=None, web=False):
        self.data = data
        self.classes = ingrs
        self.num_ingrs = len(ingrs)
        self.transform = transform
        self.img_size = img_size
        self.use_web = web
        self.data_path = data_path
        self.index = index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load the image if path exists
        if self.use_web:
            path = self.get_path_web(idx)
            img = self.load_image_web(path)
        else:
            path = self.get_path(idx)
            img = self.load_image(path)

        # Create one-hot encoded vector for the label
        label = self.get_label(idx)

        return img, label

    def __getitems__(self, idxs):
        '''Use ThreadPool to load images in parallel.'''
        if self.use_web:
            paths = [self.get_path_web(idx) for idx in idxs]
            with ThreadPool(32) as p:
                images = p.map(self.load_image_web, paths)
                labels = p.map(self.get_label, idxs)
        else:
            paths = [self.get_path(idx) for idx in idxs]
            with ThreadPool(32) as p:
                images = p.map(self.load_image, paths)
                labels = p.map(self.get_label, idxs)

        # Zip the images and labels
        return list(zip(images, labels))

    def get_path(self, idx):
        '''Get the path for the given index.'''
        return self.data[idx][0]

    def load_image(self, image_name):
        '''Load image from tar file using the index.'''
        # Load image from tar file
        with tarfile.open(f"{self.data_path}\\recipe1M+_{image_name[0]}.tar", 'r') as tar:
            info = self.index[image_name]

            # Seek directly to the start of the file data
            tar.fileobj.seek(info[0])

            # Read the file data
            file_obj = tar.fileobj.read(info[1])
            if file_obj:
                # Open the image
                img = Image.open(io.BytesIO(file_obj)).convert('RGB')
            else:
                print(f"Image {image_name} does not exist.")
                return None

        # Apply the transform
        if self.transform:
            img = self.transform(img)

        return img

    def get_path_web(self, idx):
        '''Get the path for the given index.'''
        return f"http://wednesday.csail.mit.edu/temporal/release/recipe1M+_images/{'/'.join(self.data[idx][0][:4])}/{self.data[idx][0]}"

    def load_image_web(self, path):
        '''Load image from path.'''
        image = urllib.request.urlopen(path)

        # Check if the image exists and is of type jpeg
        if image.status != 200 or image.getheader('Content-Type') != 'image/jpeg':
            print(f"Image {path} does not exist.")
            return None

        # Open the image
        img = Image.open(image).convert('RGB')

        # Apply the transform
        if self.transform:
            img = self.transform(img)

        return img

    def get_label(self, idx):
        '''Get the labels for the given index.'''
        label = torch.zeros(self.num_ingrs, dtype=torch.float32)
        for i in self.data[idx][1]:
            label[i] = 1
        return label


def preprocess_images(images):
    """Preprocess images."""
    # TODO: Implement image preprocessing
    return images


def load_data(dataset_files, data_path):
    """Load the data from the JSON files."""
    # Create a dictionary to store the data
    data = {'train': [], 'val': [], 'test': []}
    ingrs = set()
    recipes_dict = {}

    print('Reading Ingredients...')

    # Load the data from the JSON files
    with open('data/det_ingrs_processed.json', 'rb') as f:
        layer1 = json.load(f)

        # Get the number of recipes
        num_recipes = len(layer1)

        # Get the unique ingredients
        for recipe in tqdm(layer1, total=num_recipes):
            recipes_dict[recipe['id']] = set()
            for ingredient in recipe['ingredients']:
                ingrs.add(ingredient['text'])
                recipes_dict[recipe['id']].add(ingredient['text'])

    ingr_index = {ingr: idx for idx, ingr in enumerate(ingrs)}

    # Load the data from data/recipes_removed.txt as a set
    with open('data/recipes_removed.txt', 'r') as f:
        recipes_removed = set(f.read().splitlines())

    print('Reading Recipes & Images...')
    with open('data/layer2+.json', 'rb') as f:
        i = 0
        for recipe in tqdm(ijson.items(f, 'item'), total=num_recipes+len(recipes_removed)):
            if recipe['id'] not in recipes_removed:
                # Save ingredients by list of indecies for multi-label classification
                ingrs_vector = torch.zeros(len(recipes_dict[recipe['id']]), dtype=torch.int16)
                for j, ingr in enumerate(recipes_dict[recipe['id']]):
                    ingrs_vector[j] = ingr_index[ingr]

                # Save the data
                for image in recipe['images']:
                    # If the image is in the tar files
                    if image['id'][0] in dataset_files:
                        data[layer1[i]['partition']].append((image['id'], ingrs_vector))
                i += 1

    print('Loading tar indecies...')
    # Load the tar indecies
    index = {}
    for file in tqdm(dataset_files):
        with tarfile.open(f"{data_path}\\recipe1M+_{file}.tar", 'r') as tar:
            # Build the index
            for member in tar.getmembers():
                if member.isfile():
                    # Split the path and check if it's 4 folders deep (5 parts including the filename)
                    parts = member.name.split('/')
                    if len(parts) == 5:
                        index[parts[-1]] = (member.offset_data, member.size)

    return data, ingrs, index


def transform(img_size):
    """Transform images."""
    return transforms.Compose([
        transforms.Resize(img_size),  # rescale the image with the maximum size of img_size while keeping the aspect ratio
        transforms.CenterCrop(img_size),  # crop the image at the center to img_sizeximg_size
        transforms.ToTensor(),
        # Normalize the images with mean and standard deviation from paper (https://github.com/torralba-lab/im2recipe-Pytorch/blob/master/image2embedding.py line 58)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def augment_transform():
    """Transform images with data augmentation."""
    # TODO: Tweak the parameters for data augmentation
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
        transforms.RandomRotation(random.randint(0, 10)),
        # transforms.RandomErasing(p=0.9, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        # Add random noise
        transforms.Lambda(lambda x: torch.clamp(x + 0.01 * torch.randn_like(x), 0, 1)),
    ])


def create_dataset(dataset_files, data_path):
    """Create dataset train/validation/test split as DataLoader."""

    # Load the data
    print('Reading the data...')
    data, ingrs, index = load_data(dataset_files, data_path)

    # pickle the data
    with open('data/data.pkl', 'wb') as f:
        print('Pickling the data...')
        pickle.dump((data, ingrs, index), f)

    return data, ingrs, index


def load_dataset(img_size):
    """Load the dataset from the pickle file."""
    # Path to folder of tars
    data_path = "D:"

    # If the pickle file exists, load the data from the file
    if os.path.exists('data/data.pkl'):
        print('Loading the data from the pickle file...')
        with open('data/data.pkl', 'rb') as f:
            data, ingrs, index = pickle.load(f)
    else:
        print('Creating the dataset...')

        # List of dataset files
        dataset_files = ['0', '1', '2']
        # dataset_files = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

        # Create the dataset
        data, ingrs, index = create_dataset(dataset_files, data_path)
        
    # count ouccurences of each ingredient in dataset
    ingrs_count = {}
    for dataset in [data['train'], data['val']]:
        for recipe in dataset:
            for ingr in recipe[1]:
                if int(ingr) in ingrs_count:
                    ingrs_count[int(ingr)] += 1
                else:
                    ingrs_count[int(ingr)] = 1
    # Weight for the loss function (inverse of the frequency of the ingredient)
    weights = torch.tensor([1/ingrs_count[i] for i in range(len(ingrs_count))]) * max(ingrs_count.values())
    print('sorted ingredients by count:', sorted(ingrs_count.items(), key=lambda x: x[1], reverse=True))
    
    # Create dataset
    print('Creating dataset classes...')
    train_data = Recipe1MPlusDataset(data['train'], ingrs, index, data_path, img_size, transform=transforms.Compose([transform(img_size), augment_transform()]))
    val_data = Recipe1MPlusDataset(data['val'], ingrs, index, data_path, img_size, transform=transform(img_size))
    test_data = Recipe1MPlusDataset(data['test'], ingrs, index, data_path, img_size, transform=transform(img_size))

    return train_data, val_data, test_data, weights
