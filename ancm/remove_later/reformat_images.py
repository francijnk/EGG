import os
import argparse
import torch
import random
import numpy as np
import tqdm
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

def extract_label_from_filename(filename):
    return "".join(filename.split(".")[0])  

def load_image_paths_and_labels(data_dir):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images to a fixed size (optional)
        transforms.ToTensor()          # Convert to tensor
    ])


    labels = []
    tensors = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".png"):
            labels.append(extract_label_from_filename(filename))
            # Open image and convert to tensor
            image = Image.open(os.path.join(data_dir, filename)).convert("RGB")
            tensor = transform(image)
            tensors.append(tensor)
    return tensors, labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int, default=4)
    parser.add_argument("-s", type=int, default=5)
    args = parser.parse_args()
    n_distractors = args.d
    n_samples = args.s
    return n_distractors, n_samples

def reshape_make_tensor(img_data, n_distractors, n_samples, dim=3, height=128, width=128):

    labels = []
    
    data_reshaped = torch.empty((len(img_data)*n_samples, n_distractors + 1, dim, height, width))
    for concept_i in range(len(img_data)):
        for sample_j in range(n_samples):
            target_pos = np.random.randint(0, n_distractors+1)
            distractor_pos = [i for i in range(n_distractors) if i != target_pos]
            data_reshaped[concept_i + sample_j, target_pos, :] = img_data[concept_i]
               
            # randomly pick other distractors
            distractors = [tensor for i, tensor in enumerate(img_data) if i != concept_i]
            
            distractor_ids = np.random.choice(len(img_data) - 1, n_distractors, replace=False)
            distractors = [distractors[i] for i in distractor_ids]
                
            for distractor_k, distractor_pos in enumerate(distractor_pos):
                if distractor_pos == target_pos:
                    continue
                data_reshaped[concept_i + sample_j, distractor_pos] = distractors[distractor_k]

            labels.append(target_pos)

    return data_reshaped, labels

def get_filename(n_distractors, n_samples, extension=False):
    filename = f"ancm//data/input_data//img-{n_distractors+1}-{n_samples}" 
    if extension:
        filename += '.npz'
    return filename

def reformat(n_distractors, n_samples, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    data_dir = "ancm\\data\\image_data"  # Path to folder
    tensors, labels = load_image_paths_and_labels(data_dir)

    # divide 70% for train, 15% test and valid
    train_imgs, temp_imgs, train_textlabels, temp_labels = train_test_split(tensors, labels, test_size=0.3, shuffle=True)
    valid_imgs, test_imgs, valid_textlabels, test_textlabels = train_test_split(temp_imgs, temp_labels, test_size=0.5, shuffle=True)

    # depending on the dimensions of your images
    dim = 3
    height = 128
    width = 128

    train, train_labels = reshape_make_tensor(train_imgs, n_distractors, n_samples, dim, height, width)
    valid, valid_labels = reshape_make_tensor(valid_imgs, n_distractors, n_samples, dim, height, width)
    test, test_labels = reshape_make_tensor(test_imgs, n_distractors, n_samples, dim, height, width)

    print('  train:', len(train_labels))
    print('  val:', len(valid_labels))
    print('  test:', len(test_labels))

    filename = get_filename(n_distractors, n_samples)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez(filename, train=train, valid=valid, test=test,
             train_labels=train_labels, valid_labels=valid_labels, test_labels=test_labels,
              n_distractors=n_distractors)
    print('dataset saved to ' + f"{filename}.npz")

    
def main():
    n_distractors, n_samples = parse_args()
    reformat(n_distractors, n_samples)
    


if __name__ == '__main__':
    main()