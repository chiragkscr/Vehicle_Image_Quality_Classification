import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        self.samples = []
        self.transform = transform

        valid_ext = (".jpg", ".jpeg", ".png")

     
        blur_root = os.path.join(root_dir, "Blur", split)
        if not os.path.exists(blur_root):
            if os.path.exists(os.path.join(root_dir, "Blur", split.capitalize())):
                blur_root = os.path.join(root_dir, "Blur", split.capitalize())
            elif os.path.exists(os.path.join(root_dir, "Blur", split.title())):
                blur_root = os.path.join(root_dir, "Blur", split.title())
            elif os.path.exists(os.path.join(root_dir, "Blur", split.lower())):
                blur_root = os.path.join(root_dir, "Blur", split.lower())

        for label_name in ["blur_images", "good_images"]:
            folder = os.path.join(blur_root, label_name)

            for img_name in os.listdir(folder):
                if not img_name.lower().endswith(valid_ext):
                    continue

                img_path = os.path.join(folder, img_name)

                label = [0, 0]  # [blur, low_light]

                if label_name == "blur_images":
                    label[0] = 1

                self.samples.append((img_path, label))

      
        low_root = os.path.join(root_dir, "Low_Light", split)
        if not os.path.exists(low_root):
            if os.path.exists(os.path.join(root_dir, "Low_Light", split.capitalize())):
                low_root = os.path.join(root_dir, "Low_Light", split.capitalize())
            elif os.path.exists(os.path.join(root_dir, "Low_Light", split.title())):
                low_root = os.path.join(root_dir, "Low_Light", split.title())
            elif os.path.exists(os.path.join(root_dir, "Low_Light", split.lower())):
                low_root = os.path.join(root_dir, "Low_Light", split.lower())

        for label_name in ["low_light_images", "good_images"]:
            folder = os.path.join(low_root, label_name)

            for img_name in os.listdir(folder):
                if not img_name.lower().endswith(valid_ext):
                    continue

                img_path = os.path.join(folder, img_name)

                label = [0, 0]

                if label_name == "low_light_images":
                    label[1] = 1

                self.samples.append((img_path, label))

        import random
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.float32)

        return img, label