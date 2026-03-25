
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class TabularPreprocessor:
    def __init__(self, num_cols, cat_cols, min_cat_count=100, max_num_cat=20):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.min_cat_count = min_cat_count
        self.max_num_cat = max_num_cat
        self.num_means = None
        self.num_stds = None
        self.cat_maps = {}
        self.cat_default = {}

    def fit(self, df):
        # Numerical columns: fit mean/std
        self.num_means = df[self.num_cols].mean()
        self.num_stds = df[self.num_cols].std().replace(0, 1)
        # Categorical columns: frequent category grouping and index mapping
        for col in self.cat_cols:
            counts = df[col].value_counts()
            valid = counts[counts >= self.min_cat_count].index.tolist()
            if len(valid) > self.max_num_cat:
                valid = counts.nlargest(self.max_num_cat).index.tolist()
            mapping = {cat: idx+1 for idx, cat in enumerate(valid)}
            self.cat_maps[col] = mapping
            self.cat_default[col] = 0 

    def transform(self, df):
        # Numerical
        num = (df[self.num_cols] - self.num_means) / self.num_stds
        num = num.astype(float)  
        num = num.fillna(0)      
        num = num.to_numpy(dtype=np.float32)      
        # Categorical
        cats = []
        for col in self.cat_cols:
            mapped = df[col].map(self.cat_maps[col]).fillna(self.cat_default[col]).astype(np.int64)
            cats.append(mapped.to_numpy())
        cats = np.stack(cats, axis=1)
        return num, cats

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

def get_train_image_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        #transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        #transforms.CenterCrop(448),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

def get_val_image_transform():
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        #transforms.Resize(448, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        #transforms.CenterCrop(448),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class MultimodalDataset(Dataset):
    def __init__(self, df, tab_preproc, image_cols, y_col, train=True, root_dir = None):
        self.df = df.reset_index(drop=True)
        self.tab_preproc = tab_preproc
        self.image_cols = image_cols
        self.y_col = y_col
        self.train = train
        self.image_transform = get_train_image_transform() if train else get_val_image_transform()
        self.root_dir = root_dir
    
    # Auto-detect base folder if not provided
        if self.root_dir is None:
            # Look for the first existing image path
            for col in self.image_cols:
                for path in self.df[col].dropna():
                    if isinstance(path, str) and os.path.exists(path):
                        self.root_dir = os.path.dirname(os.path.abspath(path))
                        print(f"[INFO] Auto-detected image root directory: {self.root_dir}")
                        break
                if self.root_dir is not None:
                    break
            if self.root_dir is None:
                print("[WARN] Could not auto-detect image root; using paths as-is.")

    def __len__(self):
        return len(self.df)
    
    def _resolve_path(self, img_path):
        if not isinstance(img_path, str) or pd.isna(img_path):
            return None
        if self.root_dir and not os.path.isabs(img_path):
            candidate = os.path.join(self.root_dir, img_path)
        else:
            candidate = img_path
        if not os.path.exists(candidate):
            # Uncomment next line for debugging if needed
            # print(f"[WARN] Missing image file: {candidate}")
            return None
        return candidate

    def _load_image(self, img_path):
        #if pd.isna(img_path) or not isinstance(img_path, str) or not os.path.isabs(img_path):
        if pd.isna(img_path) or not isinstance(img_path, str):
            return None
        if self.root_dir is not None and not os.path.isabs(img_path):
            img_path = os.path.join(self.root_dir, img_path)
        if not os.path.exists(img_path):
            print(f"[WARN] Missing image path: {img_path}")
            return None
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Could not open {img_path}: {e}")
            return None

        img = Image.open(img_path).convert("RGB")
        return self.image_transform(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        num, cat = self.tab_preproc.transform(row.to_frame().T)
        num = num[0]
        cat = cat[0]
        images = []
        image_valid_num = 0
        for col in self.image_cols:
            img = self._load_image(row[col])
            if img is not None:
                images.append(img)
                image_valid_num += 1
            else:
                images.append(torch.zeros(3, 224, 224))
                #images.append(torch.zeros(3, 448, 448))
        images = torch.stack(images, dim=0)
        label = int(row[self.y_col])
        return {
            "numerical": torch.tensor(num, dtype=torch.float32),
            "categorical": torch.tensor(cat, dtype=torch.long),
            "images": images,
            "image_valid_num": image_valid_num,
            "label": label,
        }

class TabularDataset(torch.utils.data.Dataset):
    """
    Dataset for FTTransformer-only experiments (tabular data only).
    Uses the same TabularPreprocessor as the multimodal dataset.
    """
    def __init__(self, df, tab_preproc, y_col):
        self.df = df.reset_index(drop=True)
        self.tab_preproc = tab_preproc
        self.y_col = y_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        num, cat = self.tab_preproc.transform(row.to_frame().T)
        num = torch.tensor(num[0], dtype=torch.float32)
        cat = torch.tensor(cat[0], dtype=torch.long)
        label = int(row[self.y_col])
        return {
            "numerical": num,
            "categorical": cat,
            "label": label,
        }
