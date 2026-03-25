import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold
from .data import MultimodalDataset, TabularPreprocessor

class ObstetricsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.batch_size

    def setup(self, stage=None, fold_idx=None, k_folds=5):
        df = pd.read_csv(self.cfg.train_csv_path)
        df = df.dropna(subset=[self.cfg.y_col])
        if "processo" in df.columns:
            df = df.drop(columns=["processo"])

        self.tab_preproc = TabularPreprocessor(
            self.cfg.num_cols,
            self.cfg.cat_cols
        )
    
        self.tab_preproc.fit(df)

        # If fold_idx is None → standard train/val split
        if fold_idx is None:
            train_df, val_df = train_test_split(
                df,
                test_size=0.25,
                stratify=df[self.cfg.y_col],
                random_state=self.cfg.split_seed
            )   
        else:
            # K-Fold splitting
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.cfg.split_seed)
            splits = list(skf.split(df, df[self.cfg.y_col]))
            train_idx, val_idx = splits[fold_idx]
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

        # create datasets
        self.train_ds = MultimodalDataset(
            train_df,
            self.tab_preproc,
            self.cfg.image_cols,
            self.cfg.y_col,
            train=True,
            root_dir=self.cfg.train_image_path
        )
        self.val_ds = MultimodalDataset(
            val_df,
            self.tab_preproc,
            self.cfg.image_cols,
            self.cfg.y_col,
            train=False,
            root_dir=self.cfg.train_image_path
        )

        self.num_classes = len(df[self.cfg.y_col].unique())
        self.num_tabular_features = len(self.cfg.num_cols)
        self.class_names = ["Cesarean", "Vaginal"]

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        df = pd.read_csv(self.cfg.test_csv_path)

        ds = MultimodalDataset(
            df,
            self.tab_preproc,
            self.cfg.image_cols,
            self.cfg.y_col,
            train=False,
            root_dir=self.cfg.test_image_path
        )

        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4)