import os
import torch
from batchgenerators.utilities.file_and_folder_operations import save_json, join, isfile, load_json
import numpy as np
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.crossval_split import generate_crossval_split
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_XPercentSplit(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, device)

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.json file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.json file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
        
        splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
        dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                        identifiers=None,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        # if the split file does not exist we need to create it
        if not isfile(splits_file):
            self.print_to_log_file("Creating new 5-fold cross-validation split...")
            all_keys_sorted = list(np.sort(list(dataset.identifiers)))
            splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
            save_json(splits, splits_file)

        else:
            self.print_to_log_file("Using splits from existing split file:", splits_file)
            splits = load_json(splits_file)
            self.print_to_log_file(f"The split file contains {len(splits)} splits.")

        self.print_to_log_file("Desired fold for training: %d" % self.fold)        
        self.print_to_log_file("INFO: You requested fold %d for training but splits "
                            "contain only %d folds. I am now creating a "
                            "random (but seeded) %s split!" % (self.fold, len(splits), self.custom_train_split))
        # if we request a fold that is not in the split file, create a random 80:20 split
        rnd = np.random.RandomState(seed=12345 + self.fold)
        keys = np.sort(list(dataset.identifiers))
        idx_tr = rnd.choice(len(keys), int(len(keys) * self.custom_train_split), replace=False)
        idx_val = [i for i in range(len(keys)) if i not in idx_tr]
        tr_keys = [keys[i] for i in idx_tr]
        val_keys = [keys[i] for i in idx_val]
        self.print_to_log_file("This random %s split has %d training and %d validation cases."
                                    % (self.custom_train_split, len(tr_keys), len(val_keys)))
        if any([i in val_keys for i in tr_keys]):
            self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                    'splits.json or ignore if this is intentional.')
        return tr_keys, val_keys


class nnUNetTrainer_5PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.05
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_10PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.1
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_20PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.2
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_30PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.3
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_40PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.4
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_50PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.5
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_60PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.6
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_70PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.7
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_80PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.8
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)

class nnUNetTrainer_90PercentSplit(nnUNetTrainer_XPercentSplit):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.custom_train_split = 0.9
        self.num_iterations_per_epoch = int(self.num_iterations_per_epoch * self.custom_train_split)
