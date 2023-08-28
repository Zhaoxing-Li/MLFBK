from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from utils import pid_collate_fn
# datasets
from Dataloaders.ednet_pid_loader import EDNET_PID_SPLIT
from Dataloaders.assistments09_pid_loader import ASSISTMENTS09_PID_SPLIT
from Dataloaders.assistments12_pid_loader import ASSISTMENTS12_PID_SPLIT
from Dataloaders.algebra06_pid_loader import ALGEBRA06_PID_SPLIT
from Dataloaders.toy_example_pid_loader import TOY_PID_SPLIT


def get_loaders(config):
    # 1. choose loader
    train_dataset = None
    valid_dataset = None
    test_dataset = None
    if config.dataset_name == "ednet":
        train_dataset, test_dataset, valid_dataset, fe = EDNET_PID_SPLIT(config.max_seq_len)
        num_q = fe.num_q
        num_r = fe.num_r
        num_pid = fe.num_pid
        num_ap = fe.num_ap
        num_pd = fe.num_pd
        num_sm = fe.num_sm
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "assistments09":
        train_dataset, test_dataset, valid_dataset, fe = ASSISTMENTS09_PID_SPLIT(config.max_seq_len)
        num_q = fe.num_q
        num_r = fe.num_r
        num_pid = fe.num_pid
        num_ap = fe.num_ap
        num_pd = fe.num_pd
        num_sm = fe.num_sm
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "assistments12":
        train_dataset, test_dataset, valid_dataset, fe = ASSISTMENTS12_PID_SPLIT(config.max_seq_len)
        num_q = fe.num_q
        num_r = fe.num_r
        num_pid = fe.num_pid
        num_ap = fe.num_ap
        num_pd = fe.num_pd
        num_sm = fe.num_sm
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "algebra06":
        train_dataset, test_dataset, valid_dataset, fe = ALGEBRA06_PID_SPLIT(config.max_seq_len)
        num_q = fe.num_q
        num_r = fe.num_r
        num_pid = fe.num_pid
        num_ap = fe.num_ap
        num_pd = fe.num_pd
        num_sm = fe.num_sm
        num_diff = None
        collate = pid_collate_fn
    elif config.dataset_name == "toy":
        train_dataset, test_dataset, valid_dataset, fe = TOY_PID_SPLIT(config.max_seq_len)
        num_q = fe.num_q
        num_r = fe.num_r
        num_pid = fe.num_pid
        num_ap = fe.num_ap
        num_pd = fe.num_pd
        num_sm = fe.num_sm
        num_diff = None
        collate = pid_collate_fn

    # 3. get DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size = config.batch_size,
        shuffle = True, # train_loader use shuffle
        collate_fn = collate
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size = config.batch_size,
        shuffle = False, # valid_loader don't use shuffle
        collate_fn = collate
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = config.batch_size,
        shuffle = False, # test_loader don't use shuffle
        collate_fn = collate
    )

    return train_loader, valid_loader, test_loader, num_q, num_r, num_pid, num_diff, \
        num_ap, num_pd, num_sm