from torch.utils.data import DataLoader

from data.datasets import Default_dataset


DATASET_TASK_CLASS = {
    "TSPN_SUDA_DEMO": Default_dataset,
}


def get_data(args):
    dataset_class = DATASET_TASK_CLASS[args.dataset_task]
    pin_memory = bool(getattr(args, "pin_memory", False))
    num_workers = int(getattr(args, "num_workers", 0))

    train_dataset = dataset_class(args, flag="train")
    val_dataset = dataset_class(args, flag="val")
    test_dataset = dataset_class(args, flag="test")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader
