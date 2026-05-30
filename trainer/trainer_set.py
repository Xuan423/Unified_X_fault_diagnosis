from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelPruning
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl

from data.data_provider import get_data
from model.Signal_processing import WaveFilters

from .utils import ModelParametersLoggingCallback


def trainer_set(args, path):
    callback_list = call_backs(args, path)
    logger = CSVLogger(path, name="logs")
    accelerator = "cpu" if args.device == "cpu" else "auto"
    devices = 1 if args.device == "cpu" else int(getattr(args, "gpus", 1))

    trainer = pl.Trainer(
        callbacks=callback_list,
        accelerator=accelerator,
        max_epochs=args.num_epochs,
        devices=devices,
        logger=logger,
        log_every_n_steps=1,
    )

    train_dataloader, val_dataloader, test_dataloader = get_data(args)
    return trainer, train_dataloader, val_dataloader, test_dataloader


def call_backs(args, path):
    monitor = getattr(args, "monitor", "val_loss")
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        filename="model-{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}",
        save_top_k=1,
        save_last=True,
        mode="min",
        dirpath=path,
    )
    callback_list = [checkpoint_callback]

    prune_callback = Prune_callback(args)
    if prune_callback is not None:
        callback_list.append(prune_callback)

    if bool(getattr(args, "log_parameters", False)):
        callback_list.append(ModelParametersLoggingCallback(path=path, module_type=WaveFilters))

    callback_list.append(create_early_stopping_callback(args))
    return callback_list


def Prune_callback(args):
    pruning = getattr(args, "pruning", None)
    if pruning in (None, False, "None", "none", "null", "Null"):
        return None

    def compute_amount(epoch):
        if epoch == args.num_epochs // 4:
            return pruning[0]
        if epoch == args.num_epochs // 2:
            return pruning[1]
        if 3 * args.num_epochs // 4 < epoch:
            return pruning[2]
        return 0

    if isinstance(pruning, (int, float)):
        return ModelPruning("l1_unstructured", parameter_names=["weight"], amount=pruning)
    if isinstance(pruning, list):
        return ModelPruning("l1_unstructured", parameter_names=["weight"], amount=compute_amount)
    return None


def create_early_stopping_callback(args):
    return EarlyStopping(
        monitor=getattr(args, "monitor", "val_loss"),
        min_delta=0.0,
        patience=int(getattr(args, "patience", 20)),
        verbose=True,
        mode="min",
        check_finite=True,
        check_on_train_epoch_end=False,
    )
