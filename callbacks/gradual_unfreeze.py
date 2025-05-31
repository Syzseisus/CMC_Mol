import pytorch_lightning as pl


class GradualUnfreeze(pl.Callback):
    """
    epoch_freeze = 3  → 처음 3 epoch 동안 backbone 파라미터 freeze
    unfreeze_steps = 2 → 이후 2 epoch마다 다음 block 1개씩 unfreeze
    backbone must implement .freeze_block(k) / .num_blocks
    """

    def __init__(self, epoch_freeze=3, unfreeze_steps=2):
        self.epoch_freeze = epoch_freeze
        self.unfreeze_steps = unfreeze_steps

    def on_train_start(self, trainer, pl_module):
        pl_module.model.freeze_block(range(pl_module.model.num_blocks))

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.epoch_freeze:
            return
        n_to_unfreeze = (epoch - self.epoch_freeze) // self.unfreeze_steps + 1
        pl_module.model.freeze_block(
            range(pl_module.model.num_blocks - n_to_unfreeze, pl_module.model.num_blocks),
            freeze=False,
        )
