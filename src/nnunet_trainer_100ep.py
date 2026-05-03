"""
nnunet_trainer_100ep.py — Custom nnU-Net v2 trainer limited to 100 epochs.

The paper (Joshi et al., 2025) trains for exactly 100 epochs.
nnU-Net v2 default is 1000 epochs, so we subclass nnUNetTrainer
and override num_epochs.

Usage in training:
    nnUNetv2_train 501 3d_fullres 0 -tr nnUNetTrainer_100epochs
"""

try:
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    class nnUNetTrainer_100epochs(nnUNetTrainer):
        """
        nnU-Net trainer that stops at 100 epochs (paper setting).
        SGD with momentum=0.99 and Nesterov acceleration, LR=0.01.
        All other hyperparameters are inherited from nnUNetTrainer.
        """

        def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
            super().__init__(plans, configuration, fold, dataset_json, device)
            # Paper: 100 epochs
            self.num_epochs = 100
            # Paper: SGD LR=0.01, momentum=0.99, Nesterov
            self.initial_lr = 0.01
            self.weight_decay = 3e-5  # nnU-Net default

        def configure_optimizers(self):
            """
            SGD with momentum=0.99, Nesterov acceleration, LR=0.01.
            Matches paper Section 3.3.
            """
            import torch
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=self.initial_lr,
                momentum=0.99,
                nesterov=True,
                weight_decay=self.weight_decay,
            )
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=self.num_epochs,
                power=0.9,
            )
            return optimizer, lr_scheduler

except ImportError:
    # nnunetv2 not installed yet — placeholder class for import safety
    class nnUNetTrainer_100epochs:  # type: ignore
        """Placeholder — install nnunetv2 first."""
        num_epochs = 100
