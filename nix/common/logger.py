from omegaconf import DictConfig, OmegaConf
import wandb


class Logger:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the Logger with Weights & Biases configuration.

        Args:
            config (DictConfig): Configuration containing hyperparameters and settings.
        """
        # Initialize Weights & Biases
        wandb.init(
            reinit=True,
            project=cfg.project,  # Project name in W&B
            name=cfg.run_name,      # Run name in W&B
            config=OmegaConf.to_container(cfg, resolve=True)               # Save the configuration in W&B
        )
        self.step = 0

    def log_metrics(self, metrics: dict) -> None:
        """Log metrics to Weights & Biases."""
        if metrics is not None:  # Check if metrics are provided
            self.step += 1

            # Log metrics to W&B
            wandb.log(metrics, step=self.step)