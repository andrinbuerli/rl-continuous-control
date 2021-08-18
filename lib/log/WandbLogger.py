import os
import wandb

from lib.log.BaseLogger import BaseLogger


class WandbLogger(BaseLogger):

    def __init__(
            self,
            wandb_project_name: str,
            api_key: str,
            config: dict):
        super().__init__(config=config)

        os.environ["WANDB_API_KEY"] = api_key
        wandb.login()

        self.run: wandb = wandb.init(project=wandb_project_name)
        wandb.config.update(self.config)

    def log(self, data: dict):
        wandb.log(data)

    def dispose(self):
        self.run.finish()


class WandbSweepLogger(BaseLogger):

    def __init__(
            self,
            config: dict):
        super().__init__(config=config)

    def log(self, data: dict):
        wandb.log(data)

    def dispose(self):
        pass