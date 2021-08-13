import os
import wandb

from lib.log.BaseLogger import BaseLogger


class WandbLogger(BaseLogger):

    def __init__(
            self,
            wandb_project_name: str,
            entity: str,
            api_key: str,
            config: dict):
        super().__init__(config=config)

        os.environ["WANDB_API_KEY"] = api_key
        wandb.login()

        wandb.config.update(self.config)
        self.run: wandb = wandb.init(project=wandb_project_name, entity=entity)

    def log(self, data: dict):
        wandb.log({
            "score": self.scores,
            "avg. score": self.scores_window
        })

    def dispose(self):
        self.run.finish()
