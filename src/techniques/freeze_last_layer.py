"""Fine-tune the last convolutional layer and the classifier."""

from torch import nn
from torch.utils.data import DataLoader

from src.config import config
from src.data.dataset import get_class_weights
from src.models.base import BaseImageClassifier
from src.techniques.base_technique import FineTuningTechnique
from src.train.loss import GoogLeNetLoss
from src.train.train import Trainer


class FreezeLastLayer(FineTuningTechnique):
    """Train only the last convolutional layer and the classifier."""

    def prepare_model(self) -> BaseImageClassifier:
        """Freezes all the layers of the model except the ones that will be trained.

        Returns:
            Model with the corresponding freezed parameters.
        """

        # TODO

        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        convolutions = [
            module
            for module in self.base_model.modules()
            if isinstance(module, nn.Conv2d)
        ]
        for parameter in convolutions[-1].parameters():
            parameter.requires_grad = True
        for parameter in self.base_model.classifier.parameters():
            parameter.requires_grad = True

        # this has to run before creating the Trainer, since its optimizer
        # only picks up the parameters that have requires_grad = True at that point

        return self.base_model

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, epochs: int
    ) -> BaseImageClassifier:
        """Trains and returns a model.

        Args:
            train_loader: Train loader.
            val_loader: Validation loader.
            epochs: Number of epochs to train.

        Returns:
            Trained model.
        """

        weights_path = f"{config.paths.checkpoints_dir}googlenet_finetune.pt"
        figure_path = f"{config.paths.results_dir}googlenet_finetune_training.png"

        # TODO: create a weighted criterion and pass it to `Trainer`

        prepared_model = self.prepare_model()

        loss = GoogLeNetLoss(class_weights=get_class_weights(train_loader))

        trainer = Trainer(model=prepared_model, criterion=loss)

        trained_model = trainer.fit(
            train_loader=train_loader,
            validation_loader=val_loader,
            epochs=epochs,
            path_weights=weights_path,
            path_figure=figure_path,
        )

        return trained_model
