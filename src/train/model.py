# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Code adapted from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
# Code adapted from: https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029
# Special thanks to A. Badilla

# ----------------------------- Imports ---------------------------------------
import torch
from torch.optim import AdamW

import pytorch_lightning

from transformers import BertModel, get_linear_schedule_with_warmup

import config.configuration as config

# ----------------------------- Class definition ------------------------------
# Wrapper class for the Model with its head
class Model(pytorch_lightning.LightningModule):
    # Constructor
    def __init__(self, n_classes: int, n_training_steps=None,
                 n_warmup_steps=None):
        super().__init__()
        self.model = BertModel.from_pretrained(config.MODEL_NAME,
                                               return_dict=True)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size,
                                          n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.loss_function = config.LOSS_FUNCTION

        self.outputs = {}
        
        self.train_metrics = config.METRICS.clone(prefix="train/")
        self.val_metrics = config.METRICS.clone(prefix="val/")
        self.test_metrics = config.METRICS.clone(prefix="test/")
        self.test_metrics_outputs = []

    # Override method to execute the model using the tokens
    def forward(self, input_ids, attention_mask, labels=None):
        # Pass the data to the model
        output = self.model(input_ids, attention_mask=attention_mask)
        # Pass the result to the classifier (head)
        output = self.classifier(output.pooler_output)
        # Pass the result to the activation function
        output = torch.sigmoid(output)

        # Calculate the loss using the loss function
        loss = 0
        if labels is not None:
            loss = self.loss_function(output, labels)
        return loss, output

    # Override method to call every change in weights in
    # the training
    def training_step(self, batch, batch_idx):
        # The batch idx will not be used
        _ = batch_idx

        # Get the token data from the batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # Execute the model
        loss, outputs = self.forward(input_ids, attention_mask, labels)

        # Update the metrics based on the results
        self.train_metrics.update(outputs, labels)

        # Update the progress bar indicating the loss
        self.log("train_loss", loss, prog_bar=True, logger=True)

        # Store the outputs (since they are still needed) and return them
        self.outputs = {"loss": loss, "predictions": outputs, "labels": labels}
        return {"loss": loss, "predictions": outputs, "labels": labels}

    # Override method to call every change in weights in
    # the validation
    def validation_step(self, batch, batch_idx):
        # The batch idx will not be used
        _ = batch_idx

        # Get the token data from the batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        # Execute the model
        loss, outputs = self.forward(input_ids, attention_mask, labels)

        # Update the metrics based on the results
        self.val_metrics.update(outputs, labels)

        # Update the progress bar indicating the loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # Return the loss
        return loss

    # Override method to call every change in weights in
    # the testing
    def test_step(self, batch, batch_idx):
        # The batch idx will not be used
        _ = batch_idx

        # Get the token data from the batch
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Execute the model
        loss, outputs = self.forward(input_ids, attention_mask, labels)

        # Update the metrics based on the results
        self.test_metrics.update(outputs, labels)

        # Update the progress bar indicating the loss
        self.log("test_loss", loss, prog_bar=True, logger=True)
        # Return the loss
        return loss

    # Override method to call every end of a training epoch
    def on_train_epoch_end(self):
        # Log the metrics of the epoch
        self.log_dict(self.train_metrics.compute(), on_step=False)
        # Reset the metrics of the epoch
        self.train_metrics.reset()
    
    # Override method to call every end of a validation epoch
    def on_validation_epoch_end(self):
        # Log the metrics of the epoch
        self.log_dict(self.val_metrics.compute(), on_step=False)
        # Reset the metrics of the epoch
        self.val_metrics.reset()
    
    # Override method to call every end of a testing epoch
    def on_test_epoch_end(self):
        # Log the metrics of the epoch
        test_compute = self.test_metrics.compute()
        self.log_dict(test_compute, on_step=False)

        # Store the metrics of the epoch
        self.test_metrics_outputs.append(test_compute)

        # Reset the metrics of the epoch
        self.test_metrics.reset()

    # Override method to configure the optimize algorithm
    def configure_optimizers(self):
        # Use AdamW as the optimizer with the parameters and
        # the learning rate
        optimizer = AdamW(self.parameters(), lr=config.LEARNING_RATE)

        # Decide based on the optimizer when to pass from warmup to 
        # training (which can have an early stopping)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )
        # Return the results in a dictionary
        return dict(optimizer=optimizer,
                    lr_scheduler=dict(scheduler=scheduler,
                                      interval="step"))