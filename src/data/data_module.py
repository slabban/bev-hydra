from typing import Dict, Optional, Any

from torch.utils.data import DataLoader, Dataset

from lightning import LightningDataModule

from src.data.components.data import FuturePredictionDataset

from omegaconf import DictConfig



class DampDataModule(LightningDataModule):
    """`LightningDataModule` for the Furture Prediction dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        dataset_parameters: DictConfig = None,
        common: DictConfig = None,
    ) -> None:
        """
        Initializes a new instance of the class.

        Args:
            data_dir (str, optional): The directory where the data is located. Defaults to "data/".
            dataset_version (str, optional): The version of the dataset. Defaults to "trainval".
            batch_size (int, optional): The number of samples per batch. Defaults to 1.
            ignore_index (int, optional): The index to ignore in the dataset. Defaults to 255.
            num_workers (int, optional): The number of workers for data loading. Defaults to 5.
            pin_memory (bool, optional): Whether to pin memory for data loading. Defaults to True.

        Returns:
            None

        This function initializes the class with the provided parameters. It sets the `version`, `dataroot`, `batch_size`, `ignore_index`, `num_workers`, and `pin_memory` attributes. It also saves the initialization parameters using the `save_hyperparameters` method. The `data_train`, `data_val`, and `data_test` attributes are initialized as optional `Dataset` objects. The `batch_size_per_device` attribute is set to the provided `batch_size`.
        """

        super().__init__()

        self.dataset_parameters = dataset_parameters
        self.common = common

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        self.data_train = FuturePredictionDataset(
            data_root=self.dataset_parameters.data_root, version=self.dataset_parameters.version,
            ignore_index=self.dataset_parameters.ignore_index,
            batch_size=self.dataset_parameters.batch_size,
            is_train=True, filter_invisible_vehicles=self.filter_invisible_vehicles
        )
        self.data_val = FuturePredictionDataset(
            data_root=self.dataroot, version=self.version,
            ignore_index=self.ignore_index,
            batch_size=self.batch_size,
            is_train=False, filter_invisible_vehicles=self.filter_invisible_vehicles
        )


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.dataset_parameters.batch_size,
            num_workers=self.dataset_parameters.num_workers,
            pin_memory=self.dataset_parameters.pin_memory,
            shuffle=True, drop_last= True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            dataset=self.data_train,
            batch_size=self.dataset_parameters.batch_size,
            num_workers=self.dataset_parameters.num_workers,
            pin_memory=self.dataset_parameters.pin_memory,
            shuffle=False, drop_last= False
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """


if __name__ == "__main__":
    _ = DampDataModule()
