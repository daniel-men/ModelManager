import glob
import json
import os
import pickle

from tensorflow.keras.utils import Sequence

from python_utility_functions import mkdir, create_timestamp

from .utils import (ConfigurationAlreadyExistsError,
                    deserialize_function, prepare_for_json, serialize_function)


class ModelManager:
    def __init__(self, log_dir: str, model=None, save_history : bool = False, save_weights : bool = False, save_model: bool = False):
        """
        Args:
            log_dir (str): Path to the parent directory where all training runs will be stored.
            model ([type], optional): Keras Model. Defaults to None.
            save_history (bool, optional): [description]. Defaults to False.
            save_weights (bool, optional): [description]. Defaults to False.
            save_model (bool, optional): [description]. Defaults to False.
        """
        self._log_dir = log_dir
        self._model = model
        self.key_params = {}
        self._save_history = save_history
        self._save_weights = save_weights
        self._save_model = save_model
        self.overwrite = False
        self._save_path = None
        self._timestamp = None
        self.fit_has_been_run = False

        mkdir(self._log_dir)

    def new_timestamp(self) -> str:
        """Creates a new timestamp string

        Returns:
            [String]: [timestamp]
        """
        self._timestamp = create_timestamp()
        return self._timestamp


    def _fit(self, kwargs) -> None:
        """[summary]

        Args:
            kwargs ([type]): [description]
        """
        if self.fit_has_been_run:
            self.new_timestamp()
        
        self.get_compile_params()
        self.get_fit_params(kwargs)

        prepared_params = prepare_for_json(self.key_params)
        self.check_for_existing_runs(json.dumps(prepared_params))

        history = self.model.fit(**kwargs)

        self.save_path
        self.log()
 
        if self.save_history:
            self.save_history_pickle(history)

        if self._save_model:
            self.model.save(os.path.join(self.save_path, "model.h5"))

        if self._save_weights:
            self.model.save_weights(os.path.join(self.save_path, "weights.h5"))

        self.fit_has_been_run = True

    def fit(self, **kwargs):
        """Wrapper for Sequential.fit()
        """
        self._fit(kwargs)

    def get_compile_params(self):
        """Read key parameters from the model instance.
        """
        optimizer_config = self.model.optimizer.get_config()
        self.key_params["optimizer"] = optimizer_config
        self.key_params["optimizer"]["learning_rate"] = self.model.optimizer.lr.numpy()
        if 'name' not in optimizer_config.keys():
            opt_name = str(self.model.optimizer.__class__).split('.')[-1] \
                .replace('\'', '').replace('>', '')
            self.key_params["optimizer"]["name"] = opt_name
        if callable(self.model.loss):
            self.key_params['loss'] = serialize_function(self.model.loss)
        else:
            self.key_params["loss"] = self.model.loss

    @property
    def timestamp(self) -> str:
        """Create a new timestamp if needed or return the current timestamp.

        Returns:
            str: [description]
        """
        if self._timestamp is None:
            self._timestamp = self.new_timestamp()
        return self._timestamp

    @property
    def save_path(self) -> str:
        if self._save_path is None or self.timestamp not in self._save_path:
            self._save_path = os.path.join(self.log_dir, self.timestamp)
        
        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)
        
        return self._save_path

    @property
    def callback_log_path(self) -> str:
        return os.path.join(self.save_path, "logs")

    @property
    def save_history(self) -> bool:
        return self._save_history

    @save_history.setter
    def save_history(self, save_history):
        self._save_history = save_history

    @property
    def description(self):
        if "description" in self.key_params:
            return self.key_params["description"]
        else:
            return None

    @description.setter
    def description(self, description):
        self.key_params["description"] = description

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def log_dir(self):
        return self._log_dir

    @log_dir.setter
    def log_dir(self, log_dir):
        self._log_dir = log_dir

    def log(self):
        """Save parameters as JSON
        """
        with open(os.path.join(self.save_path, "config.json"), 'w') as json_file:
            json.dump(self.key_params, json_file)

    def get_fit_params(self, kwargs):
        """Extract parameters supplied during fit() or fit_generator() call

        Arguments:
            kwargs {[type]} -- [description]
        """
        
        _kwargs = kwargs.copy()

        self.key_params["epochs"] = _kwargs["epochs"]

        #self.key_params['model_summary'] = self.model.summary()

        if "batch_size" in _kwargs:
            self.key_params["batch_size"] = _kwargs["batch_size"]
        else:
            self.key_params["batch_size"] = 32

        if "callbacks" in _kwargs:
            self.key_params["callbacks"] = [serialize_function(f) for f in _kwargs["callbacks"]]

        if "validation_data" in _kwargs:
            if not isinstance(_kwargs['validation_data'], Sequence):
                self.validation_data = _kwargs["validation_data"]
                validation_data_path = os.path.join(self.save_path, "validation_data.p")
                with open(validation_data_path, 'wb') as pickle_file:
                    pickle.dump(_kwargs["validation_data"], pickle_file)
                self.key_params["validation_data"] = validation_data_path
            else:
                _kwargs.pop('validation_data')
        
        opt_params = ["class_weight", "sample_weight"]
        
        for param in kwargs:
            if param not in self.key_params and param not in ["x", "y", "generator"]:
                self.key_params[param] = kwargs[param]


    def check_for_existing_runs(self, json_conf):
        """Check if already existing runs with the same configuration were logged

        Arguments:
            json_conf {[type]} -- Current Parameters as JSON object

        Raises:
            ConfigurationAlreadyExistsError: Raised if the current parameter configuration had already been run before
        """
        for folder in glob.glob(os.path.join(self.log_dir, "*")):
            other_config_path = os.path.join(folder, "config.json")
            if os.path.isdir(folder) and os.path.isfile(other_config_path):
                with open(other_config_path) as conf_file:
                    existing_conf = json.load(conf_file)
                    existing_conf = json.dumps(existing_conf)
                    if existing_conf == json_conf and not self.overwrite:
                        raise ConfigurationAlreadyExistsError("Configuraiton already exists in {}".format(folder))
    
    def save_history_pickle(self, history):
        with open(os.path.join(self.save_path, "history.p"), 'wb') as pickle_file:
            pickle.dump(history.history, pickle_file)

