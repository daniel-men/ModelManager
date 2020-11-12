import pickle
import json
import datetime
import glob
import os

from .utils import ConfigurationAlreadyExistsError, serialize_function, deserialize_function

class ModelManager:
    def __init__(self, log_dir, model=None, save_history=False, save_weights=False, save_model=False):
        self._log_dir = log_dir
        self._model = model
        self.key_params = {}
        self._save_history = save_history
        self._save_weights = save_weights
        self._save_model = save_model
        self.timestamp = None
        self.overwrite = False

        if not os.path.exists(self._log_dir):
            os.mkdir(self._log_dir)

    def create_timestamp(self):
        self.timestamp = "{}".format(datetime.datetime.now()).replace(" ", "_").replace(":", "_").replace(".", "_")
        return self.timestamp

    def _fit(self, kwargs, gen=False):
        self.create_timestamp()
        self.get_compile_params()
        self.get_fit_params(kwargs)

        self.check_for_existing_runs(json.dumps(self.key_params))
        if gen:
            history = self.model.fit_generator(**kwargs)
        else:
            history = self.model.fit(**kwargs)
 
        self.log()

        if self.save_history:
            self.save_history_pickle(history)

        if self._save_model:
            self.model.save(os.path.join(self.log_dir, self.timestamp, "model.h5"))

        if self._save_weights:
            self.model.save_weights(os.path.join(self.log_dir, self.timestamp, "weights.h5"))

    def fit(self, **kwargs):
        """Wrapper for Sequential.fit()
        """
        self._fit(kwargs)

    def fit_generator(self, **kwargs):
        """Wrapper for Sequential.fit_generator()
        """
        self._fit(kwargs, gen=True)

    def get_compile_params(self):
        optimizer_config = self.model.optimizer.get_config()
        self.key_params["optimizer"] = optimizer_config
        self.key_params["optimizer"]["learning_rate"] = str(self.model.optimizer.lr.numpy())
        if 'name' not in optimizer_config.keys():
            opt_name = str(self.model.optimizer.__class__).split('.')[-1] \
                .replace('\'', '').replace('>', '')
            self.key_params["optimizer"]["name"] = opt_name
        if callable(self.model.loss):
            self.key_params['loss'] = serialize_function(self.model.loss)
        else:
            self.key_params["loss"] = self.model.loss

    @property
    def save_history(self):
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
        os.mkdir(os.path.join(self.log_dir, self.timestamp))
        with open(os.path.join(self.log_dir, self.timestamp, "config.json"), 'w') as json_file:
            json.dump(self.key_params, json_file)

    def get_fit_params(self, kwargs):
        """Extract parameters supplied during fit() or fit_generator() call

        Arguments:
            kwargs {[type]} -- [description]
        """
        self.key_params["epochs"] = kwargs["epochs"]

        if "batch_size" in kwargs:
            self.key_params["batch_size"] = kwargs["batch_size"]
        else:
            self.key_params["batch_size"] = 32

        if "callbacks" in kwargs:
            self.key_params["callbacks"] = serialize_function(kwargs["callbacks"])
        
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
            if os.path.isdir(folder):
                with open(os.path.join(folder, "config.json")) as conf_file:
                    existing_conf = json.load(conf_file)
                    existing_conf = json.dumps(existing_conf)
                    if existing_conf == json_conf and not self.overwrite:
                        raise ConfigurationAlreadyExistsError("Configuraiton already exists in {}".format(folder))
    
    def save_history_pickle(self, history):
        with open(os.path.join(self.log_dir, self.timestamp, "history.p"), 'wb') as pickle_file:
            pickle.dump(history.history, pickle_file)
