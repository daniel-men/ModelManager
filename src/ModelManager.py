import pickle
import json
import datetime
import glob
import os

from utils import ConfigurationAlreadyExistsError

class ModelManager:
    def __init__(self, log_dir, model=None, save_history=False, save_weights=False, save_model=False):
        self._log_dir = log_dir
        self._model = model
        self.key_params = {}
        self._save_history = save_history
        self._save_weights = save_weights
        self._save_model = save_model
        self.timestamp = None

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

    def fit(self, **kwargs):
        self._fit(kwargs)

    def fit_generator(self, **kwargs):
        self._fit(kwargs, gen=True)

    def get_compile_params(self):
        optimizer_config = self.model.optimizer.get_config()
        self.key_params["optimizer"] = optimizer_config
        self.key_params["optimizer"]["learning_rate"] = float(str(self.model.optimizer.lr.numpy()))
        if 'name' not in optimizer_config.keys():
            opt_name = str(self.model.optimizer.__class__).split('.')[-1] \
                .replace('\'', '').replace('>', '')
            self.key_params["optimizer"]["name"] = opt_name
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
        os.mkdir(os.path.join(self.log_dir, self.timestamp))
        with open(os.path.join(self.log_dir, self.timestamp, "config.json"), 'w') as json_file:
            json.dump(self.key_params, json_file)

    def get_fit_params(self, kwargs):
        if "batch_size" in kwargs:
            self.key_params["batch_size"] = kwargs["batch_size"]
        else:
            self.key_params["batch_size"] = 32

        self.key_params["epochs"] = kwargs["epochs"]
        
        opt_params = ["class_weight", "sample_weight"]
        
        for param in opt_params:
            if param in kwargs:
                self.key_params[param] = kwargs[param]


    def check_for_existing_runs(self, json_conf):
        for folder in glob.glob(os.path.join(self.log_dir, "*")):
            if os.path.isdir(folder):
                with open(os.path.join(folder, "config.json")) as conf_file:
                    existing_conf = json.load(conf_file)
                    existing_conf = json.dumps(existing_conf)
                    if existing_conf == json_conf:
                        raise ConfigurationAlreadyExistsError("Configuraiton already exists in {}".format(folder))
    
    def save_history_pickle(self, history):
        with open(os.path.join(self.log_dir, self.timestamp, "history.p"), 'wb') as pickle_file:
            pickle.dump(history.history, pickle_file)

