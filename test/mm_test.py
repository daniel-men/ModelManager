import unittest
import os
import shutil
import json
import glob
import pickle

import dill
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import numpy as np

from src.ModelManager import ModelManager, ConfigurationAlreadyExistsError
from src.utils import deserialize_function


class TestModelManagerSet(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.abspath(os.path.join("..", "test_dir"))
        os.mkdir(self.test_path)
        self.mm = ModelManager(self.test_path)
        return super().setUp()

    def tearDown(self):
        shutil.rmtree(self.test_path)
        return super().tearDown()

    def test_history(self):
        self.assertFalse(self.mm.save_history)
        self.mm.save_history = True
        self.assertTrue(self.mm.save_history)

    def test_set_model(self):
        self.assertIsNone(self.mm.model)
        self.mm.model = Sequential()
        self.assertIsNotNone(self.mm.model)

    def test_set_description(self):
        self.assertIsNone(self.mm.description)

        self.mm.description = "Test Run Model 123"
        self.assertEqual(self.mm.description, "Test Run Model 123")


class TestModelManagerModelFunctions(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.abspath(os.path.join("..", "test_dir"))
        os.mkdir(self.test_path)
        self.mm = ModelManager(self.test_path)
        self.simple_model = simple_model()
        return super().setUp()

    def tearDown(self):
        shutil.rmtree(self.test_path)
        return super().tearDown()

    def test_model_fit(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
        self.mm.description = "Test on numeric data, two hidden layers"

        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))
        
        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
            self.assertEqual(json_config["batch_size"], 1)
            self.assertEqual(json_config["epochs"], 3)
            self.assertEqual(json_config["optimizer"]["learning_rate"], 0.01)
            self.assertEqual(json_config["description"], "Test on numeric data, two hidden layers")

    def test_exisiting_config_exception(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')

        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        self.assertRaises(ConfigurationAlreadyExistsError, self.mm.fit, x=x, y=y, batch_size=1, epochs=3)

    def test_multiple_runs(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')

        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        self.mm.fit(x=x, y=y, batch_size=1, epochs=5)
        self.assertTrue(len(glob.glob(os.path.join(self.test_path, "*"))) == 2)

    def test_save_history(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
        self.mm.save_history = True
        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        pickle_path = os.path.join(self.test_path, self.mm.timestamp, "history.p")
        self.assertTrue(os.path.isfile(pickle_path))
        with open(pickle_path, 'rb') as pickle_file:
            history = pickle.load(pickle_file)
            self.assertTrue(len(history["loss"]) == 3)

    def test_save_custom_loss(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=rmse)

        self.mm.save_history = True
        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))
        
        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
            loss_func = deserialize_function(json_config["loss"])
            self.assertTrue(callable(loss_func))

            self.mm.overwrite = True
            self.mm.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=loss_func)
            self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

    def test_callback_saving(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
        es = keras.callbacks.EarlyStopping(monitor='loss', patience=1, verbose=1)
        es_2 = keras.callbacks.EarlyStopping(monitor='loss', patience=3, verbose=1)

        self.mm.fit(x=x, y=y, batch_size=1, epochs=3, callbacks=[es, es_2])

        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))

        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
            self.assertTrue("callbacks" in json_config)
            self.assertTrue(len(json_config["callbacks"]) == 2)
            deserialized_callbacks = deserialize_function(json_config["callbacks"])
            self.mm.overwrite = True
            self.mm.fit(x=x, y=y, batch_size=1, epochs=3, callbacks=deserialized_callbacks)




def simple_model():
    model = Sequential()
    model.add(Dense(2, input_dim=1, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def rmse(y_true, y_pred): 
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

if __name__ == '__main__':
    unittest.main()