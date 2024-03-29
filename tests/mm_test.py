import unittest
import os
import shutil
import json
import glob
import pickle
import time
from tensorflow.keras import callbacks

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, TensorBoard

from KerasModelManager.ModelManager import ModelManager, ConfigurationAlreadyExistsError
from KerasModelManager.utils import deserialize_function
from test_utils import simple_model, rmse, SimpleGenerator


class TestModelManagerSet(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.abspath(os.path.join(".", "test_dir"))
        if not os.path.exists(self.test_path):
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
        description = "Test Run Model 123"
        self.mm.description = description
        self.assertEqual(self.mm.description, description)


class TestModelManagerModelFunctions(unittest.TestCase):
    def setUp(self):
        self.test_path = os.path.abspath(os.path.join(".", "test_dir"))
        if not os.path.exists(self.test_path):
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
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
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

    def test_model_fit_generator(self):
        self.mm.model = self.simple_model
        x = np.asarray([1, 2, 3, 4, 5])
        y = np.asarray([1, 2, 3, 4, 5])
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        gen = SimpleGenerator(x, y, 1)
        self.mm.fit(x=gen, steps_per_epoch=1, epochs=3)

        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
            self.assertEqual(json_config["steps_per_epoch"], 1)


    def test_exisiting_config_exception(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        self.assertRaises(ConfigurationAlreadyExistsError, self.mm.fit, x=x, y=y, batch_size=1, epochs=3)

    def test_multiple_runs(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)
        time.sleep(3) # Necessary to ensure different timestamp between runs
        self.mm.fit(x=x, y=y, batch_size=1, epochs=5)
        self.assertTrue(len(glob.glob(os.path.join(self.test_path, "*"))) == 2)

    def test_save_history(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        self.mm.save_history = True
        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        pickle_path = os.path.join(self.test_path, self.mm.timestamp, "history.p")
        self.assertTrue(os.path.isfile(pickle_path))
        with open(pickle_path, 'rb') as pickle_file:
            history = pickle.load(pickle_file)
            self.assertTrue(len(history["loss"]) == 3)

    def test_save_model(self):
        self.mm.model = self.simple_model
        self.mm._save_model = True
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        self.mm.save_history = True
        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        model_path = os.path.join(self.test_path, self.mm.timestamp, "model.h5")
        self.assertTrue(os.path.isfile(model_path))

    def test_save_weights(self):
        self.mm.model = self.simple_model
        self.mm._save_weights = True
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        self.mm.save_history = True
        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        weights_path = os.path.join(self.test_path, self.mm.timestamp, "weights.h5")
        self.assertTrue(os.path.isfile(weights_path))


    def test_save_custom_loss(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss=rmse)

        self.mm.save_history = True
        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))
        
        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
        loss_func = deserialize_function(json_config["loss"])
        self.assertTrue(callable(loss_func))

        self.mm.overwrite = True
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss=loss_func)
        self.mm.fit(x=x, y=y, batch_size=1, epochs=3)

    def test_callback_saving(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        es = EarlyStopping(monitor='loss', patience=1, verbose=1)
        es_2 = EarlyStopping(monitor='loss', patience=3, verbose=1)
        reducer = ReduceLROnPlateau(patience=3)

        self.mm.fit(x=x, y=y, batch_size=1, epochs=3, callbacks=[es, es_2, reducer])

        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))

        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
            self.assertTrue("callbacks" in json_config)
            self.assertTrue(len(json_config["callbacks"]) == 3)
            deserialized_callbacks = [deserialize_function(f) for f in json_config["callbacks"]]
            self.mm.overwrite = True
            self.mm.fit(x=x, y=y, batch_size=1, epochs=3, callbacks=deserialized_callbacks)

    def test_validation_data(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        val = (np.array([2, 3, 4, 1]), np.array([0.7, 1, 3, 0.5]))
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        self.mm.fit(x=x, y=y, validation_data=val, batch_size=1, epochs=3)
        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))

        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
            self.assertTrue("validation_data" in json_config)

    def test_callback_save_path(self):
        self.mm.model = self.simple_model
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        val = (np.array([2, 3, 4, 1]), np.array([0.7, 1, 3, 0.5]))
        self.mm.model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        tb = TensorBoard(log_dir=self.mm.callback_log_path)
        self.mm.fit(x=x, y=y, validation_data=val, batch_size=1, epochs=3, callbacks=[tb])

        json_path = os.path.join(self.test_path, self.mm.timestamp, "config.json")
        self.assertTrue(os.path.isfile(json_path))

        with open(json_path, 'r') as json_file:
            json_config = json.load(json_file)
            self.assertTrue("callbacks" in json_config)
            deserialized_callbacks = [deserialize_function(f) for f in json_config["callbacks"]]
            self.assertTrue(deserialized_callbacks[0].log_dir ==  self.mm.callback_log_path)



if __name__ == '__main__':
    unittest.main()