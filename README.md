# ModelManager

```python
import keras
from keras.models import Sequential

from ModelManager import ModelManager

# Create some simple model
my_model = Sequential()
my_model.add(Dense(2, input_dim=1, activation='relu'))
my_model.add(Dense(1, activation='sigmoid'))
my_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy')

# Some data
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

mm = ModelManager("some/path", model=my_model, save_history=True, save_weights=True, save_model=True)

mm.fit(x=x, y=y, batch_size=1, epochs=3)
