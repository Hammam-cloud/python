# /src/model.py
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam




def make_model(input_shape=(28, 28), num_classes=52):
	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(256, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(128, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(optimizer=Adam(learning_rate=1e-3),
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])
	return model




def save_model(model, path):
	model.save(path)




def load_trained_model(path):
	return load_model(path)




if __name__ == '__main__':
	# quick smoke test
	m = make_model()
	m.summary()