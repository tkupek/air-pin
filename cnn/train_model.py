import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

import cnn.data_utils as utils
import cnn.get_model as get_model


if __name__ == '__main__':
	model_path = os.path.join('model', 'airpin-model_v3.h5')
	num_epochs = 100
	batch_size = 16 
	validation_size = 0.2
	resize = (int(1280 / 4), int(720 / 4))
	label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'null']

	# array of image folders 
	img_folders = [
		'../captures/0',
		'../captures/1',
		'../captures/2',
		'../captures/3',
		'../captures/4',
		'../captures/5',
		'../captures/6',
		'../captures/7',
		'../captures/8',
		'../captures/9',
		'../captures/null']

	print('read samples...')
	train_data, train_labels, test_data, test_labels = utils.get_dataset(img_folders, validation_size, resize)
	print('got ' + str(len(train_data)) + ' training and ' + str(len(test_data)) + ' test samples')

	print('create model...')
	model = get_model.get_resnet_v1_20(train_data.shape[1:], len(img_folders))
	model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

	print('start training...')
	save_best = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
	model.fit(train_data, train_labels, validation_data=[test_data, test_labels], epochs=num_epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[save_best])

