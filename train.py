from data_utils import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import *

X_train, Y_train = read_train_data()

u_net = get_unet()

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-3.h5', verbose=1, save_best_only=True)
results = u_net.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 
                    callbacks=[earlystopper, checkpointer])
