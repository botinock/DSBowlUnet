from data_utils import *
from model import *

X_train, Y_train = read_train_data()
X_test, sizes_test = read_test_data()

u_net = load_model('model-dsbowl2018-3.h5', custom_objects={'dice_coef': dice_coef})
preds_train = u_net.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = u_net.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = u_net.predict(X_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

a = np.vstack((preds_train_t, preds_val_t))
np.save('train_preds.npy', a)
np.save('test_preds.npy', preds_test_t)
np.save('preds_test_upsampled.npy', preds_test_upsampled)
