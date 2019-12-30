import zipfile
import os

try:
    os.mkdir('stage1_train')
except Exception:
    print(Exception)
with zipfile.ZipFile('stage1_train.zip', 'r') as zip_train:
    zip_train.extractall('stage1_train')
try:
    os.mkdir('stage1_test')
except Exception:
    print(Exception)
with zipfile.ZipFile('stage1_test.zip', 'r') as zip_test:
    zip_test.extractall('stage1_test')
