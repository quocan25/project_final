from tqdm import tqdm
from lib import remove_tone_line, _save_pickle, _load_pickle
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np


configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=configuration)


with open('data/train_tieng_viet.txt', 'r', encoding='utf-8') as f:
    train_output = f.readlines()

print('Number of sequences: ', len(train_output))
print('First sequence: ', train_output[0])

train_idx_500k = []
train_opt_500k = []
train_ipt_500k = []
val_idx_50k = []
val_opt_50k = []
val_ipt_50k = []
test_idx_50k = []
test_opt_50k = []
test_ipt_50k = []

for i in tqdm(range(600000)):
    [idx, origin_seq] = train_output[i].split('\t')
    try:
        non_acc_seq = remove_tone_line(origin_seq)
    except:
        print('error remove tone line at sequence {}', str(i))
        continue
    if i < 500000:
        train_idx_500k.append(idx)
        train_opt_500k.append(origin_seq)
        train_ipt_500k.append(non_acc_seq)
    elif i < 550000:
        val_idx_50k.append(idx)
        val_opt_50k.append(origin_seq)
        val_ipt_50k.append(non_acc_seq)
    else:
        test_idx_50k.append(idx)
        test_opt_50k.append(origin_seq)
        test_ipt_50k.append(non_acc_seq)


_save_pickle('data/train_tv_idx_500k.pkl', train_idx_500k)
_save_pickle('data/train_tv_opt_500k.pkl', train_opt_500k)
_save_pickle('data/train_tv_ipt_500k.pkl', train_ipt_500k)


_save_pickle('data/val_tv_idx_50k.pkl', val_idx_50k)
_save_pickle('data/val_tv_opt_50k.pkl', val_opt_50k)
_save_pickle('data/val_tv_ipt_50k.pkl', val_ipt_50k)

_save_pickle('data/test_tv_idx_50k.pkl', test_idx_50k)
_save_pickle('data/test_tv_opt_50k.pkl', test_opt_50k)
_save_pickle('data/test_tv_ipt_50k.pkl', test_ipt_50k)
# print(val_idx_50k)
# print(val_opt_50k)
# print(val_ipt_50k)