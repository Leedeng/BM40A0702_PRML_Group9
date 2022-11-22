import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataGenerator :
    """Simple Data Generator that consumes paired (img, text)
    and outputs batch of (X, Y)
    """
    def __init__(self, data_pairs,
                 output_shape=(256,256),
                 batch_size=64,
                 nb_batches_per_epoch=1000,
                 mode='training',
                 seed=123455):
        self.data_pairs = self._read_data_if_necessary(data_pairs)
        self.nb_samples = len(data_pairs)
        self.output_shape = output_shape
        self.mode = mode
        if mode != 'testing' :
            self.batch_size = batch_size
            self.nb_batches_per_epoch = nb_batches_per_epoch
        else :
            self.batch_size = 1
            self.nb_batches_per_epoch = self.nb_samples
        self.batch_idx = 0
        self.prng = self.get_prng(seed)
    def _read_data_if_necessary(self, data_pairs) :
        rets = []
        for src in data_pairs :
            if isinstance(src, str) :
                text = int(src.split('_')[1])
                src = cv2.imread(src, 0)
                rets.append([src, text])
        return rets
    def get_prng(self, seed=None) :
        if (seed is not None) :
            return np.random.RandomState(seed)
        else :
            return np.random.RandomState(None)
    def __len__(self) :
        return self.nb_batches_per_epoch
    def __iter__(self) :
        return self
    def __next__(self) :
        bidx = self.batch_idx
        if (self.batch_idx >= self.nb_batches_per_epoch) :
            bidx = self.batch_idx = 0
        else :
            self.batch_idx += 1
        return self[bidx]
    def __getitem__(self, batch_idx) :
        if self.mode != 'testing' :
            if self.mode == 'training' :
                prng = self.prng
            else :
                prng = self.get_prng(batch_idx)
            indices = prng.randint(0, self.nb_samples, size=(self.batch_size,))
        else :
            indices = [batch_idx]
            prng = self.prng
        X, Y = [], []
        for i in indices :
            img, text = self.data_pairs[i]
            x = img
            y = text
            X.append(x)
            Y.append(y)
        return self.postprocess_image(X),self.postprocess_text(Y)
    def postprocess_image(self, X):
        X = [ (x-x.min())/(x.max()-x.min()+.1) for x in X]
        return np.expand_dims(np.stack(X, axis=0), axis=-1).astype('float32')
    def postprocess_text(self, Y):
        return np.expand_dims(np.asarray(Y),axis=-1).astype('float32')
    
def splite_train_valid(dataset_dir,ratio,seed=123455):
    """ dataset: dataset root, example: 'dataset/digital_3d_processed/' 
    ratio: ratio of traning samples, exmaple: 0.7
    return training fold and validation fold"""
    np.random.RandomState(seed)
    samples_list = []
    train_samples_list = [] 
    for f in os.listdir(dataset_dir):
        samples_list.append(f)
    samples_list = np.sort(samples_list)
    valid_sample_list = samples_list.copy().tolist()
    
    # sampling every 50, beacause each class have 50 samples
    sampling_step = 50
    for i in range(0,len(samples_list),sampling_step):
        temp = np.random.choice(samples_list[i:i+sampling_step],int(sampling_step * ratio),replace=False)
        for j in range(len(temp)):
            train_samples_list.append(temp[j])
            valid_sample_list.remove(temp[j])
    
    # create training fold and validation fold
    train_list = []
    valid_list = []
    if os.path.exists('Train') == False:
        os.mkdir('Train')
        for i in range(len(train_samples_list)):
            os.rename(dataset_dir + train_samples_list[i], 'Train/' + train_samples_list[i])
            train_list.append('Train/' + train_samples_list[i])
    if os.path.exists('Valid') == False:
        os.mkdir('Valid')
        for i in range(len(valid_sample_list)):
            os.rename(dataset_dir + valid_sample_list[i], 'Valid/' + valid_sample_list[i])
            valid_list.append('Valid/' + valid_sample_list[i])
            
    return train_list,valid_list
    