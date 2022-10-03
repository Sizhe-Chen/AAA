import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import cv2
import time
import shutil

import argparse
import PIL.Image
import matplotlib.pyplot as plt
import random

def random_classes_except_current(y_test):
    n_cls = y_test.shape[1]
    y_test = y_test.argmax(1)
    y_test_new = np.zeros_like(y_test)
    for i_img in range(y_test.shape[0]):
        lst_classes = list(range(n_cls))
        lst_classes.remove(y_test[i_img])
        y_test_new[i_img] = np.random.choice(lst_classes)
    return dense_to_onehot(y_test_new, n_cls)


def p_selection(p_init, it, num_iter):
    """ Piece-wise constant schedule for p in Square attack. """
    it = int(it / num_iter * 10000)
    if   10 < it <= 50:       return p_init / 2
    elif 50 < it <= 200:      return p_init / 4
    elif 200 < it <= 500:     return p_init / 8
    elif 500 < it <= 1000:    return p_init / 16
    elif 1000 < it <= 2000:   return p_init / 32
    elif 2000 < it <= 4000:   return p_init / 64
    elif 4000 < it <= 6000:   return p_init / 128
    elif 6000 < it <= 8000:   return p_init / 256
    elif 8000 < it <= 10000:  return p_init / 512
    elif 10000 < it <= 12000: return p_init / 1024
    elif 12000 < it <= 14000: return p_init / 2048
    else:                     return p_init

def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def ece_score(y_pred, y_test, n_bins=15):
    py = softmax(y_pred, axis=1) if y_pred.max() > 1 else y_pred

    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)





class DataManager():
    def __init__(self, corr_data, logits_label, epsilon, result_dir=None, loss_init=None):
        self.data, self.label = corr_data, logits_label # eliminated incorrectly predicted samples
        assert self.data.shape[0] == self.label.shape[0] 
        self.ground_truth = np.argmax(self.label, axis=1)
        self.num_sample = len(self.data)

        self.loss = loss_init
        self.clean_sample_indexes = np.array(range(self.num_sample+1), dtype=np.int32) # 1 more to record the end index
        self.adv_sample_indexes = np.array(range(self.num_sample+1), dtype=np.int32) # 1 more to record the end index
        
        self.epsilon = epsilon
        self.result_dir = result_dir
        if self.result_dir is not None: os.makedirs(self.result_dir, exist_ok=True)
        self.iter = np.ones(self.num_sample, dtype=np.int32)
        self.suc = np.zeros(self.num_sample, dtype=np.bool)
        self.lipschitz = np.zeros(self.num_sample, dtype=np.float32)
        self.max_negative_loss = - self.loss

    def generate_batch_forward(self, batch_size):
        indexes = np.random.choice(range(self.data.shape[0]), size=batch_size, replace=True) # False
        np.random.shuffle(indexes)
        return self.data[indexes], self.label[indexes]
    
    def update_square(self, img_adv, lbl_adv, loss, logger, save_only=False, targeted=False, **kwargs):
        data_indexes = np.argwhere(1-self.suc).reshape(-1)
        #assert img_adv.shape[0] == data_indexes.shape[0], '%d/%d' % (img_adv.shape[0], data_indexes.shape[0])
        insert_indexes = self.clean_sample_indexes[data_indexes]
        lbl_ori = self.label[self.clean_sample_indexes[data_indexes]]

        if not save_only:
            self.data = np.insert(self.data, insert_indexes, img_adv, axis=0)
            self.label = np.insert(self.label, insert_indexes, lbl_adv, axis=0)
            self.loss = np.insert(self.loss, insert_indexes, loss, axis=0)
            for index in data_indexes: self.clean_sample_indexes[index+1:] += 1
            for i, (index, next_index) in enumerate(zip(self.clean_sample_indexes[:-1], self.clean_sample_indexes[1:])):
                self.adv_sample_indexes[i] = index + np.argmax(self.loss[index:next_index])
            self.adv_sample_indexes[-1] = self.clean_sample_indexes[-1]

        success_index = np.argmax(lbl_adv, axis=1) != np.argmax(lbl_ori, axis=1)
        if targeted is not False: success_index = np.argmax(lbl_adv, axis=1) == np.argmax(targeted, axis=1)
        self.iter[data_indexes] += 1
        self.suc[data_indexes[success_index]] = True 
        
        if img_adv.shape[0] == data_indexes.shape[0] and logger is not None: 
            save_imgs(img_adv[success_index], data_indexes[success_index], logger.result_paths['adv']) ####

    def norm2(self, a, b):
        assert a.shape == b.shape, str(a.shape) + ' ' + str(b.shape)
        return np.linalg.norm(a.reshape(a.shape[0], -1) - b.reshape(b.shape[0], -1), ord=2, axis=1)

    def update_lipschitz(self):
        def calculate_lipschitz(index1, index2): return np.abs(self.loss[index1]-self.loss[index2]) / self.norm2(self.data[index1], self.data[index2])

        unsuccess_indexes = (1-self.suc).astype(np.bool)
        old_sample_indexes = self.clean_sample_indexes[:-1][unsuccess_indexes]
        new_sample_indexes = self.clean_sample_indexes[1:] [unsuccess_indexes]-1
        while 1:
            lipschitz = calculate_lipschitz(old_sample_indexes, new_sample_indexes)
            self.lipschitz[unsuccess_indexes] = np.where(lipschitz > self.lipschitz[unsuccess_indexes], lipschitz, self.lipschitz[unsuccess_indexes])
            if np.sum(old_sample_indexes) == np.sum(new_sample_indexes-1): break
            old_sample_indexes = np.clip(old_sample_indexes + 1, old_sample_indexes, new_sample_indexes-1)

    def judge_potential_maximizer(self, tentative_query):
        assert tentative_query.shape[0] == np.sum(1-self.suc), '%d/%d' % (tentative_query.shape[0], np.sum(1-self.suc))

        unsuccess_indexes = (1-self.suc).astype(np.bool)
        old_sample_indexes = self.clean_sample_indexes[:-1][unsuccess_indexes]
        new_sample_indexes = self.clean_sample_indexes[1:] [unsuccess_indexes]-1
        self.max_negative_loss[unsuccess_indexes] = np.where(
            -self.loss[new_sample_indexes] > self.max_negative_loss[unsuccess_indexes], 
            -self.loss[new_sample_indexes],  self.max_negative_loss[unsuccess_indexes])
        
        is_potential_maximizer = np.ones(np.sum(1-self.suc), dtype=np.bool)
        while 1:
            left = -self.loss[old_sample_indexes] + self.lipschitz[unsuccess_indexes] * self.norm2(tentative_query, self.data[old_sample_indexes])
            is_potential_maximizer = np.where(left < self.max_negative_loss[unsuccess_indexes] * 0.7, False, is_potential_maximizer)
            # self.max_negative_loss[unsuccess_indexes] < 0: * 0.7 or + 3 means stricter
            if np.sum(old_sample_indexes) == np.sum(new_sample_indexes-1): break
            old_sample_indexes = np.clip(old_sample_indexes + 1, old_sample_indexes, new_sample_indexes-1)
        return is_potential_maximizer

    def save(self, iter):
        info = f'Iter{iter}_Size{self.clean_sample_indexes[-1]}' #-{self.epsilon_step}
        for file in os.listdir(self.result_dir):
            if 'data' in file or 'label' in file or 'index' in file or 'adv_index' in file or 'iter' in file or 'suc' in file or 'loss' in file: 
                os.remove(self.result_dir + '/' + file) # only save the latest data to save storage
        np.save(self.result_dir + '/data_%s.npy' % info, self.data)
        np.save(self.result_dir + '/label_%s.npy' % info, self.label)
        np.save(self.result_dir + '/index_%s.npy' % info, self.clean_sample_indexes)
        np.save(self.result_dir + '/adv_index_%s.npy' % info, self.adv_sample_indexes)
        np.save(self.result_dir + '/iter_%s.npy' % info, self.iter)
        np.save(self.result_dir + '/suc_%s.npy' % info, self.suc)
        np.save(self.result_dir + '/loss_%s.npy' % info, self.loss)

    def load(self, path):
        files = os.listdir(path)
        def get_iteration(item):
            start_index = item.find('Iter') + 4
            end_index = item[start_index:].find('_') + start_index
            return int(item[start_index:end_index])

        def get_latest_item_path(item, return_outer_itr=False):
            item_files = [x for x in files if item in x]
            item_files.sort(key=get_iteration)
            return path + '/' + item_files[-1] if not return_outer_itr else get_iteration(item_files[-1])

        self.data = np.load(get_latest_item_path('data'))
        self.label = np.load(get_latest_item_path('label'))
        self.clean_sample_indexes = np.load(get_latest_item_path('index'))
        self.adv_sample_indexes = np.load(get_latest_item_path('adv_index'))
        self.iter = np.load(get_latest_item_path('iter'))
        self.suc = np.load(get_latest_item_path('suc'))
        self.loss = np.load(get_latest_item_path('loss'))
        return get_latest_item_path('data', return_outer_itr=True)


class LoggerUs():
    def __init__(self, result_path):
        self.result_paths = {}
        self.result_paths['base'] = result_path
        for sub_folder in ['adv']:
            self.result_paths[sub_folder] = self.result_paths['base'] + '/' + sub_folder
            os.makedirs(self.result_paths[sub_folder], exist_ok=True)
        self.copy_files()
    
    def copy_files(self): 
        if not os.path.exists(self.result_paths['base'] + '/src'): 
            copy_files(self.result_paths['base'] + '/src')
            return
        copy_files(self.result_paths['base'] + '/src_' + get_time())

    def remove_more_log(self, save_interval, outer_itr):
        for file_name in ['train', 'process']:
            log_file = open(self.file_paths[file_name], 'r')
            records = list(log_file)[:outer_itr]
            log_file.close()
            with open(self.file_paths[file_name],'w') as f: f.write(''.join(records))


class Logger:
    def __init__(self, path):
        self.path = path
        if path != '':
            folder = '/'.join(path.split('/')[:-1])
            if not os.path.exists(folder):
                os.makedirs(folder)

    def reset_path(self, path): self.__init__(path)

    def print(self, message):
        print(message)
        if self.path != '':
            with open(self.path, 'a') as f:
                f.write(message + '\n')
                f.flush()


def dense_to_onehot(y_test, n_cls):
    y_test_onehot = np.zeros([len(y_test), n_cls], dtype=bool)
    y_test_onehot[np.arange(len(y_test)), y_test] = True
    return y_test_onehot


def load_cifar10_1(n_ex):
    data = np.transpose(np.load('data/cifar10.1_v6_data.npy').astype(np.float32), axes=[0, 3, 1, 2]) / 255.0
    label = dense_to_onehot(np.load('data/cifar10.1_v6_labels.npy'), 10).astype(np.float32)
    return data[:n_ex], label[:n_ex]


def load_cifar10(n_ex, train=False):
    testset = torchvision.datasets.CIFAR10(root='data', train=train, download=True)
    data = np.transpose(testset.data.astype(np.float32), axes=[0, 3, 1, 2]) / 255.0
    label = dense_to_onehot(testset.targets, 10).astype(np.float32)
    return data[:n_ex], label[:n_ex]

def load_cifar100(n_ex, train=False):
    testset = torchvision.datasets.CIFAR100(root='data', train=train, download=True)
    data = np.transpose(testset.data.astype(np.float32), axes=[0, 3, 1, 2]) / 255.0
    label = dense_to_onehot(testset.targets, 100).astype(np.float32)
    return data[:n_ex], label[:n_ex]

def load_mnist(n_ex, train=False):
    testset = torchvision.datasets.MNIST(root='data', train=train, download=True)
    data = testset.data.numpy().astype(np.float32) / 255.0
    label = dense_to_onehot(testset.targets.numpy(), 10).astype(np.float32)
    return data[:n_ex, np.newaxis, ...], label[:n_ex]


def load_imagenet(n_ex, model, seed=0):
    try: 
        arch = model.arch_ori
        assert os.path.exists('data/imagenet_%s_imgs_%d.npy' % (arch, seed))
    except AttributeError: arch = model.arch
    data_path = 'data/imagenet_%s_imgs_%d.npy' % (arch, seed)
    label_path = 'data/imagenet_%s_lbls_%d.npy' % (arch, seed)
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        with open('data/val.txt', 'r') as f: txt = f.read().split('\n')
        labels = {}
        for item in txt:
            if ' ' not in item: continue
            file, cls = item.split(' ')
            labels[file] = int(cls)
        
        data = []
        files = os.listdir('data/ILSVRC2012_img_val')
        label = np.zeros((min([1000, n_ex]), 1000), dtype=np.uint8)
        label_done = []
        random.seed(seed)
        
        for i in random.sample(range(len(files)), len(files)):
            file = files[i]
            lbl = labels[file]
            if lbl in label_done: continue
            
            img = np.array(PIL.Image.open(
                'data/ILSVRC2012_img_val' + '/' + file).convert('RGB').resize((224, 224))) \
                .astype(np.float32).transpose((2, 0, 1)) / 255
            prd = model(img[np.newaxis, ...]).argmax(1)
            if prd != lbl: continue
            
            label[len(data), lbl] = 1
            data.append(img)
            label_done.append(lbl)
            print('selecting samples in different classes...', len(label_done), '/',1000, end='\r')
            if len(label_done) == min([1000, n_ex]): break

        x_test = np.array(data)
        y_test = np.array(label)
        np.save(data_path, x_test)
        np.save(label_path, y_test)
    else:
        x_test = np.load(data_path)
        y_test = np.load(label_path)
    return x_test[:n_ex], y_test[:n_ex]



def save_imgs(imgs, indexes, result_path_adv):
    assert imgs.shape[0] == indexes.shape[0], 'imgs shape %d != indexes shape %d' % (imgs.shape[0], indexes.shape[0])
    for i in range(imgs.shape[0]):
        img = (imgs[i]*255).astype(np.uint8).transpose(1, 2, 0)
        PIL.Image.fromarray(img if img.shape[2] == 3 else img[:, :, 0]).save(result_path_adv + '/%d.png' % indexes[i])


def get_time(): return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))


def copy_files(result_dir, forms=['.py'], eliminated=['-', '__pycache__', 'data']):
    for root, _, files in os.walk('.'):
        do_continue = False
        for item in eliminated:
            if item in root: do_continue = True
        if do_continue: continue
        for file in files:
            do_copy = False
            for item in forms:
                if item in file: do_copy = True
            if not do_copy: continue
            destiny_path = result_dir + root[1:]
            os.makedirs(destiny_path, exist_ok=True)
            shutil.copyfile(root + '/' + file, destiny_path + '/' + file)


def output(value_dict, stream=None, bit=3, prt=True, end='\n'):
    output_str = ''
    for key, value in value_dict.items():
        if isinstance(value, list): #value = value[-1]
            for i in range(len(value)): value[i] = round(value[i], bit)
        if isinstance(value, float) or isinstance(value, np.float32) or isinstance(value, np.float64): value = round(value, bit)
        output_str += '[ ' + str(key) + ' ' + str(value) + ' ] '
    if prt: print(output_str, end=end)
    if stream is not None: print(output_str, file=stream)
