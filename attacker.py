import torch
import torchvision
import os
import time
import numpy as np
from copy import deepcopy
import threading

from utils import *
from PCDARTS import *

class QueryNet():
    def __init__(self, sampler, victim_name, surrogate_names, use_horizontal_info, use_random_info, nas, linfty, eps, batch_size, iter_square_s):
        self.surrogate_names = surrogate_names
        self.use_vertical_info = self.surrogate_names != []
        self.use_horizontal_info = use_horizontal_info
        self.use_random_info = use_random_info
        assert (self.use_vertical_info     and self.use_horizontal_info     and self.use_random_info) or \
               (self.use_vertical_info     and not self.use_horizontal_info and self.use_random_info) or \
               (self.use_vertical_info     and not self.use_horizontal_info and not self.use_random_info) or \
               (not self.use_vertical_info and self.use_horizontal_info     and not self.use_random_info) or \
               (not self.use_vertical_info and not self.use_horizontal_info and self.use_random_info)
        self.eps = eps
        self.nas = nas

        self.batch_size = batch_size
        self.victim_name = victim_name
        self.horizontal_max_trial = 50
        self.surrogate_train_iter = 500 
        self.save_surrogate = False
        
        self.sampler = sampler
        self.generator = PGDGeneratorInfty(int(batch_size / 2)) if linfty else PGDGenerator2(int(batch_size / 2))
        self.square_attack = self.square_attack_linfty if linfty else self.square_attack_l2
        self.surrogates = []
        
        gpus = torch.cuda.device_count()
        num_class = self.sampler.label.shape[1]
        self.nas_layers = [10, 6, 8, 4, 12]

        if   gpus == 1: gpu_ids = [0, 0, 0]
        elif gpus == 2: gpu_ids = [0, 1, 1]
        elif gpus == 3: gpu_ids = [0, 1, 2]
        elif gpus == 4: gpu_ids = [1, 2, 3]
        else: raise NotImplementedError
        for i, surrogate_name in enumerate(surrogate_names):
            self.surrogates.append(NASSurrogate(init_channels=16,layers=self.nas_layers[i],num_class=num_class,gpu_id=gpu_ids[i])) 
        
        self.num_attacker = len(surrogate_names) + int(use_horizontal_info) + int(use_random_info)
        self.attacker_authority = [1] * len(surrogate_names) + [0, 0]
        self.eva_weights_threshold = 10
        self.iter_square_s = iter_square_s
    
    def pseudo_gaussian_pert_rectangles(self, x, y):
        delta = np.zeros([x, y])
        x_c, y_c = x // 2 + 1, y // 2 + 1
        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[max(counter2[0], 0):min(counter2[0] + (2 * counter + 1), x),
            max(0, counter2[1]):min(counter2[1] + (2 * counter + 1), y)] += 1.0 / (counter + 1) ** 2

            counter2[0] -= 1
            counter2[1] -= 1
        delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
        return delta

    def meta_pseudo_gaussian_pert(self, s):
        delta = np.zeros([s, s])
        n_subsquares = 2
        if n_subsquares == 2:
            delta[:s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s)
            delta[s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s) * (-1)
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))
            if np.random.rand(1) > 0.5: delta = np.transpose(delta)

        elif n_subsquares == 4:
            delta[:s // 2, :s // 2] = self.pseudo_gaussian_pert_rectangles(s // 2, s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, :s // 2] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s // 2) * np.random.choice([-1, 1])
            delta[:s // 2, s // 2:] = self.pseudo_gaussian_pert_rectangles(s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta[s // 2:, s // 2:] = self.pseudo_gaussian_pert_rectangles(s - s // 2, s - s // 2) * np.random.choice([-1, 1])
            delta /= np.sqrt(np.sum(delta ** 2, keepdims=True))

        return delta

    def square_attack_l2(self, x_curr, x_best_curr, deltas, is_potential_maximizer, min_val, max_val, p, **kwargs):
        c, h, w = x_curr.shape[1:]
        n_features = c * h * w
        s = max(int(round(np.sqrt(p * n_features / c))), 3)

        if s % 2 == 0: s += 1
        s2 = s + 0
        
        ### window_1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        new_deltas_mask = np.zeros(x_curr.shape)
        new_deltas_mask[:, :, center_h:center_h + s, center_w:center_w + s] = 1.0

        ### window_2
        center_h_2 = np.random.randint(0, h - s2)
        center_w_2 = np.random.randint(0, w - s2)
        new_deltas_mask_2 = np.zeros(x_curr.shape)
        new_deltas_mask_2[:, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 1.0

        ### compute total norm available
        curr_norms_window = np.sqrt(
            np.sum(((x_best_curr - x_curr) * new_deltas_mask) ** 2, axis=(2, 3), keepdims=True))
        curr_norms_image = np.sqrt(np.sum((x_best_curr - x_curr) ** 2, axis=(1, 2, 3), keepdims=True))
        mask_2 = np.maximum(new_deltas_mask, new_deltas_mask_2)
        norms_windows = np.sqrt(np.sum((deltas * mask_2) ** 2, axis=(2, 3), keepdims=True))

        ### create the updates
        new_deltas = np.ones([x_curr.shape[0], c, s, s])
        new_deltas = new_deltas * self.meta_pseudo_gaussian_pert(s).reshape([1, 1, s, s])
        new_deltas *= np.random.choice([-1, 1], size=[x_curr.shape[0], c, 1, 1])
        old_deltas = deltas[:, :, center_h:center_h + s, center_w:center_w + s] / (1e-10 + curr_norms_window)
        new_deltas += old_deltas
        new_deltas = new_deltas / np.sqrt(np.sum(new_deltas ** 2, axis=(2, 3), keepdims=True)) * (
                np.maximum(self.eps ** 2 - curr_norms_image ** 2, 0) / c + norms_windows ** 2) ** 0.5
        deltas[~is_potential_maximizer, :, center_h_2:center_h_2 + s2, center_w_2:center_w_2 + s2] = 0.0  # set window_2 to 0
        deltas[~is_potential_maximizer, :, center_h:center_h + s, center_w:center_w + s] = new_deltas[~is_potential_maximizer, ...] + 0  # update window_1

        x_new = x_curr + deltas / np.sqrt(np.sum(deltas ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps
        x_new = np.clip(x_new, min_val, max_val)
        return x_new, deltas
    
    def square_attack_linfty(self, x_curr, x_best_curr, deltas, is_potential_maximizer, min_val, max_val, p, **kwargs):
        c, h, w = x_curr.shape[1:]
        n_features = c * h * w
        s = int(round(np.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1
        center_h = np.random.randint(0, h - s)
        center_w = np.random.randint(0, w - s)
        deltas[~is_potential_maximizer, :, center_h:center_h + s, center_w:center_w + s] = np.random.choice([-self.eps, self.eps], size=[c, 1, 1])

        # judge overlap
        for i_img in range(x_best_curr.shape[0]):
            if is_potential_maximizer[i_img]: continue
            center_h_tmp, center_w_tmp, s_tmp = center_h, center_w, s
            while np.sum(np.abs(np.clip(
                x_curr[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp] + 
                deltas[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp], 
                min_val, max_val) - 
                x_best_curr[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp]) 
                < 10 ** -7) == c * s * s:
                s_tmp = int(round(np.sqrt(p * n_features / c)))
                s_tmp = min(max(s_tmp, 1), h - 1) 
                center_h_tmp, center_w_tmp = np.random.randint(0, h - s_tmp), np.random.randint(0, w - s_tmp)
                deltas[i_img, :, center_h_tmp:center_h_tmp + s_tmp, center_w_tmp:center_w_tmp + s_tmp] = np.random.choice([-self.eps, self.eps], size=[c, 1, 1])
        return np.clip(x_curr + deltas, min_val, max_val), deltas
    
    def square_attacker(self, x_curr, x_best_curr, **kwargs): 
        x_next, _ = self.square_attack(x_curr, x_best_curr, x_best_curr-x_curr, np.zeros(x_best_curr.shape[0], dtype=np.bool), **kwargs)
        return x_next

    def square_attacker_s(self, x_curr, x_best_curr, y_curr, get_surrogate_loss, **kwargs):
        loss_min = self.get_surrogate_loss_multi_threading(get_surrogate_loss, x_best_curr, y_curr)
        loss_min = sum(loss_min) / len(loss_min)
        x_next = deepcopy(x_best_curr)
        itr = 0
        loss_min_mean = loss_min.mean()
        while 1:
            print(itr, loss_min.mean(), end='\r') 
            x_square = self.square_attacker(x_curr, x_best_curr, **kwargs)
            loss = self.get_surrogate_loss_multi_threading(get_surrogate_loss, x_square, y_curr)
            loss = sum(loss) / len(loss)
            idx_improved = loss < loss_min
            loss_min = idx_improved * loss + ~idx_improved * loss_min
            idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x_next.shape[:-1])])
            x_next = idx_improved * x_square + ~idx_improved * x_next
            loss_min_mean = loss_min.mean()
            itr += 1
            if itr > loss_min_mean * 10: break
        return x_next

    def horizontal_attacker(self, x_curr, x_best_curr, **kwargs):
        is_potential_maximizer = np.zeros(x_best_curr.shape[0], dtype=np.bool)
        deltas = x_best_curr-x_curr
        for i in range(self.horizontal_max_trial):
            x_next, deltas = self.square_attack(x_curr, x_best_curr, deltas, is_potential_maximizer, **kwargs)
            is_potential_maximizer = self.sampler.judge_potential_maximizer(x_next)
            if np.sum(is_potential_maximizer) == x_best_curr.shape[0]: break
        return x_next
    
    def surrogate_attacker(self, x_curr, x_best_curr, y_curr, attacker_id, targeted, **kwargs):
        assert attacker_id < len(self.surrogate_names)
        log_file_path = '%s/NAStrain_%s_%d.log' % (self.sampler.result_dir[:-3], self.victim_name, self.nas_layers[attacker_id])

        train_synchronized_flag = [False for _ in range(len(self.surrogate_names))]
        for i in range(attacker_id): train_synchronized_flag[i] = True
        while train_synchronized_flag != self.train_synchronized_flag: time.sleep(0.1)
        training_data = self.surrogates[attacker_id].get_training_data(self.surrogate_train_iter * self.batch_size, self.sampler)
        if attacker_id == len(self.surrogate_names) - 1: self.train_synchronized_flag = [False for _ in range(len(self.surrogate_names))]
        else: self.train_synchronized_flag[attacker_id] = True

        for i in range(self.surrogate_train_iter): self.surrogates[attacker_id].train(self.sampler, self.batch_size, i, log_file_path, data=training_data)
        self.x_new_tmp[attacker_id] = self.generator(x_best_curr, x_curr, self.eps, self.surrogates[attacker_id], y_curr, targeted=targeted)

    def save(self, n_iter):
        self.sampler.save(n_iter)
        for i, surrogate in enumerate(self.surrogates): surrogate.save('%s/surrogate%d.pth' % (self.sampler.result_dir, i))
    
    def load(self, path):
        path = path + '/var'
        n_iter = self.sampler.load(path)
        for i, surrogate in enumerate(self.surrogates): surrogate.load('%s/surrogate%d.pth' % (path, i))
        return n_iter

    def surrogate_attacker_multi_threading(self, x_curr, x_best_curr, y_curr, targeted, **kwargs):
        threads = [] # train and attack via different surrogates simultaneously
        self.train_synchronized_flag = [False for _ in range(len(self.surrogate_names))]
        self.x_new_tmp = [0 for _ in range(len(self.surrogate_names))]
        for attacker_id in range(len(self.surrogate_names)):
            threads.append(threading.Thread(target=self.surrogate_attacker, args=(x_curr, x_best_curr, y_curr, attacker_id, targeted)))
        for attacker_id in range(len(self.surrogate_names)): threads[attacker_id].start()
        for attacker_id in range(len(self.surrogate_names)): 
            if threads[attacker_id].is_alive(): threads[attacker_id].join()
        return self.x_new_tmp

    def get_surrogate_loss(self, get_surrogate_loss, evaluator_id, x_new_candidate_attacker_id, y_curr):
        self.surrogate_loss_tmp[evaluator_id] = get_surrogate_loss(self.surrogates[evaluator_id], x_new_candidate_attacker_id, y_curr)

    def get_surrogate_loss_multi_threading(self, get_surrogate_loss, x_new_candidate_attacker_id, y_curr):
        threads = [] # train and attack via different surrogates simultaneously
        self.surrogate_loss_tmp = [0 for _ in range(len(self.surrogate_names))]
        for evaluator_id in range(len(self.surrogate_names)):
            threads.append(threading.Thread(target=self.get_surrogate_loss, args=(get_surrogate_loss, evaluator_id, x_new_candidate_attacker_id, y_curr)))
        for evaluator_id in range(len(self.surrogate_names)): threads[evaluator_id].start()
        for evaluator_id in range(len(self.surrogate_names)): 
            if threads[evaluator_id].is_alive(): threads[evaluator_id].join()
        return self.surrogate_loss_tmp

    def yield_candidate_queries(self, x_curr, x_best_curr, y_curr, get_surrogate_loss, **kwargs):
        if max(self.attacker_authority) == self.attacker_authority[-2]:
            x_new_potential = []
            if self.use_horizontal_info:  x_new_potential.append(self.horizontal_attacker(x_curr, x_best_curr, **kwargs))
            if self.use_random_info:      x_new_potential.append(self.square_attacker(x_curr, x_best_curr, **kwargs))
            return x_new_potential
        elif max(self.attacker_authority) == self.attacker_authority[-1]:
            self.iter_square_s -= 1
            if self.iter_square_s > 0: return [self.square_attacker_s(x_curr, x_best_curr, y_curr, get_surrogate_loss, **kwargs)]
            else: return [self.square_attacker(x_curr, x_best_curr, **kwargs)]
            #return [self.square_attacker(x_curr, x_best_curr, **kwargs)]
        else:
            self.iter_square_s += 1
            x_new_potential = self.surrogate_attacker_multi_threading(x_curr, x_best_curr, y_curr, **kwargs)
            if self.use_horizontal_info: x_new_potential.append(self.horizontal_attacker(x_curr, x_best_curr, **kwargs))
            elif self.use_random_info:   x_new_potential.append(self.square_attacker(x_curr, x_best_curr, **kwargs))
            return x_new_potential

    def forward(self, x_curr, x_best_curr, y_curr, get_surrogate_loss, **kwargs):
        x_new_potential = self.yield_candidate_queries(x_curr, x_best_curr, y_curr, get_surrogate_loss, **kwargs)
        if len(x_new_potential) == 1: return x_new_potential[0], None
        else:
            loss_potential = [] # num_attacker * num_sample
            for attacker_id in range(len(x_new_potential)):
                loss_candidate_for_one_attacker = self.get_surrogate_loss_multi_threading(get_surrogate_loss, x_new_potential[attacker_id], y_curr)
                for evaluator_id in range(len(self.surrogate_names)):
                    loss_candidate_for_one_attacker[evaluator_id] *= self.attacker_authority[evaluator_id]
                loss_potential.append(sum(np.array(loss_candidate_for_one_attacker))/len(loss_candidate_for_one_attacker))
            loss_potential = np.array(loss_potential)
            
            x_new_index = np.argmin(loss_potential, axis=0)
            x_new = np.zeros(x_curr.shape)
            for attacker_id in range(len(x_new_potential)):
                attacker_index = x_new_index == attacker_id
                x_new[attacker_index] = x_new_potential[attacker_id][attacker_index]
        return x_new, x_new_index
    
    def backward(self, idx_improved, x_new_index, **kwargs):
        if self.use_vertical_info: 
            if self.use_horizontal_info or self.use_random_info:
                self.sampler.update_square(save_only=max(self.attacker_authority) == self.attacker_authority[-1], **kwargs)
            else:
                self.sampler.update_square(save_only=False, **kwargs)
        elif self.use_horizontal_info: # 010
            self.sampler.update_square(save_only=False, **kwargs)
            self.sampler.update_lipschitz()
        elif self.use_random_info: # 001
            self.sampler.update_square(save_only=True, **kwargs)

        if x_new_index is None: 
            if max(self.attacker_authority) == self.attacker_authority[-2]: 
                return self.attacker_authority, [0 for _ in range(len(self.surrogate_names))] + [1, 0]
            elif max(self.attacker_authority) == self.attacker_authority[-1]: 
                return self.attacker_authority, [0 for _ in range(len(self.surrogate_names))] + [0, 1]
            else: raise ValueError
        
        if max(self.attacker_authority) == self.attacker_authority[-2]:
            assert x_new_index.max() == 1 and self.use_horizontal_info and self.use_random_info
            attacker_selected = [0 for _ in range(len(self.surrogate_names))]
            for attacker_id in range(x_new_index.max()+1):

                attacker_index = x_new_index == attacker_id
                attacker_selected.append(np.mean(attacker_index))
                attacker_id_real = attacker_id + len(self.surrogate_names)
                
                if np.sum(attacker_index) < self.eva_weights_threshold: self.attacker_authority[attacker_id_real] = 0 ###
                else: self.attacker_authority[attacker_id_real] = np.sum(idx_improved[attacker_index]) / np.sum(attacker_index)

        else:
            assert x_new_index.max() in [len(self.surrogate_names)-1, len(self.surrogate_names)] # vertical only, vertical + others
            attacker_selected = []
            for attacker_id in range(x_new_index.max()+1):

                attacker_index = x_new_index == attacker_id
                if x_new_index.max() == len(self.surrogate_names)-1 or attacker_id != x_new_index.max():
                    attacker_id_real = attacker_id
                    attacker_selected += [np.mean(attacker_index)]
                    if attacker_id == x_new_index.max(): attacker_selected += [0, 0]
                elif self.use_horizontal_info:
                    attacker_id_real = attacker_id
                    attacker_selected += [np.mean(attacker_index), 0]
                else:
                    attacker_id_real = attacker_id + 1
                    attacker_selected += [0, np.mean(attacker_index)]

                if np.sum(attacker_index) < self.eva_weights_threshold: self.attacker_authority[attacker_id_real] = 0 ###
                else: self.attacker_authority[attacker_id_real] = np.sum(idx_improved[attacker_index]) / np.sum(attacker_index)
        return self.attacker_authority, attacker_selected
        

class NASSurrogate():
    def __init__(self, init_channels=16, layers=8, gpu_id=0, num_class=10):
        self.criterion = torch.nn.CrossEntropyLoss() if num_class == 1000 else torch.nn.MSELoss()
        self.device = torch.device(('cuda:%d' % gpu_id) if torch.cuda.is_available() else 'cpu')
        self.surrogate = NASNetwork(C=init_channels, num_classes=num_class, layers=layers, criterion=self.criterion, device=self.device)
        self.surrogate = self.surrogate.to(self.device)
        self.architect = Architect(self.surrogate, momentum=0.9, weight_decay=3e-4, arch_learning_rate=6e-4, arch_weight_decay=1e-3)
        
        self.optimizer = torch.optim.SGD(self.surrogate.parameters(), lr=0.1, momentum=0.9, weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 5.0, eta_min=0.0)
        self.num_class = num_class
        self.iter_train = -1

    def __call__(self, img, no_grad=True):
        # img : B * C * H * W  0~1 torch.Tensor
        # return: B * P  torch.Tensor
        if no_grad: 
            self.surrogate.eval()
            with torch.no_grad(): return self.surrogate(img.to(self.device))
        else:
            return self.surrogate(img.to(self.device))

    def get_training_data(self, num_sample, sampler):
        img_ori, lbl_ori = sampler.generate_batch_forward(num_sample)
        img_ori_search, lbl_ori_search = sampler.generate_batch_forward(num_sample)
        return img_ori, lbl_ori, img_ori_search, lbl_ori_search

    
    def train(self, sampler, batch_size, iter_train, log_file_path, data=None):
        log_file = open(log_file_path, 'a')
        #return 0 ########
        def get_batch():
            _img, lbl = sampler.generate_batch_forward(batch_size)

            if self.num_class != 1000: img = _img
            else:
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                
                img = deepcopy(_img)
                for i in range(3): img[:, i, :, :] = (img[:, i, :, :]-mean[i])/std[i]
            return img, lbl

        if data is None: 
            img_ori, lbl_ori = get_batch()
            img_ori_search, lbl_ori_search = get_batch()
        else:
            img_ori, lbl_ori, img_ori_search, lbl_ori_search = \
                data[0][iter_train*batch_size:(iter_train+1)*batch_size], \
                data[1][iter_train*batch_size:(iter_train+1)*batch_size], \
                data[2][iter_train*batch_size:(iter_train+1)*batch_size], \
                data[3][iter_train*batch_size:(iter_train+1)*batch_size]

        img_ori = torch.Tensor(img_ori).to(self.device)
        img_ori.requires_grad = True
        lbl_ori = torch.Tensor(lbl_ori).to(self.device)

        if iter_train != self.iter_train:
            self.scheduler.step()
            self.iter_train = iter_train
            self.lr = self.scheduler.get_lr()[0]
        self.surrogate.train()
        self.optimizer.zero_grad()

        # NAS
        
        img_ori_search = torch.tensor(img_ori_search, dtype=torch.float32, requires_grad=False).to(self.device)
        lbl_ori_search = torch.tensor(lbl_ori_search, dtype=torch.float32, requires_grad=False).to(self.device)
        #print(img_ori.shape, lbl_ori.shape, img_ori_search.shape, lbl_ori_search.shape); exit()
        self.architect.step(img_ori, lbl_ori, img_ori_search, lbl_ori_search, self.lr, self.optimizer, unrolled=False)
        # if epoch > 15 in official implementation
        
        # normal
        lbl = self.__call__(img_ori, no_grad=False)
        loss = self.criterion(lbl, lbl_ori.argmax(axis=1).int().long()) if self.num_class == 1000 else self.criterion(lbl, lbl_ori)
        acc = (lbl.argmax(axis=1).int() == lbl_ori.argmax(axis=1).int()).float().mean().detach().cpu().numpy()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.surrogate.parameters(), 5)###
        self.optimizer.step()
        
        output({'Batch': iter_train, 'Loss': '%.5f' % loss.detach(), 'Acc': round(acc*100, 2)}, end='\r', stream=log_file) # 
        log_file.close()
        return loss.detach()

    def save(self, save_name):
        self.surrogate.eval()
        #for file in os.listdir('.'): os.remove(file) # only save the latest data to save storage
        torch.save(self.surrogate.state_dict(), save_name)
    
    def load(self, model_path):
        print('Load surrogate from', model_path)
        if self.num_class == 10: self.surrogate.load_state_dict(state_dict=torch.load(model_path))
        else: self.surrogate.model.load_state_dict(state_dict=torch.load(model_path))


class PGDGeneratorInfty():
    def __init__(self, max_batch_size):
        self.device = torch.device('cpu')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_batch_size = max_batch_size

    def _call(self, img, lbl, surrogate, epsilon, targeted):
        # img : B * H * W * C  0~1 np.float32 array
        img = img.to(surrogate.device)
        img.requires_grad = True
        lbl = torch.Tensor(lbl).to(surrogate.device)
        
        if not targeted:
            alpha = epsilon * 2
            num_iter = 1
        else:
            alpha = 4 / 255
            num_iter = 10
        for i in range(num_iter):
            """
            surrogate.surrogate.zero_grad()
            img = torch.autograd.Variable(img.data, requires_grad=True)
            random_direction = torch.rand(lbl.shape).to(surrogate.device) * 2 - 1
            with torch.enable_grad():
                loss = (surrogate(img, no_grad=False) * random_direction).sum()
            loss.backward()
            img = img + alpha * img.grad.data.sign() 
            """
            surrogate.surrogate.zero_grad()
            loss = self.criterion(surrogate(img, no_grad=False), lbl.argmax(dim=-1))
            grad = torch.autograd.grad(loss.sum(), img)[0]
            img = img + alpha * grad.sign() 
            
        return img.to(self.device)

    def __call__(self, img, ori, epsilon, surrogate, lbl, return_numpy=True, targeted=False):
        # img : B * H * W * C  0~1 np.float32 array
        # return: B * H * W * C  np.float32 array   /   B * C * H * W  0~1  torch.Tensor
        # CPU
        #torch.cuda.empty_cache()
        img, ori = torch.Tensor(img), torch.Tensor(ori)
        batch_size = min([self.max_batch_size, img.shape[0]])
        if batch_size < self.max_batch_size: adv = self._call(img, lbl, surrogate, epsilon, targeted=targeted)
        else:
            batch_num = int(img.shape[0] / batch_size)
            if batch_size * batch_num != int(img.shape[0]): batch_num += 1
            adv = self._call(img[:batch_size], lbl[:batch_size], surrogate, epsilon, targeted=targeted)
            for i in range(batch_num-1):
                new_adv = torch.cat((adv, 
                                self._call(img[batch_size*(i+1):batch_size*(i+2)], 
                                           lbl[batch_size*(i+1):batch_size*(i+2)], 
                                           surrogate, epsilon, targeted=targeted)), 0) 
                del adv; #torch.cuda.empty_cache()
                adv = new_adv
        
        adv = torch.min(torch.max(adv, ori - epsilon), ori + epsilon)
        adv = torch.clamp(adv, 0.0, 1.0)
        if return_numpy: return adv.detach().numpy()
        else: return adv


class PGDGenerator2():
    def __init__(self, max_batch_size):
        self.device = torch.device('cpu')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.max_batch_size = max_batch_size

    def _call(self, img, ori, lbl, surrogate, epsilon, targeted):
        # img : B * H * W * C  0~1 np.float32 array
        img = img.to(surrogate.device)
        img.requires_grad = True
        lbl = torch.Tensor(lbl).to(surrogate.device)
        
        alpha = epsilon * 2
        surrogate.surrogate.zero_grad()
        loss = self.criterion(surrogate(img, no_grad=False), lbl.argmax(dim=-1))
        grad = torch.autograd.grad(loss.sum(), img)[0]
        #momentum_grad += grad
        #print(torch.norm(grad.reshape(grad.shape[0], -1), dim=1, p=2, keepdim=True).shape) #1025*1
        img = img + alpha * grad / \
            torch.norm(grad.reshape(grad.shape[0], -1), dim=1, p=2, keepdim=True).reshape(-1).repeat(grad.shape[1], grad.shape[2], grad.shape[3], 1).permute(3, 0, 1, 2)
        #.sign() # maximum attack step: FGSM
        #torch.cuda.empty_cache()
        return img.to(self.device)

    def __call__(self, img, ori, epsilon, surrogate, lbl, return_numpy=True, targeted=False):
        # img : B * H * W * C  0~1 np.float32 array
        # return: B * H * W * C  np.float32 array   /   B * C * H * W  0~1  torch.Tensor
        # CPU
        #torch.cuda.empty_cache()
        img, ori = torch.Tensor(img), torch.Tensor(ori)
        batch_size = min([self.max_batch_size, img.shape[0]])
        if batch_size < self.max_batch_size: adv = self._call(img, ori, lbl, surrogate, epsilon, targeted=targeted)
        else:
            batch_num = int(img.shape[0] / batch_size)
            if batch_size * batch_num != int(img.shape[0]): batch_num += 1
            adv = self._call(img[:batch_size], ori[:batch_size], lbl[:batch_size], surrogate, epsilon, targeted=targeted)
            for i in range(batch_num-1):
                new_adv = torch.cat((adv, 
                                self._call(img[batch_size*(i+1):batch_size*(i+2)], 
                                           ori[batch_size*(i+1):batch_size*(i+2)], 
                                           lbl[batch_size*(i+1):batch_size*(i+2)], 
                                           surrogate, epsilon, targeted=targeted)), 0) 
                del adv
                #torch.cuda.empty_cache()
                adv = new_adv
        
        per = adv - ori
        adv = ori + per / \
            torch.norm(per.reshape(per.shape[0], -1), dim=1, p=2, keepdim=True).reshape(-1).repeat(per.shape[1], per.shape[2], per.shape[3], 1).permute(3, 0, 1, 2) * epsilon
        #torch.min(torch.max(adv, ori - epsilon), ori + epsilon)
        adv = torch.clamp(adv, 0.0, 1.0)
        #adv = ((adv*255).int()/255).float() # quantize
        if return_numpy: return adv.detach().numpy()
        else: return adv