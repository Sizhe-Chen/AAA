import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from robustbench.utils import load_model
device = torch.device('cuda:0')
from utils import softmax, ece_score
import os
verbose = False

def loss(y, logits, targeted=False, loss_type='margin_loss'):
    if loss_type == 'margin_loss':
        preds_correct_class = (logits * y).sum(1, keepdims=True)
        diff = preds_correct_class - logits
        diff[y] = np.inf
        margin = diff.min(1, keepdims=True)
        loss = margin * -1 if targeted else margin
    elif loss_type == 'cross_entropy':
        probs = softmax(logits)
        loss = -np.log(probs[y])
        loss = loss * -1 if not targeted else loss
    else:
        raise ValueError('Wrong loss.')
    return loss.flatten()

def predict(x, model, batch_size, device, mean=[0], std=[1]):
    if isinstance(x, np.ndarray):
        x = np.floor(x * 255.0) / 255.0
        x = ((x - np.array(mean)[np.newaxis, :, np.newaxis, np.newaxis]) / np.array(std)[np.newaxis, :, np.newaxis, np.newaxis]).astype(np.float32)
        n_batches = math.ceil(x.shape[0] / batch_size)
        logits_list = []
        with torch.no_grad():
            for counter in range(n_batches):
                if verbose: print('predicting', counter, '/', n_batches, end='\r')
                x_curr = torch.as_tensor(x[counter * batch_size:(counter + 1) * batch_size], device=device)
                logits_list.append(model(x_curr).detach().cpu().numpy())
        logits = np.vstack(logits_list)
        return logits
    else:
        return model(x)

class Normalize_layer(nn.Module):
    def __init__(self, mean, std):
        super(Normalize_layer, self).__init__()
        self.mean = nn.Parameter(torch.Tensor(mean).unsqueeze(1).unsqueeze(1), requires_grad=False)
        self.std = nn.Parameter(torch.Tensor(std).unsqueeze(1).unsqueeze(1), requires_grad=False)
    def forward(self, input): return input.sub(self.mean).div(self.std)

class Model(nn.Module):
    def __init__(self, dataset, arch, norm, model_dir, device=device, batch_size=1000, n_in=0, n_out=0, do_softmax=False, **kwargs):
        super(Model, self).__init__()
        self.arch = arch
        self.dataset = dataset
        try: 
            self.cnn = getattr(torchvision.models, arch)(pretrained=True).to(device).eval()
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        except AttributeError: 
            if 'trs' in arch: 
                from resnet import resnet, Ensemble, Switch
                import sys
                models = []
                base_path = 'data/trs-checkpoint.pth.tar'
                for i in range(3):
                    checkpoint = torch.load(base_path + ".%d" % (i))
                    model = resnet(depth=20, num_classes=10)
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items(): new_state_dict[k[9:]] = v
                    model.load_state_dict(new_state_dict)
                    model.to(device).train() #eval() fuck it
                    models.append(model)
                self.cnn = Ensemble(models)
                self.cnn.to(device).eval()
            elif 'pni' in arch:
                from noisy_resnet_cifar import noise_resnet20
                self.cnn = torch.nn.Sequential(Normalize_layer(
                    [x / 255 for x in [125.3, 123.0, 113.9]], [x / 255 for x in [63.0, 62.1, 66.7]])
                    , noise_resnet20()).to(device)
                self.cnn.load_state_dict(torch.load('data/pni-checkpoint.pth.tar')['state_dict'])
            else: self.cnn = load_model(model_name=arch, dataset=dataset, threat_model=norm, model_dir=model_dir).to(device).eval()
            self.mean = [0]
            self.std = [1]
        
        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.do_softmax = do_softmax

        if n_in: 
            self.arch_ori = arch
            self.arch += '_InRND-%.2f' % n_in
        if n_out:
            self.arch_ori = arch
            self.arch += '_OutRND-%.2f' % n_out
        self.n_in = n_in
        self.n_out = n_out
        self.attractor_interval = 1000

    def forward_undefended(self, x): return predict(x, self.cnn, self.batch_size, self.device, self.mean, self.std)

    def forward(self, x):
        if not isinstance(x, np.ndarray): return predict(x, self.cnn, self.batch_size, self.device, self.mean, self.std)
        noise_in = np.random.normal(scale=self.n_in, size=x.shape)
        logits = predict(np.clip(x + noise_in, 0, 1), self.cnn, self.batch_size, self.device, self.mean, self.std)
        noise_out = np.random.normal(scale=self.n_out, size=logits.shape)
        if self.do_softmax: return softmax(logits + noise_out)
        return logits + noise_out


class AAALinear(nn.Module):
    def __init__(self, dataset, arch, norm, model_dir, 
        device=device, batch_size=1000, attractor_interval=4, reverse_step=1, num_iter=100, calibration_loss_weight=5, optimizer_lr=0.1, do_softmax=False, **kwargs):
        super(AAALinear, self).__init__()
        self.dataset = dataset
        try: 
            self.cnn = getattr(torchvision.models, arch)(pretrained=True).to(device).eval()
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        except AttributeError: 
            self.cnn = load_model(model_name=arch, dataset=dataset, threat_model=norm, model_dir=model_dir).to(device).eval()
            self.mean = [0] #if dataset != 'imagenet' else [0, 0, 0]
            self.std = [1] #if dataset != 'imagenet' else [1, 1, 1]
            self.cnn.to(device)
        
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.dev = 0.5
        self.optimizer_lr = optimizer_lr
        self.calibration_loss_weight = calibration_loss_weight
        self.num_iter = num_iter
        self.arch_ori = arch
        self.arch = '%s_AAAlinear-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)
        self.temperature = 1 # 2.08333 #
        self.do_softmax = do_softmax

    def set_hp(self, reverse_step, attractor_interval=6, calibration_loss_weight=5):
        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.calibration_loss_weight = calibration_loss_weight
        self.arch = '%s_AAAlinear-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)

    def forward_undefended(self, x): return predict(x, self.cnn, self.batch_size, self.device, self.mean, self.std)
    
    def get_tuned_temperature(self):
        t_dict = {
            'Standard': 2.08333,
            'resnet50': 1.1236,
            'resnext101_32x8d': 1.26582,
            'vit_b_16': 0.94,
            'wide_resnet50_2': 1.20482,
            'Rebuffi2021Fixing_28_10_cutmix_ddpm': 0.607,
            'Salman2020Do_50_2': 0.83,
            'Dai2021Parameterizing': 0.431,
            'Rade2021Helper_extra': 0.58
        }
        return t_dict.get(self.arch_ori, None)

    def temperature_rescaling(self, x_val, y_val, step_size=0.001):
        ts, eces = [], []
        ece_best, y_best = 100, 1
        y_pred = self.forward_undefended(x_val)
        for t in np.arange(0, 1, step_size):
            y_pred1 = y_pred / t
            y_pred2 = y_pred * t

            ts += [t, 1/t]
            ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
            eces += [ece1, ece2]
            if ece1 < ece_best: 
                ece_best = ece1
                t_best = t
            if ece2 < ece_best: 
                ece_best = ece2
                t_best = 1/t
            print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
            (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
            ece1 * 100, ece2 * 100,
            t_best, ece_best * 100))
        self.temperature = t_best

        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["font.family"] = "times new roman"
        plt.scatter(ts, eces, color='#9467bd')
        plt.xscale('log')
        plt.xlabel('temperature')
        plt.ylabel('ece on validation set')
        plt.savefig('demo/t-%s-%.4f.png' % (self.arch, self.temperature))
        plt.close()

    def temperature_rescaling_with_aaa(self, x_val, y_val, step_size=0.001):
        self.temperature = self.get_tuned_temperature()
        if self.temperature is not None: return

        ts, eces = [], []
        ece_best, y_best = 100, 1
        for t in np.arange(0, 1, step_size):
            self.temperature = t
            y_pred1 = self.forward(x_val)
            self.temperature = 1/t
            y_pred2 = self.forward(x_val)

            ts += [t, 1/t]
            ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
            eces += [ece1, ece2]
            if ece1 < ece_best: 
                ece_best = ece1
                t_best = t
            if ece2 < ece_best: 
                ece_best = ece2
                t_best = 1/t
            print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
            (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
            ece1 * 100, ece2 * 100,
            t_best, ece_best * 100))
        self.temperature = t_best

        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["font.family"] = "times new roman"
        plt.scatter(ts, eces, color='#9467bd')
        plt.xscale('log')
        plt.xlabel('temperature')
        plt.ylabel('ece on validation set')
        plt.savefig('demo/taaa-%s-%.4f.png' % (self.arch, self.temperature))
        plt.close()

    def forward(self, x):
        if isinstance(x, np.ndarray): 
            x = np.floor(x * 255.0) / 255.0
            x = ((x - np.array(self.mean)[np.newaxis, :, np.newaxis, np.newaxis]) / np.array(self.std)[np.newaxis, :, np.newaxis, np.newaxis]).astype(np.float32)
        else: 
            x = torch.floor(x * 255.0) / 255.0
            x = ((x - torch.as_tensor(self.mean, device=self.device)[None, :, None, None]) / torch.as_tensor(self.std, device=self.device)[None, :, None, None])
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []

        for counter in range(n_batches):
            with torch.no_grad():
                if verbose: print('predicting', counter, '/', n_batches, end='\r')
                x_curr = x[counter * self.batch_size:(counter + 1) * self.batch_size]
                if isinstance(x, np.ndarray): x_curr = torch.as_tensor(x_curr, device=self.device) 
                logits = self.cnn(x_curr)
            
            logits_ori = logits.detach()
            prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
            prob_max_ori = prob_ori.max(1)[0] ###
            value, index_ori = torch.topk(logits_ori, k=2, dim=1)
            #"""
            mask_first = torch.zeros(logits.shape, device=self.device)
            mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
            mask_second = torch.zeros(logits.shape, device=self.device)
            mask_second[torch.arange(logits.shape[0]), index_ori[:, 1]] = 1
            #"""
            
            margin_ori = value[:, 0] - value[:, 1]
            attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
            target = attractor - self.reverse_step * (margin_ori - attractor)
            diff_ori = (margin_ori - target)
            real_diff_ori = margin_ori - attractor
            #"""
            with torch.enable_grad():
                logits.requires_grad = True
                optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
                i = 0 
                los_reverse_rate = 0
                prd_maintain_rate = 0
                for i in range(self.num_iter):
                #while i < self.num_iter or los_reverse_rate != 1 or prd_maintain_rate != 1:
                    prob = F.softmax(logits, dim=1)
                    #loss_calibration = (prob.max(1)[0] - prob_max_ori).abs().mean()
                    loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
                    #loss_calibration = (prob - prob_ori).abs().mean()

                    value, index = torch.topk(logits, k=2, dim=1) 
                    margin = value[:, 0] - value[:, 1]
                    #margin = (logits * mask_first).max(1)[0] - (logits * mask_second).max(1)[0]

                    diff = (margin - target)
                    real_diff = margin - attractor
                    loss_defense = diff.abs().mean()
                    
                    loss = loss_defense + loss_calibration * self.calibration_loss_weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #i += 1
                    los_reverse_rate = ((real_diff * real_diff_ori) < 0).float().mean()
                    prd_maintain_rate = (index_ori[:, 0] == index[:, 0]).float().mean()
                    #print('%d, %.2f, %.2f' % (i, los_reverse_rate * 100, prd_maintain_rate * 100), end='\r')
                    #print('%d, %.4f, %.4f, %.4f' % (itre, loss_calibration, loss_defense, loss))
                logits_list.append(logits.detach().cpu())
                #print('main [los=%.2f, prd=%.2f], margin [ori=%.2f, tar=%.2f, fnl=%.2f], logits [ori=%.2f, fnl=%.2f], prob [tar=%.2f, fnl=%.2f]' % 
                    #(los_reverse_rate * 100, prd_maintain_rate * 100, 
                    #margin_ori[0], target[0], margin[0], logits_ori.max(1)[0][0], logits.max(1)[0][0], prob_max_ori[0], prob.max(1)[0][0]))
            #"""
            #logits_list.append(logits_ori.detach().cpu() / self.temperature)
        logits = torch.vstack(logits_list)
        if isinstance(x, np.ndarray): logits = logits.numpy()
        if self.do_softmax: logits = softmax(logits)
        return logits


class AAASine(nn.Module):
    def __init__(self, dataset, arch, norm, model_dir, 
        device=device, batch_size=1000, attractor_interval=6, reverse_step=1, num_iter=100, calibration_loss_weight=5, optimizer_lr=0.1, do_softmax=False, **kwargs):
        super(AAASine, self).__init__()
        self.dataset = dataset
        try: 
            self.cnn = getattr(torchvision.models, arch)(pretrained=True).to(device).eval()
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        except AttributeError: 
            self.cnn = load_model(model_name=arch, dataset=dataset, threat_model=norm, model_dir=model_dir).to(device).eval()
            self.mean = [0] #if dataset != 'imagenet' else [0, 0, 0]
            self.std = [1] #if dataset != 'imagenet' else [1, 1, 1]
            self.cnn.to(device)
        
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.dev = 0.5
        self.optimizer_lr = optimizer_lr
        self.calibration_loss_weight = calibration_loss_weight
        self.num_iter = num_iter
        self.arch_ori = arch
        self.arch = '%s_AAAsine-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)
        self.temperature = 1 # 2.08333 #
        self.do_softmax = do_softmax

    def set_hp(self, reverse_step, attractor_interval=6, calibration_loss_weight=5):
        self.attractor_interval = attractor_interval
        self.reverse_step = reverse_step
        self.calibration_loss_weight = calibration_loss_weight
        self.arch = '%s_AAAsine-Lr-%.1f-Ai-%d-Cw-%d' % (self.arch_ori, self.reverse_step, self.attractor_interval, self.calibration_loss_weight)

    def forward_undefended(self, x): return predict(x, self.cnn, self.batch_size, self.device, self.mean, self.std)
    
    def get_tuned_temperature(self):
        t_dict = {
            'Standard': 2.08333,
            'resnet50': 1.1236,
            'resnext101_32x8d': 1.26582,
            'vit_b_16': 0.94,
            'wide_resnet50_2': 1.20482,
            'Rebuffi2021Fixing_28_10_cutmix_ddpm': 0.607,
            'Salman2020Do_50_2': 0.83,
            'Dai2021Parameterizing': 0.431,
            'Rade2021Helper_extra': 0.58
        }
        return t_dict.get(self.arch_ori, None)

    def temperature_rescaling(self, x_val, y_val, step_size=0.001):
        ts, eces = [], []
        ece_best, y_best = 100, 1
        y_pred = self.forward_undefended(x_val)
        for t in np.arange(0, 1, step_size):
            y_pred1 = y_pred / t
            y_pred2 = y_pred * t

            ts += [t, 1/t]
            ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
            eces += [ece1, ece2]
            if ece1 < ece_best: 
                ece_best = ece1
                t_best = t
            if ece2 < ece_best: 
                ece_best = ece2
                t_best = 1/t
            print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
            (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
            ece1 * 100, ece2 * 100,
            t_best, ece_best * 100))
        self.temperature = t_best

        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["font.family"] = "times new roman"
        plt.scatter(ts, eces, color='#9467bd')
        plt.xscale('log')
        plt.xlabel('temperature')
        plt.ylabel('ece on validation set')
        plt.savefig('demo/t-%s-%.4f.png' % (self.arch, self.temperature))
        plt.close()

    def temperature_rescaling_with_aaa(self, x_val, y_val, step_size=0.001):
        self.temperature = self.get_tuned_temperature()
        if self.temperature is not None: return

        ts, eces = [], []
        ece_best, y_best = 100, 1
        for t in np.arange(0, 1, step_size):
            self.temperature = t
            y_pred1 = self.forward(x_val)
            self.temperature = 1/t
            y_pred2 = self.forward(x_val)

            ts += [t, 1/t]
            ece1, ece2 = ece_score(y_pred1, y_val), ece_score(y_pred2, y_val)
            eces += [ece1, ece2]
            if ece1 < ece_best: 
                ece_best = ece1
                t_best = t
            if ece2 < ece_best: 
                ece_best = ece2
                t_best = 1/t
            print('t-curr=%.3f, acc=%.2f, %.2f, ece=%.4f, %.4f, t-best=%.5f, ece-best=%.4f' % 
            (t, (y_pred1.argmax(1) == y_val.argmax(1)).mean() * 100, (y_pred2.argmax(1) == y_val.argmax(1)).mean() * 100, 
            ece1 * 100, ece2 * 100,
            t_best, ece_best * 100))
        self.temperature = t_best

        plt.rcParams["figure.dpi"] = 500
        plt.rcParams["font.family"] = "times new roman"
        plt.scatter(ts, eces, color='#9467bd')
        plt.xscale('log')
        plt.xlabel('temperature')
        plt.ylabel('ece on validation set')
        plt.savefig('demo/taaa-%s-%.4f.png' % (self.arch, self.temperature))
        plt.close()

    def forward(self, x):
        if isinstance(x, np.ndarray): 
            x = np.floor(x * 255.0) / 255.0
            x = ((x - np.array(self.mean)[np.newaxis, :, np.newaxis, np.newaxis]) / np.array(self.std)[np.newaxis, :, np.newaxis, np.newaxis]).astype(np.float32)
        else: 
            x = torch.floor(x * 255.0) / 255.0
            x = ((x - torch.as_tensor(self.mean, device=self.device)[None, :, None, None]) / torch.as_tensor(self.std, device=self.device)[None, :, None, None])
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []

        for counter in range(n_batches):
            with torch.no_grad():
                if verbose: print('predicting', counter, '/', n_batches, end='\r')
                x_curr = x[counter * self.batch_size:(counter + 1) * self.batch_size]
                if isinstance(x, np.ndarray): x_curr = torch.as_tensor(x_curr, device=self.device) 
                logits = self.cnn(x_curr)
            

            logits_ori = logits.detach()
            prob_ori = F.softmax(logits_ori / self.temperature, dim=1)
            prob_max_ori = prob_ori.max(1)[0] ###
            value, index_ori = torch.topk(logits_ori, k=2, dim=1)
            #"""
            mask_first = torch.zeros(logits.shape, device=self.device)
            mask_first[torch.arange(logits.shape[0]), index_ori[:, 0]] = 1
            mask_second = torch.zeros(logits.shape, device=self.device)
            mask_second[torch.arange(logits.shape[0]), index_ori[:, 1]] = 1
            #"""
            
            margin_ori = value[:, 0] - value[:, 1]
            attractor = ((margin_ori / self.attractor_interval + self.dev).round() - self.dev) * self.attractor_interval
            #target = attractor - self.reverse_step * (margin_ori - attractor)

            target = margin_ori - 0.7 * self.attractor_interval * torch.sin(
                (1 - 2 / self.attractor_interval * (margin_ori - attractor)) * torch.pi)
            diff_ori = (margin_ori - target)
            real_diff_ori = margin_ori - attractor

            with torch.enable_grad():
                logits.requires_grad = True
                optimizer = torch.optim.Adam([logits], lr=self.optimizer_lr)
                i = 0 
                los_reverse_rate = 0
                prd_maintain_rate = 0
                for i in range(self.num_iter):
                #while i < self.num_iter or los_reverse_rate != 1 or prd_maintain_rate != 1:
                    prob = F.softmax(logits, dim=1)
                    #loss_calibration = (prob.max(1)[0] - prob_max_ori).abs().mean()
                    loss_calibration = ((prob * mask_first).max(1)[0] - prob_max_ori).abs().mean() # better
                    #loss_calibration = (prob - prob_ori).abs().mean()

                    value, index = torch.topk(logits, k=2, dim=1) 
                    margin = value[:, 0] - value[:, 1]
                    #margin = (logits * mask_first).max(1)[0] - (logits * mask_second).max(1)[0]

                    diff = (margin - target)
                    real_diff = margin - attractor
                    loss_defense = diff.abs().mean()
                    
                    loss = loss_defense + loss_calibration * self.calibration_loss_weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    #i += 1
                    los_reverse_rate = ((real_diff * real_diff_ori) < 0).float().mean()
                    prd_maintain_rate = (index_ori[:, 0] == index[:, 0]).float().mean()
                    #print('%d, %.2f, %.2f' % (i, los_reverse_rate * 100, prd_maintain_rate * 100), end='\r')
                    #print('%d, %.4f, %.4f, %.4f' % (itre, loss_calibration, loss_defense, loss))
                logits_list.append(logits.detach().cpu())

        logits = torch.vstack(logits_list)
        if isinstance(x, np.ndarray): logits = logits.numpy()
        if self.do_softmax: logits = softmax(logits)
        return logits


class DENTModel(nn.Module):
    def __init__(self, dataset, arch, norm, model_dir, device=device, batch_size=1000, **kwargs):
        super(DENTModel, self).__init__()
        from robustbench.utils import load_model
        self.cnn = load_model(model_name=arch, dataset=dataset, threat_model=norm, model_dir=model_dir)
        self.arch = arch + '_DENT'
        self.cnn.to(device)
        self.dataset = dataset

        from dent import Dent
        self.cnn = Dent(self.cnn)

        self.batch_size = batch_size
        self.device = device
        self.loss = loss
        self.forward = lambda x: predict(x, self.cnn, self.batch_size, self.device)

    def forward_undefended(self, x): return predict(x, self.cnn, self.batch_size, self.device)


class ResNeXtDenoise101():
    def __init__(self, batch_size=100, **kwargs):
        from dfdmodels import nets
        import tensorflow as tf
        from tensorpack.tfutils import get_model_loader
        from tensorpack import TowerContext
        self.arch = 'resnext101_denoise'
        self.inputs = tf.placeholder(tf.float32, [None, 3, 224, 224])
        self.load_net_denoise(self.preprocess_input(self.inputs))
        self.loss = loss
        self.batch_size = batch_size
    
    def preprocess_input(self, x): return (x * 2 - 1.0)[:, ::-1, :, :]

    def load_net_denoise(self, inp):
        self.cnn = nets.ResNeXtDenoiseAllModel()
        with TowerContext('', is_training=False): logits = self.cnn.get_logits(inp)
        self.sess = tf.InteractiveSession()
        get_model_loader('data/X101-DenoiseAll.npz').init(self.sess)
        self.output = logits

    def __call__(self, x):
        x = np.floor(x * 255.0) / 255.0
        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        for counter in range(n_batches):
            if verbose: print('predicting', counter, '/', n_batches, end='\r')
            logits_list.append(self.sess.run(self.output, {self.inputs: x[counter * self.batch_size:(counter + 1) * self.batch_size]}))
        logits = np.vstack(logits_list)
        return logits