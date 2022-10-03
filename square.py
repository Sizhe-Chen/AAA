import argparse
import time
import numpy as np
import os
from copy import deepcopy
import torch
import random
import PIL.Image as Image
import cv2

from attacker import QueryNet
from victim import *
from utils import *


def attack(model, x, y, corr, y_pred, y_undefended, l2, eps, n_iters, stop_iters, p_init, num_s, batch_size, targeted, loss_type, resume_path, plot):
    # 1st query: with clean samples
    #corr = y_pred.argmax(1) == y.argmax(1) if not targeted else y_undefended.argmax(1) == y.argmax(1)
    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    n_features = c * h * w
    n_ex_total = x.shape[0]
    ece = ece_score(y_pred, y)
    y_pred_all = deepcopy(y_pred)
    y_test_all = deepcopy(y)
    x, y, y_pred = x[corr], y[corr], y_pred[corr]
    margin_min = model.loss(y, y_pred, targeted, loss_type='margin_loss')
    
    # setup directories
    method = ('QueryNet_' if num_s else '')
    result_path = get_time() + (('_L2-%.1f' % eps) if l2 else ('_Linf-%d' % (eps*255))) + \
        f'_{model.dataset}_{model.arch}_' + method + ('targeted_' if targeted else 'untargeted_') + loss_type
    logger = LoggerUs(result_path)
    log = Logger('{}/{}.log'.format(result_path, 'log'))
    log.reset_path(result_path + '/log.log')
    print(result_path)
    process_path = result_path + '/var'
    if resume_path is None: log.print('{}: acc={:.2%}, ece={:.2f}, avg_margin={:.2f}'.format(1, np.mean(corr), ece * 100, np.mean(margin_min)))

    # setup attackers
    sampler = DataManager(x, y_pred, eps, result_dir=process_path, loss_init=model.loss(y=y, logits=y_pred, targeted=False, loss_type='margin_loss'))
    querynet = QueryNet(sampler, model.arch, ['DenseNet121', 'ResNet50', 'DenseNet169', 'ResNet101', 'DenseNet201'][:num_s], # 'VGG16'
        use_horizontal_info=False, use_random_info=True, nas=True, linfty=not l2, eps=eps, batch_size=batch_size, iter_square_s=0) # -100 if pre else 0
    np.save(process_path + "/corr.npy", corr)
    
    # square vertical stripes https://arxiv.org/pdf/1912.00049.pdf
    if l2:
        delta_square = np.zeros(x.shape)
        s = h // 5
        sp_init = (h - s * 5) // 2
        center_h = sp_init + 0
        for counter in range(h // s):
            center_w = sp_init + 0
            for counter2 in range(w // s):
                delta_square[:, :, center_h:center_h + s, center_w:center_w + s] += querynet.meta_pseudo_gaussian_pert(s).reshape(
                    [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
                center_w += s
            center_h += s
        delta = delta_square#delta_nobox + delta_square
        x_best = np.clip(x + delta / np.sqrt(np.sum(delta ** 2, axis=(1, 2, 3), keepdims=True)) * eps, 0, 1)
    else:
        delta_square = np.random.choice([-eps, eps], size=[x.shape[0], c, 1, w])
        delta = delta_square#delta_nobox + delta_square
        x_best = np.clip(x + delta, min_val, max_val) 
    logits = model(x_best)  # !!!

    y_pred = deepcopy(logits)
    y_pred_all[corr] = y_pred

    margin_min = model.loss(y, logits, targeted, loss_type='margin_loss')
    ce_min = model.loss(y, logits, targeted, loss_type='cross_entropy')
    acc = (margin_min > 0.0).sum() / n_ex_total
    acc_corr = (margin_min > 0.0).sum() / corr.sum()
    if resume_path is None: log.print('{}: acc={:.2%}, acc_corr={:.1%}, ece={:.2f}, avg#q={:.2f}, avg#q_all={:.2f}, med#q={:.0f}, med#q_all={:.0f}, avg_margin={:.2f}, buff={:.0f}, eps={:.1f}, {:.2f}s'.
        format(2, acc, acc_corr, ece_score(y_pred_all, y_test_all) * 100, 2, 2, 2, 2, np.mean(margin_min), 0, eps*(1 if l2 else 255), 0))

    # setup attackers
    querynet.sampler.update_square(x_best, logits, margin_min, logger, targeted=False)
    def get_surrogate_loss(srgt, x_adv, y_ori): 
        s_inf_batch_size = batch_size * 8
        if x_adv.shape[0] <= s_inf_batch_size:
            return model.loss(y_ori, srgt(torch.Tensor(x_adv)).cpu().detach().numpy(), targeted, loss_type='margin_loss')
        batch_num = int(x_adv.shape[0]/s_inf_batch_size)
        if s_inf_batch_size * batch_num != int(x_adv.shape[0]): batch_num += 1
        loss_value = model.loss(y_ori[:s_inf_batch_size], srgt(torch.Tensor(x_adv[:s_inf_batch_size])).cpu().detach().numpy(), targeted, loss_type='margin_loss')
        for i in range(batch_num-1):
            new_loss_value = model.loss(y_ori[s_inf_batch_size*(i+1):s_inf_batch_size*(i+2)], 
                                        srgt(torch.Tensor(x_adv[s_inf_batch_size*(i+1):s_inf_batch_size*(i+2)])).cpu().detach().numpy(), 
                                        targeted, loss_type='margin_loss')
            loss_value = np.concatenate((loss_value, new_loss_value), axis=0)
            del new_loss_value
        return loss_value

    # resume attack
    if resume_path is None:
        i_iter = 0
        time_start = time.time()
        metrics = np.zeros([n_iters, 7])
        n_queries = np.ones(x.shape[0]) * 2#(2 if pre else 3)
    else:
        x_best = np.load(resume_path + '/var/x_best.npy')
        margin_min = np.load(resume_path + '/var/margin_min.npy')
        metrics = np.load(resume_path + '/var/metrics.npy')
        n_queries = np.load(resume_path + '/var/n_queries.npy')
        i_iter = querynet.load(resume_path) 
        time_start = time.time() - metrics[i_iter, -1]
        log.print('resumed from ' + resume_path)
    
    
    if plot:
        vis_path = result_path + '/vis'
        os.makedirs(vis_path, exist_ok=True)
        vis_path_image = vis_path + '/0.png'
        plot_curve(y_pred_all, y_undefended, y_test_all, model.attractor_interval, vis_path_image, 0)
        img = cv2.imread(vis_path_image)
        fps = 10
        size = (img.shape[1], img.shape[0])
        video = cv2.VideoWriter(vis_path + "/video.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size, True)
        video.write(cv2.resize(img, size))
    
    # begin to attack iteratively
    attacker_directions = np.ones(x.shape[0], dtype=bool)
    while n_queries.max() < stop_iters and acc != 0:
        # only handle unsuccessful adverarial examples
        idx_to_fool = margin_min > 0 
        x_curr, x_best_curr, y_curr, margin_min_curr = x[idx_to_fool], x_best[idx_to_fool], y[idx_to_fool], margin_min[idx_to_fool]
        ce_min_curr, ad_curr = ce_min[idx_to_fool], attacker_directions[idx_to_fool]
        
        x_new, x_new_index = querynet.forward(x_curr, x_best_curr, y_curr, get_surrogate_loss,
            min_val=min_val, max_val=max_val, p=p_selection(p_init, i_iter, n_iters), targeted=targeted)

        # query
        logits = model(x_new) 
        margin = model.loss(y_curr, logits, targeted, loss_type='margin_loss')
        ce = model.loss(y_curr, logits, targeted, loss_type='cross_entropy')

        #idx_improved = (ce < ce_min_curr) if loss_type == 'ce' else (margin < margin_min_curr)
        idx_improved_down = (ce < ce_min_curr) if loss_type == 'ce' else (margin < margin_min_curr) # down
        idx_improved_up = (margin > margin_min_curr) + ((margin - margin_min_curr) < -3) # up
        idx_improved_bi = np.where(ad_curr, margin < margin_min_curr, (margin > margin_min_curr) + ((margin - margin_min_curr) < -3))
        ad_tmp = attacker_directions[idx_to_fool]
        ad_tmp[(margin - margin_min_curr) > 3] = 0
        attacker_directions[idx_to_fool] = ad_tmp
        
        if loss_type == 'up': idx_improved = idx_improved_up
        elif loss_type == 'bi': idx_improved = idx_improved_bi
        else: idx_improved = idx_improved_down 


        ce_min[idx_to_fool] = idx_improved * ce + ~idx_improved * ce_min_curr
        margin_min[idx_to_fool] = idx_improved * margin + ~idx_improved * margin_min_curr
        y_pred[idx_to_fool] = idx_improved[:, np.newaxis] * logits + ~idx_improved[:, np.newaxis] * y_pred[idx_to_fool]
        y_pred_all[corr] = y_pred
        
        if plot:
            vis_path_image = vis_path + '/%d.png' % i_iter
            plot_curve(y_pred_all, y_undefended, y_test_all, model.attractor_interval, vis_path_image, i_iter)
            video.write(cv2.resize(cv2.imread(vis_path_image), size))

        idx_improved = np.reshape(idx_improved, [-1, *[1] * len(x.shape[:-1])])
        x_best[idx_to_fool] = idx_improved * x_new + ~idx_improved * x_best_curr
        n_queries[idx_to_fool] += 1
        i_iter += 1
        
        attacker_authority, attacker_selected = querynet.backward(idx_improved, x_new_index, 
            img_adv=x_new, lbl_adv=logits, loss=margin, logger=logger, targeted=False)
        if x_new_index is not None: log.print(
                'EvalWeight   ' + ' '.join([('%.3f' % x) for x in attacker_authority if x != 0]) + ' ' * 30 + '\n' + \
                'ChosenRate   ' + ' '.join([('%.3f' % x) for x in attacker_selected  if x != 0]))
    
        # stats
        acc = (margin_min > 0.0).sum() / n_ex_total
        acc_corr = (margin_min > 0.0).sum() / corr.sum()
        mean_nq, mean_nq_ae = np.mean(n_queries), np.mean(n_queries[margin_min <= 0]), 
        median_nq, median_nq_ae = np.median(n_queries), np.median(n_queries[margin_min <= 0])
        log.print('{}: acc={:.2%}, acc_corr={:.1%}, ece={:.2f}, avg#q={:.2f}, avg#q_all={:.2f}, med#q={:.0f}, med#q_all={:.0f}, avg_margin={:.2f}, buff={:.0f}, eps={:.1f}, {:.2f}s'.
            format(i_iter + 2, acc, acc_corr, ece_score(y_pred_all, y_test_all) * 100, mean_nq_ae, mean_nq, median_nq_ae, median_nq, np.mean(margin_min), x.shape[0], eps*(1 if l2 else 255), time.time() - time_start))
        
        # save for resume
        metrics[i_iter] = [acc, acc_corr, mean_nq, mean_nq_ae, median_nq, margin_min.mean(), time.time() - time_start]
        np.save(process_path + '/metrics.npy', metrics)
        np.save(process_path + '/margin_min.npy', margin_min)
        np.save(process_path + '/n_queries.npy', n_queries)
        if h != 224 or resume_path == 'allowed':
            np.save(process_path + '/x_best.npy', x_best)
            querynet.save(i_iter)
    #log.print('ece_adv={:.5f}'.format(ece_score(model(x_best[margin_min > 0]), y[margin_min > 0])))
    if plot:
        video.release()
        cv2.destroyAllWindows()


def plot_curve(logits, y_undefended, y_test, attractor_interval, save_path, i_iter, n_division=100):
    plt.rcParams["figure.figsize"] = (10.0, 5.0)
    plt.rcParams["figure.dpi"] = 500
    plt.rcParams["font.family"] = "times new roman"
    plt.rcParams["font.size"] = 18
    n_sample_per_division = int(y_undefended.shape[0] / n_division)
    prob_corr = (y_undefended * y_test).max(1)
    prob_ori_index = np.argsort(prob_corr)

    def _plot(ls, label, color, linestyle):
        values = []
        for i in range(n_division):
            values.append(np.mean(ls[prob_ori_index[i * n_sample_per_division: (i+1) * n_sample_per_division]]))
        plt.plot(values, label=label, color=color, linestyle=linestyle)
        return min(values)-3, max(values)+3

    prob_margin = prob_corr - (y_undefended * ~y_test).max(1)
    min_y, max_y = _plot(prob_margin, 'Undefended margin loss (clean data)', '#ff7f0e', 'solid')
    #_plot(prob_corr, 'Undefended logits (largest, clean data)', '#ff7f0e', 'dotted')

    prob_adv = (logits * y_test).max(1)
    _plot(prob_adv - (logits * ~y_test).max(1), 'AAA-defended margin loss (adv data)', '#2ca02c', 'solid')
    #_plot(prob_adv, 'AAA-Defended (largest, adversarial data)', '#2ca02c', 'dotted')
    
    plt.ylim(min_y, max_y)
    is_first = True
    for i in range(-10, 10):
        attractor = attractor_interval * (i + 0.5)
        if attractor < min_y or attractor > max_y: continue
        if is_first:
            is_first = False
            label = 'loss attractor'
        else: label = ''
        plt.plot([attractor] * n_division, label=label, color='grey', linestyle='dotted')

    plt.legend(loc='lower right')
    plt.title('Query Iteration %d' % i_iter)
    plt.savefig(save_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--model', default='Standard', type=str, help='model name in robustbench or torchvision')
    parser.add_argument('--dataset', default='cifar10', type=str, help='cifar10 / imagenet')
    parser.add_argument('--defense', default=None, type=str, help='AAA / inRND / DENT')
    parser.add_argument('--l2', action='store_true', help='perform l2 attack')
    parser.add_argument('--plot', action='store_true', help='plot image')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number')
    parser.add_argument('--loss', type=str, default='margin', help='margin / ce')
    parser.add_argument('--model_dir', type=str, default='rbmodels', help='dirs for robustbench models')
    parser.add_argument('--n_ex', type=int, default=10000, help='Number of test ex to test on.')
    parser.add_argument('--num_s', type=int, default=0, help='Number of surrogates for QueryNet attack.')
    parser.add_argument('--p', type=float, default=0.05, help='Probability of changing a coordinate, Linf standard: 0.05, L2 standard: 0.1. But robust models require higher p.')
    parser.add_argument('--eps', type=float, default=8, help='Radius of the Lp ball.')
    
    parser.add_argument('--num_sample_tune', type=int, default=1000, help='Number of test ex to test on.')
    parser.add_argument('--lr', type=float, default=1, help='reverse step size for AAA model')
    parser.add_argument('--attractor_interval', type=float, default=6, help='margin loss attractor interval for AAA model')
    parser.add_argument('--calibration_loss_weight', type=float, default=5, help='weight for maintaining probability score for AAA')
    parser.add_argument('--aaa_iter', type=int, default=100, help='number of iterations to modify logits in AAA')
    parser.add_argument('--aaa_optimizer_lr', type=float, default=0.1, help='learning rate to optimize logits by Adam')
    
    parser.add_argument('--stop_iters', type=int, default=2500)
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128) # 128
    parser.add_argument('--resume_path', type=str, default=None, help='Path to restore attack')
    args = parser.parse_args()

    args.p = 0.3 if args.model != 'Standard' else 0.05
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert not (args.dataset == 'imagenet' and args.eps == 8)
    return args


def load_model(args):
    if args.model == 'resnext101_denoise': victimModel = ResNeXtDenoise101
    elif args.defense is None or args.defense == 'inRND' or args.defense == 'outRND': victimModel = Model
    elif args.defense == 'AAALinear':  victimModel = AAALinear
    elif args.defense == 'DENT': victimModel = DENTModel
    elif args.defense == 'AAASine': victimModel = AAASine
    else: raise NotImplementedError

    return victimModel(
        dataset=args.dataset, 
        arch=args.model, 
        norm='L2' if args.l2 else 'Linf', 
        device=torch.device('cuda:0'), 
        batch_size=args.batch_size,
        model_dir=args.model_dir,
        do_softmax=args.loss == 'prob',
        
        n_in=(0.02 if ((args.model == 'Standard' and args.dataset == 'cifar10') or ('Salman2020Do' not in args.model and args.dataset == 'imagenet')) else 0.05) if (args.defense == 'inRND') else 0,
        n_out=(1 if args.model == 'Standard' else 0.3) if (args.defense == 'outRND') else 0,

        attractor_interval=args.attractor_interval, 
        reverse_step=args.lr,
        calibration_loss_weight=args.calibration_loss_weight,
        num_iter=args.aaa_iter,
        optimizer_lr=args.aaa_optimizer_lr
        )


def load_data(dataset, n_ex, model):
    if dataset == 'cifar10':    x_test, y_test = load_cifar10(n_ex)
    elif dataset == 'imagenet': x_test, y_test = load_imagenet(n_ex, model)
    return x_test, y_test


def prepare_for_attack():
    args = parse_args()
    model = load_model(args)
    x_test, y_test = load_data(args.dataset, args.n_ex, model)
    if 'AAAR' in model.arch: 
        if args.dataset == 'imagenet': 
            x_val = np.load('data/imagenet_tune_imgs.npy').astype(np.float32) / 255
            y_val = dense_to_onehot(np.load('data/imagenet_tune_lbls.npy'), 1000)
            model.temperature_rescaling_with_aaa(x_val, y_val)
        else: model.temperature_rescaling_with_aaa(x_test[:args.num_sample_tune], y_test[:args.num_sample_tune])
    return args, model, x_test, y_test, model(x_test)


if __name__ == '__main__':
    args, model, x_test, y_test, y_pred = prepare_for_attack()
    attack(
        model=model, 
        x=x_test, 
        y=dense_to_onehot(y_test.argmax(1), n_cls=y_test.shape[1]) if not args.targeted else random_classes_except_current(y_test), 
        corr=y_pred.argmax(1) == y_test.argmax(1),
        y_pred=y_pred, 
        y_undefended=model.forward_undefended(x_test),
        l2=args.l2, 
        eps=args.eps if args.l2 else (args.eps/255), 
        n_iters=args.n_iters, 
        stop_iters=args.stop_iters, 
        p_init=args.p,
        num_s=args.num_s, 
        batch_size=args.batch_size, 
        targeted=args.targeted,
        loss_type=args.loss,
        resume_path=args.resume_path,
        plot=args.plot
        )