from __future__ import print_function
import os
import random

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40, ScanObjectNN
from model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import torch.nn.functional as F
from torch.autograd import Variable
from sampler import ImbalancedDatasetSampler
import math
from munch import Munch
import time
import json
from unlabeled_sampler import Unlabeled_ImbalancedDatasetSampler
import wandb
import torch.multiprocessing
import torch
from torch import Tensor
from supcon_loss import SupConLoss

torch.multiprocessing.set_sharing_strategy('file_system')

import time


def _init_():
    args.exp_name = args.exp_name + str(random.randint(0, 100000))
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')



def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss


def reduce_tensor(tensor, mean=True):
    return tensor


def consistency_loss(logits_s, logits_w, name='ce', p_cutoff=0.95, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    assert name == 'ce', 'must ce'
    pseudo_label = F.softmax(logits_w, dim=-1)
    max_probs, max_idx = torch.max(pseudo_label, axis=-1)
    mask = torch.greater_equal(max_probs, p_cutoff).double()
    select = torch.greater_equal(max_probs, p_cutoff).int()
    if use_hard_labels:
        masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
    else:
        print('must use hard label')
    return masked_loss.mean(), mask.mean(), select, max_idx



def nl_loss(pred_s, pred_w, k, p_cutoff):
    softmax_pred = F.softmax(pred_s, dim=-1)
    pseudo_label = F.softmax(pred_w, dim=-1)
    topk = pseudo_label.topk(k=k, dim=1, largest=True, sorted=True)[1]

    mask_k = torch.scatter(torch.ones_like(pseudo_label), 1, topk, torch.zeros_like(topk).float())
    mask_k_npl = torch.where((mask_k == 1) & (softmax_pred > p_cutoff ** 2), torch.zeros_like(mask_k), mask_k)
    loss_npl = (-torch.log(1 - softmax_pred + 1e-10) * mask_k_npl).sum(axis=1).mean()

    return loss_npl


def cal_topK(pred_s, pred_w, topk=(1,)):
    target_w = torch.argmax(pred_w, axis=-1)
    output = pred_s
    target = target_w

    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # pred: [k, num of batch]

    # [1, num of batch] -> [k, num_of_batch] : bool
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    for k in list(np.arange(topk[0], topk[1] + 1)):
        correct_k = correct[:k].reshape(-1).sum(0)
        acc_single = correct_k * (100.0 / batch_size)
        acc_parallel = reduce_tensor(acc_single)
        if acc_parallel > 99.99:
            return k


def train(args, io):

    criterion_supcon = SupConLoss(temperature=0.1)
    dataset_class = getattr(sys.modules['data'], args.dataset)

    num_workers = 0 if os.path.exists('C:\\Users\\SN_PAUL\\Desktop') else args.num_workers
    train_loader_labeled = DataLoader(
        dataset_class(partition='train', num_points=args.num_points, data_split='labeled', perceptange=args.perceptange, args=args),
        num_workers=num_workers,
        batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_loader_unlabeled = DataLoader(
        dataset_class(partition='train', num_points=args.num_points, data_split='unlabeled', perceptange=args.perceptange, args=args),
        num_workers=num_workers,
        batch_size=args.batch_size * args.unlabeled_ratio, shuffle=True, drop_last=True)

    test_loader = DataLoader(dataset_class(partition='test', num_points=args.num_points, args=args), num_workers=num_workers,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    validate_loader = DataLoader(dataset_class(partition='test', num_points=args.num_points, args=args), num_workers=num_workers,
                                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args, output_channels=args.num_classes).to(device)
    print(str(model))

    eval_model = Pct(args, output_channels=args.num_classes).to(device)  # ema model initialization
    print(str(eval_model))

    it = 0  # initializing the number of iteration
    ema_m = args.ema_m

    for param_q, param_k in zip(model.parameters(), eval_model.parameters()):
        param_k.data.copy_(param_q.detach().data)  # initialize
        param_k.requires_grad = False  # not update by gradient for eval_net

    eval_model.eval()

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss
    best_test_acc = 0
    epoch_counter = 0

    section_interation = args.epochs / args.section_size
    for current_section in range(int(section_interation)):
        np.random.seed()
        if current_section != 0:
            train_loader_labeled = DataLoader(
                dataset_class(partition='train', num_points=args.num_points, data_split='labeled', perceptange=args.perceptange, args=args),
                num_workers=num_workers, sampler=ImbalancedDatasetSampler(
                    dataset_class(partition='train', num_points=args.num_points, data_split='labeled', perceptange=args.perceptange, args=args),
                    args=args),
                batch_size=args.batch_size, drop_last=True)

            train_loader_unlabeled = DataLoader(
                dataset_class(partition='train', num_points=args.num_points, data_split='unlabeled', perceptange=args.perceptange, args=args),
                num_workers=num_workers, sampler=Unlabeled_ImbalancedDatasetSampler(
                    dataset_class(partition='train', num_points=args.num_points, data_split='unlabeled', perceptange=args.perceptange, args=args), args=args),
                batch_size=args.batch_size * args.unlabeled_ratio, drop_last=True)

        for epoch in range(args.section_size):  # loop over the dataset multiple times
            scheduler.step()
            train_loss = 0.0
            npl_loss = 0.0
            count = 0.0
            model.train()
            train_pred = []
            train_true = []
            idx = 0
            total_time = 0.0
            ulb_true = []
            ulb_pred = []
            for l_data, u_data in zip(train_loader_labeled, train_loader_unlabeled):
                # labeled and unlabeled data for each iteration
                start_time = time.time()
                loss_dict = {}

                data_unaug, data, data_strongaug, label, index, easy_mask_lb = l_data
                data_u_unaug, data_u, data_u_strongaug, label_u, index, easy_mask_ulb= u_data

                data_unaug, data, data_strongaug, label = data_unaug.to(device), data.to(device), data_strongaug.to(
                    device), label.to(device).squeeze()
                data_u_unaug, data_u, data_u_strongaug, label_u = data_u_unaug.to(device), data_u.to(
                    device), data_u_strongaug.to(device), label_u.to(device).squeeze()

                data_unaug = data_unaug.permute(0, 2, 1)
                data = data.permute(0, 2, 1)
                data_strongaug = data_strongaug.permute(0, 2, 1)

                data_u_unaug = data_u_unaug.permute(0, 2, 1)
                data_u = data_u.permute(0, 2, 1)
                data_u_strongaug = data_u_strongaug.permute(0, 2, 1)

                batch_size = data.size()[0]
                opt.zero_grad()

                logits_l_w, tokens_l_w = model(data)
                logits_l_s, tokens_l_s = model(data_strongaug)
                tokens_l_s = F.normalize(tokens_l_s, dim=1).unsqueeze(1)
                tokens_l_w = F.normalize(tokens_l_w, dim=1).unsqueeze(1)
                feature_l = torch.cat((tokens_l_s, tokens_l_w), dim=1)

                logits_u, tokens_u = model(data_u)
                logits_u_s, tokens_u_s = model(data_u_strongaug)
                tokens_u_s = F.normalize(tokens_u_s, dim=1).unsqueeze(1)
                tokens_u = F.normalize(tokens_u, dim=1).unsqueeze(1)
                feature_u = torch.cat((tokens_u_s, tokens_u), dim=1)

                # supervised loss 1
                labeled_cross_entropy_loss = criterion(logits_l_w, label) + criterion(logits_l_s, label)
                # supcon loss 2
                supcon_loss = criterion_supcon(feature_l, label)

                pseudo_label = torch.softmax(logits_u.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                flex_threshold = torch.zeros(batch_size * args.unlabeled_ratio).to(device)

                if epoch + args.section_size * current_section == 0:
                    mask = max_probs.ge(0.2).float()
                else:
                    z = open('checkpoints/' + args.exp_name + '/' + "dict_avgconf.txt", "r")
                    k = z.read()
                    dict_avgconf = json.loads(k)
                    z.close()

                    if args.threshold_styl == 'confid':
                        for current_label in dict_avgconf:
                            current_label = int(current_label)

                            if dict_avgconf['%d' % current_label] / (2 - dict_avgconf['%d' % current_label]) < 0.2:
                                flex_threshold[targets_u == current_label] = 0.2

                            elif dict_avgconf['%d' % current_label] / (2 - dict_avgconf['%d' % current_label]) > 0.8:
                                flex_threshold[targets_u == current_label] = 0.8
                            else:
                                learning_effect = dict_avgconf['%d' % current_label]
                                flex_threshold[targets_u == current_label] = learning_effect / (2 - learning_effect)

                    elif args.threshold_styl == 'confid_flex':
                        for current_label in dict_avgconf:
                            current_label = int(current_label)
                            learning_effect = dict_avgconf['%d' % current_label]
                            flex_threshold[targets_u == current_label] = learning_effect / (2 - learning_effect)

                    elif args.threshold_styl == 'flex_match':
                        for current_label in dict_avgconf:
                            current_label = int(current_label)
                            flex_threshold[targets_u == current_label] = dict_avgconf['%d' % current_label]

                mask = max_probs.ge(flex_threshold).float()
                ulb_true.extend(label_u.cpu().numpy())
                ulb_pred.extend(targets_u.detach().cpu().numpy())

                mask_label = torch.ones(mask.shape[0])
                mask_label = Variable(mask_label).to(device)
                # historical loss calc
                it_loss = (F.cross_entropy(logits_u_s, targets_u, reduction='none'))

                for x_l in it_loss:
                    if torch.isnan(x_l):
                        print('nan')

                # unsup loss calc 1 (fixmatch loss)
                pt_pseudo_ce_loss = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask)
                pt_pseudo_ce_loss = pt_pseudo_ce_loss.mean()

                # unsup loss calc 2
                unsup_contrastive_loss = criterion_supcon(feature_u)

                train_loader_unlabeled.dataset.update_loss(index, it_loss.cpu())  # update historical loss

                if epoch < args.fake_epoch:
                    full_match_loss = torch.tensor(0)
                    loss_npl = torch.tensor(0)
                else:
                    unsup_loss, mask, select, pseudo_lb = consistency_loss(logits_u_s, logits_u, 'ce',
                                                                           args.p_cutoff,
                                                                           use_hard_labels=args.hard_label)
                    k_value = cal_topK(logits_u_s.detach(), logits_u.detach(), topk=(2, args.num_classes))
                    loss_npl = nl_loss(logits_u_s, logits_u.detach(), k_value,
                                                   args.p_cutoff)

                    full_match_loss = args.nl_lambda * loss_npl

                loss = args.lambda_ce * labeled_cross_entropy_loss + args.u_lambda * pt_pseudo_ce_loss + args.supcon_lambda * supcon_loss + args.unsupcon_lambda * unsup_contrastive_loss + full_match_loss

                loss_dict['labeled_cross_entropy_loss'] = labeled_cross_entropy_loss.item()
                loss_dict['pt_pseudo_ce_loss'] = pt_pseudo_ce_loss.item()
                loss_dict['supcon_loss'] = supcon_loss.item()
                loss_dict['unsup_contrastive_loss'] = unsup_contrastive_loss.item()
                loss_dict['full_match_loss'] = full_match_loss.item()
                loss_dict['loss_npl'] = loss_npl.item()
                
                loss_dict['loss'] = loss.item()
                if not os.path.exists('/scratch/a/'):
                    wandb.log(loss_dict)

                loss.backward()
                opt.step()
                end_time = time.time()
                total_time += (end_time - start_time)

                preds = logits_l_w.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                npl_loss += loss_npl.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
                idx += 1
                it += 1

                with torch.no_grad():  # update the eval_model at each iteration
                    for param_train, param_eval in zip(model.parameters(), eval_model.parameters()):
                        alpha = min(1 - 1 / (it + 1), ema_m)
                        param_eval.copy_(param_eval * alpha + param_train.detach() * (1 - alpha))

                    for buffer_train, buffer_eval in zip(model.buffers(), eval_model.buffers()):
                        buffer_eval.copy_(buffer_train)

            epoch_counter += 1
            if epoch_counter > args.masking_epoch:
                train_loader_unlabeled.dataset.update_mask()

            print('train total time is', total_time)
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            outstr = ('Train %d, loss: %.4f, train acc: %.4f, train avg acc: %.4f,'
                      'ce loss: %.4f, pt pseudo ce loss: %.4f,'
                      'supcon loss: %.4f, unsupcon loss: %.4f'
                      'npl loss: %.4f') % (epoch,
                                                            train_loss * 1.0 / count,
                                                            metrics.accuracy_score(
                                                                train_true, train_pred),
                                                            metrics.balanced_accuracy_score(
                                                                train_true, train_pred),
                                                            labeled_cross_entropy_loss.item(),
                                                            pt_pseudo_ce_loss.item(),
                                                            supcon_loss.item(), unsup_contrastive_loss.item(),
                                                            npl_loss * 1.0 / count)
            io.cprint(outstr)


            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            total_time = 0.0
            for data, label, index, easy_mask in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                start_time = time.time()
                logits, tokens1 = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            print('test total time is', total_time)
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            if not os.path.exists('/scratch/a/'):
                wandb.log({"test_acc": test_acc, "per_class_avg_acc": avg_per_class_acc})
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
                epoch + args.section_size * current_section,
                test_loss * 1.0 / count,
                test_acc,
                avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            torch.save(model.state_dict(), 'checkpoints/%s/models/latest_model.t7' % args.exp_name)

            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            total_time = 0.0
            for data, label, index, _ in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                start_time = time.time()
                logits, tokens1 = eval_model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            print('test total time is', total_time)
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            if not os.path.exists('/scratch/a/'):
                wandb.log({"eval model test_acc": test_acc, "per_class_avg_acc": avg_per_class_acc})
            outstr = 'eval model Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
                epoch + args.section_size * current_section,
                test_loss * 1.0 / count,
                test_acc,
                avg_per_class_acc)
            io.cprint(outstr)

            test_true = []
            test_pred = []
            test_logits = []
            test_sec_max = []
            model.eval()
            for data, label, index, easy_mask in validate_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                logits, tokens = model(data)
                m = nn.Softmax(dim=1)
                output = m(logits)
                sec_max, _ = torch.torch.sort(output, -1, descending=True)
                max_logits, preds = output.max(dim=1)
                sec_max_logits = sec_max[:, 1]
                if args.test_batch_size == 1:
                    test_true.append([label.cpu().numpy()])
                    test_pred.append([preds.detach().cpu().numpy()])
                    test_logits.append([max_logits.detach().cpu().numpy()])
                    test_sec_max.append([sec_max_logits.detach().cpu().numpy()])


                else:
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
                    test_logits.append(max_logits.detach().cpu().numpy())
                    test_sec_max.append(sec_max_logits.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_logits = np.concatenate(test_logits)
            print(test_logits.shape)

            labeled_data = np.arange(0, 40)
            dict_1 = Munch()
            dict_secmax = Munch()
            dict_high_conf_index = Munch()
            dict_ave_validate_conf = Munch()

            for i in labeled_data:
                dict_1['%d' % i] = 0
                dict_secmax['%d' % i] = 0

            for current_label in labeled_data:
                current_label_index = np.where(test_pred == current_label)
                current_label_logit = test_logits[current_label_index]
                if len(current_label_index[0]) != 0:
                    dict_ave_validate_conf['%d' % current_label] = np.sum(current_label_logit) / len(
                        current_label_index[0])
                else:
                    dict_ave_validate_conf['%d' % current_label] = 0.2

            for current_label in range(40):
                label_pos = np.where(test_pred == current_label)

                if dict_ave_validate_conf['%d' % current_label] > 0.8:
                    high_conf_label_pos = label_pos[0][
                        np.where(test_logits[label_pos] > dict_ave_validate_conf['%d' % current_label])]
                    dict_high_conf_index['%d' % current_label] = high_conf_label_pos.tolist()



                elif dict_ave_validate_conf['%d' % current_label] < 0.8:
                    low_conf_label_pos = label_pos[0][
                        np.where(test_logits[label_pos] > dict_ave_validate_conf['%d' % current_label])]

                    dict_high_conf_index['%d_low' % current_label] = low_conf_label_pos.tolist()

            f = open('checkpoints/' + args.exp_name + '/' + "dict_highconf_indx.txt", "w")
            js = json.dumps(dict_high_conf_index)
            f.write(js)
            f.close()

            f = open('checkpoints/' + args.exp_name + '/' + "dict_avgconf.txt", "w")
            js = json.dumps(
                dict_ave_validate_conf
            )
            f.write(js)
            f.close()

            f = open('checkpoints/' + args.exp_name + '/' + "dict_logits.txt", "w")
            js = json.dumps(test_logits.tolist())
            f.write(js)
            f.close()

            f = open('checkpoints/' + args.exp_name + '/' + "current_epoch.txt", "w")
            js = json.dumps(epoch + args.section_size * current_section)
            f.write(js)
            f.close()


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args).to(device)
    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits, tokens = model(data)
        preds = logits.max(dim=1)[1]
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)

    if not os.path.exists('/scratch/a/'):
        wandb.log({"test_acc": test_acc, "per_class_avg_acc": avg_per_class_acc})

    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--section_size', type=int, default=50, metavar='N',
                        help='how many epoches a section has ')
    parser.add_argument('--unlabeled_ratio', type=int, default=4, metavar='N',
                        help='unlabeled labeled ratio ')
    parser.add_argument('--exp_name', type=str, default='train', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--fake_epoch', type=int, default=5,
                        help='num of epochs to ignore fullmatch loss')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--threshold_styl', type=str, default='confid', metavar='N',
                        help='threshold style')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--masking_epoch', type=float, default=50, help='epoch for begining masking')
    parser.add_argument('--perceptange', type=float, default=10, help='percentage of labeled data')
    parser.add_argument('--unsup_method', type=str, default='none', metavar='N',
                        help='threshold style')
    parser.add_argument('--p_cutoff', type=float, default=0.95, help='fixmatch style pseudo label cutoff')
    parser.add_argument('--hard_label', type=bool, default=True)
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    parser.add_argument('--u_lambda', type=float, default=1.0)
    parser.add_argument('--lambda_ce', type=float, default=1.0)
    parser.add_argument('--unsupcon_lambda', type=float, default=1.0)
    parser.add_argument('--supcon_lambda', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='ModelNet40', choices=['ModelNet40', 'ScanObjectNN'])
    parser.add_argument('--aug_strength', type=float, default=4)
    parser.add_argument('--nl_lambda', type=float, default=1.0)


    args = parser.parse_args()
    if args.dataset == 'ModelNet40':
        args.num_classes = 40
    elif args.dataset == 'ScanObjectNN':
        args.num_classes = 15
    else:
        raise Exception("Dataset not properly  implemented")

    if not os.path.exists('/scratch/a/'):
        wandb.init(project='Confid_SSL', name=args.exp_name)

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
