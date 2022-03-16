import torch
import torch.nn as nn
import numpy as np
import os
import datetime
from tqdm import tqdm
from sklearn.metrics import *
import matplotlib.pyplot as plt

def cnn_train(cv_idx, train_loader, val_loader, criterion, cnn_model, cnn_opt, args, settings):
    Acc, Kappa = [], []
    W_f1, R_f1, N1_f1, N2_f1, N3_f1, N4_f1, Mean = [], [], [], [], [], [], []
    perfsum = 0

    for epoch in range(args.cnn_epoch):
        pred_list_tr = []
        true_list_tr = []
        t_loss = 0
        print('Epoch', epoch + 1)
        for idx, data in enumerate(tqdm(train_loader)):
            cnn_model.train()
            train_x, train_y = data
            train_x = train_x.unsqueeze(1).permute(0, 2, 1, 3).contiguous()
            b, _, s, _ = train_x.shape
            train_y = train_y.long()
            if settings['use_cuda']:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            cnn_opt.zero_grad()
            train_output = cnn_model(train_x, pretrain=True)
            train_y = train_y
            if (epoch < args.cnn_epoch // 3 or args.default_ce):
                train_l = criterion(train_output.view(-1, settings['n_classes']), train_y.view(-1))
            else:
                train_l = criterion(train_output.view(-1, settings['n_classes']), train_y.view(-1), f1_confusion)
            t_loss += train_l.item()
            train_l.backward()
            cnn_opt.step()

            expected_train_y = torch.argmax(torch.softmax(train_output, dim=-1), dim=-1)
            true_list_tr.append(train_y.view(-1).detach().cpu().numpy())
            pred_list_tr.append(expected_train_y.view(-1).detach().cpu().numpy())
        true_list_tr = np.array(np.concatenate(true_list_tr))
        pred_list_tr = np.array(np.concatenate(pred_list_tr))

        t_acc = accuracy_score(true_list_tr, pred_list_tr)
        t_f1 = f1_score(true_list_tr, pred_list_tr, average='macro')
        t_kappa = cohen_kappa_score(true_list_tr, pred_list_tr)
        t_confusion = confusion_matrix(true_list_tr, pred_list_tr)
        try:
            t_cls_rpt = classification_report(true_list_tr, pred_list_tr, zero_division=0,
                                              target_names=['W', 'N1', 'N2', 'N3', 'R'], output_dict=True)
        except:
            t_cls_rpt = classification_report(true_list_tr, pred_list_tr, zero_division=0,
                                              target_names=['W', 'N1', 'N2', 'N3', 'N4', 'R'], output_dict=True)

        t_loss = t_loss / (idx + 1)
        print("[CV {} CNN Train] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f} | loss:{:.4f}"
              .format(cv_idx, epoch + 1, args.cnn_epoch, t_acc, t_f1, t_kappa, t_loss))

        with torch.no_grad():
            pred_list = []
            true_list = []

            cnn_model.eval()
            for j, valid_data in enumerate(val_loader):
                valid_x, valid_y = valid_data
                s, c, _ = valid_x.shape
                valid_x = valid_x.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                b, _, s, _ = valid_x.shape

                if settings['use_cuda']:
                    valid_x = valid_x.cuda()
                    valid_y = valid_y.cuda()

                valid_output = cnn_model(valid_x, b)
                expected_valid_y = torch.argmax(torch.softmax(valid_output, dim=-1), dim=-1)
                true_list.append(valid_y.view(-1).detach().cpu().numpy())
                pred_list.append(expected_valid_y.view(-1).detach().cpu().numpy())

        true_list = np.array(np.concatenate(true_list))
        pred_list = np.array(np.concatenate(pred_list))

        acc = accuracy_score(true_list, pred_list)
        F1 = f1_score(true_list, pred_list, average='macro', zero_division=0)
        kappa = cohen_kappa_score(true_list, pred_list)
        if perfsum < (acc + F1 + kappa):
            best_cnn = cnn_model.state_dict()
            torch.save(cnn_model.state_dict(),
                       settings['output_path'] + "param/cnn_IP({:s})_SL({:d})_cv{}.pt"
                       .format(args.input_type, args.seq_len, cv_idx))
            perfsum = acc + F1 + kappa

        f1_confusion = []
        f1_confusion.append(t_cls_rpt['W']['f1-score'])
        f1_confusion.append(t_cls_rpt['N1']['f1-score'])
        f1_confusion.append(t_cls_rpt['N2']['f1-score'])
        f1_confusion.append(t_cls_rpt['N3']['f1-score'])
        try:
            f1_confusion.append(t_cls_rpt['N4']['f1-score'])
        except:
            pass
        f1_confusion.append(t_cls_rpt['R']['f1-score'])
        # for ii in range(settings['n_classes']):
        #     for jj in range(settings['n_classes']):
        #         factor = 2 * t_confusion[ii][jj] / (sum(t_confusion[ii]) + sum(np.transpose(t_confusion)[jj]))
        #         if factor == 0:
        #             factor = 1e-5
        #         f1_confusion.append(factor)
        #
        # f1_confusion = torch.tensor(f1_confusion).view([settings['n_classes'], settings['n_classes']])
        # if settings['use_cuda']:
        #     f1_confusion = f1_confusion.cuda()
        print("[CV {} CNN Valid] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f}"
              .format(cv_idx, epoch + 1, args.cnn_epoch, acc, F1, kappa))
        try:
            print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))
            cls_rpt = classification_report(true_list, pred_list, zero_division=0,
                                            target_names=['W', 'N1', 'N2', 'N3', 'R'], output_dict=True)
        except:
            print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'N4', 'R']))
            cls_rpt = classification_report(true_list, pred_list, zero_division=0,
                                            target_names=['W', 'N1', 'N2', 'N3', 'N4', 'R'], output_dict=True)

        W_f1.append(cls_rpt['W']['f1-score'])
        N1_f1.append(cls_rpt['N1']['f1-score'])
        N2_f1.append(cls_rpt['N2']['f1-score'])
        N3_f1.append(cls_rpt['N3']['f1-score'])
        try:
            N4_f1.append(cls_rpt['N4']['f1-score'])
        except:
            pass
        R_f1.append(cls_rpt['R']['f1-score'])
        Mean.append(F1)
        Acc.append(acc)
        Kappa.append(kappa)
    x_axis = [i + 1 for i in range(args.cnn_epoch)]
    plt.plot(x_axis, W_f1, marker='s', label='W')
    plt.plot(x_axis, R_f1, marker='s', label='R')
    plt.plot(x_axis, N1_f1, marker='s', label='N1')
    plt.plot(x_axis, N2_f1, marker='s', label='N2')
    plt.plot(x_axis, N3_f1, marker='s', label='N3')
    try:
        plt.plot(x_axis, N4_f1, marker='s', label='N4')
    except:
        pass
    plt.plot(x_axis, Mean, marker='s', label='Mean')
    plt.legend(loc='lower right')
    plt.xlabel('Training epoch')
    plt.ylabel('Classwise F1')
    plt.ylim(0, 1)
    plt.savefig(settings['output_path'] + 'cv{}_cnn_classwise_f1.png'.format(cv_idx))
    plt.close('all')

    return best_cnn

def lstm_train(cv_idx, train_loader, val_loader, criterion,
               cnn_model, lstm_model, lstm_opt, args, settings):
    W_f1, R_f1, N1_f1, N2_f1, N3_f1, N4_f1, Mean = [], [], [], [], [], [], []
    Acc, Kappa = [], []
    perfsum = 0

    for epoch in range(args.lstm_epoch):
        t_loss = 0
        pred_list_tr = []
        true_list_tr = []
        print('Epoch', epoch + 1)

        for idx, data in enumerate(tqdm(train_loader)):
            cnn_model.eval()
            lstm_model.train()
            train_x, train_y = data
            b, s, c, _ = train_x.shape
            train_x = train_x.permute(0, 2, 1, 3).contiguous()
            b, c, s, _ = train_x.shape
            train_y = train_y.long().view(-1)
            if settings['use_cuda']:
                train_x = train_x.cuda()
                train_y = train_y.cuda()

            state = lstm_model.init_hidden(b)
            lstm_opt.zero_grad()
            in_feature = cnn_model(train_x)
            train_output = lstm_model(in_feature, state, dense_connect=not args.no_dense_lstm)
            train_output = train_output.view(-1, settings['n_classes'])

            if (epoch < args.lstm_epoch // 3 or args.default_ce):
                train_l = criterion(train_output, train_y)
            else:
                train_l = criterion(train_output, train_y, f1_confusion)
            t_loss += train_l.item()
            train_l.backward()
            lstm_opt.step()
            expected_train_y = torch.argmax(torch.softmax(train_output, dim=-1), dim=-1)
            true_list_tr.append(train_y.view(-1).detach().cpu().numpy())
            pred_list_tr.append(expected_train_y.view(-1).detach().cpu().numpy())

        true_list_tr = np.concatenate(true_list_tr)
        pred_list_tr = np.concatenate(pred_list_tr)
        t_confusion = confusion_matrix(true_list_tr, pred_list_tr)
        t_acc = accuracy_score(true_list_tr, pred_list_tr)
        t_f1 = f1_score(true_list_tr, pred_list_tr, average='macro')
        t_kappa = cohen_kappa_score(true_list_tr, pred_list_tr)
        try:
            t_cls_rpt = classification_report(true_list_tr, pred_list_tr, zero_division=0,
                                              target_names=['W', 'N1', 'N2', 'N3', 'R'], output_dict=True)
        except:
            t_cls_rpt = classification_report(true_list_tr, pred_list_tr, zero_division=0,
                                              target_names=['W', 'N1', 'N2', 'N3', 'N4', 'R'], output_dict=True)

        t_loss = t_loss / (idx + 1)
        print("[CV {} LSTM Train] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f} | loss:{:.4f}"
              .format(cv_idx, epoch + 1, args.lstm_epoch, t_acc, t_f1, t_kappa, t_loss))

        with torch.no_grad():
            pred_list = []
            true_list = []

            cnn_model.eval()
            lstm_model.eval()
            for j, valid_data in enumerate(val_loader):
                valid_x, valid_y = valid_data
                s, c, _ = valid_x.shape
                valid_x = valid_x.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                b, _, s, _ = valid_x.shape

                if settings['use_cuda']:
                    valid_x = valid_x.cuda()
                    valid_y = valid_y.cuda()

                state = lstm_model.init_hidden(b)
                in_feature = cnn_model(valid_x)
                valid_output = lstm_model(in_feature, state, dense_connect=not args.no_dense_lstm)
                expected_valid_y = torch.argmax(torch.softmax(valid_output, dim=-1), dim=-1)
                true_list.append(valid_y.view(-1).detach().cpu().numpy())
                pred_list.append(expected_valid_y.view(-1).detach().cpu().numpy())

        true_list = np.concatenate(true_list)
        pred_list = np.concatenate(pred_list)

        acc = accuracy_score(true_list, pred_list)
        F1 = f1_score(true_list, pred_list, average='macro', zero_division=0)
        kappa = cohen_kappa_score(true_list, pred_list)

        if perfsum < (acc + F1 + kappa):
            best_lstm = lstm_model.state_dict()
            torch.save(lstm_model.state_dict(),
                       settings['output_path'] + "param/lstm_IP({:s})_SL({:d})_cv{}.pt"
                       .format(args.input_type, args.seq_len, cv_idx))
            perfsum = acc + F1 + kappa

        f1_confusion = []
        f1_confusion.append(t_cls_rpt['W']['f1-score'])
        f1_confusion.append(t_cls_rpt['N1']['f1-score'])
        f1_confusion.append(t_cls_rpt['N2']['f1-score'])
        f1_confusion.append(t_cls_rpt['N3']['f1-score'])
        try:
            f1_confusion.append(t_cls_rpt['N4']['f1-score'])
        except:
            pass
        f1_confusion.append(t_cls_rpt['R']['f1-score'])

        # for ii in range(settings['n_classes']):
        #     for jj in range(settings['n_classes']):
        #         factor = 2 * t_confusion[ii][jj] / (sum(t_confusion[ii]) + sum(np.transpose(t_confusion)[jj]))
        #         if factor == 0:
        #             factor = 1e-5
        #         f1_confusion.append(factor)

        # f1_confusion = torch.tensor(f1_confusion).view([settings['n_classes'], settings['n_classes']])
        # if settings['use_cuda']:
        #     f1_confusion = f1_confusion.cuda()
        print("[CV {} LSTM Valid] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f}"
              .format(cv_idx, epoch + 1, args.lstm_epoch, acc, F1, kappa))
        try:
            print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))
            cls_rpt = classification_report(true_list, pred_list, zero_division=0,
                                            target_names=['W', 'N1', 'N2', 'N3', 'R'], output_dict=True)
        except:
            print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'N4', 'R']))
            cls_rpt = classification_report(true_list, pred_list, zero_division=0,
                                            target_names=['W', 'N1', 'N2', 'N3', 'N4', 'R'], output_dict=True)

        W_f1.append(cls_rpt['W']['f1-score'])
        N1_f1.append(cls_rpt['N1']['f1-score'])
        N2_f1.append(cls_rpt['N2']['f1-score'])
        N3_f1.append(cls_rpt['N3']['f1-score'])
        try:
            N4_f1.append(cls_rpt['N4']['f1-score'])
        except:
            pass
        R_f1.append(cls_rpt['R']['f1-score'])
        Mean.append(F1)
        Acc.append(acc)
        Kappa.append(kappa)
    x_axis = [i + 1 for i in range(args.lstm_epoch)]
    plt.plot(x_axis, W_f1, marker='s', label='W')
    plt.plot(x_axis, R_f1, marker='s', label='R')
    plt.plot(x_axis, N1_f1, marker='s', label='N1')
    plt.plot(x_axis, N2_f1, marker='s', label='N2')
    plt.plot(x_axis, N3_f1, marker='s', label='N3')
    try:
        plt.plot(x_axis, N4_f1, marker='s', label='N4')
    except:
        pass
    plt.plot(x_axis, Mean, marker='s', label='Mean')
    plt.legend(loc='lower right')
    plt.xlabel('Training epoch')
    plt.ylabel('Classwise F1')
    plt.ylim(0, 1)
    plt.savefig(settings['output_path'] + 'cv{}_lstm_classwise_f1.png'.format(cv_idx))
    plt.close('all')

    return best_lstm

def test(cnn_model, lstm_model, test_loader, args, settings):
    performance = {}
    all_pred = []
    all_corr = []

    if args.cnn_only:
        cnn_model.eval()
        with torch.no_grad():
            pred_list = []
            true_list = []
            for j, test_data in enumerate(test_loader):
                test_x, test_y = test_data
                test_x = test_x.unsqueeze(1).permute(0, 2, 1, 3).contiguous()
                b, _, s, _ = test_x.shape
                test_y = test_y.long()

                if settings['use_cuda']:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()

                test_output = cnn_model(test_x, pretrain=True)
                expected_test_y = torch.argmax(torch.softmax(test_output, dim=-1), dim=-1)
                true_list.append(test_y.view(-1).detach().cpu().numpy())
                pred_list.append(expected_test_y.view(-1).detach().cpu().numpy())

    else:
        cnn_model.eval()
        lstm_model.eval()
        with torch.no_grad():
            pred_list = []
            true_list = []
            cnn_pred_list = []
            for j, test_data in enumerate(test_loader):
                test_x, test_y = test_data
                s, c, _ = test_x.shape
                test_x = test_x.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
                b, _, s, _ = test_x.shape
                test_y = test_y.long()

                if settings['use_cuda']:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()

                state = lstm_model.init_hidden(1)
                in_feature = cnn_model(test_x)
                cnn_res = cnn_model(test_x, pretrain=True)
                test_output = lstm_model(in_feature, state, dense_connect=not args.no_dense_lstm)
                expected_test_y = torch.argmax(torch.softmax(test_output, dim=-1), dim=-1)
                expected_cnn_test_y = torch.argmax(torch.softmax(cnn_res, dim=-1), dim=-1)
                true_list.append(test_y.view(-1).detach().cpu().numpy())
                cnn_pred_list.append(expected_cnn_test_y.view(-1).detach().cpu().numpy())
                pred_list.append(expected_test_y.view(-1).detach().cpu().numpy())

    true_list = np.concatenate(true_list)
    pred_list = np.concatenate(pred_list)
    all_corr.append(true_list)
    all_pred.append(pred_list)
    acc = accuracy_score(true_list, pred_list)
    F1 = f1_score(true_list, pred_list, average='macro', zero_division=0)
    kappa = cohen_kappa_score(true_list, pred_list)
    test_confusion = confusion_matrix(true_list, pred_list)
    performance['acc'] = acc
    performance['F1'] = F1
    performance['kappa'] = kappa

    return performance, test_confusion, true_list, pred_list

def prepare_training(args):
    if args.cnn_only:
        args.comment = args.comment + 'cnnonly_'
    if args.default_ce:
        args.comment = args.comment + 'defaultce_'
    if args.no_data_aug:
        args.comment = args.comment + 'noaug_'
    if args.no_dense_lstm:
        args.comment = args.comment + 'nodense_'
    if args.input_type == 'R&K':
        args.comment = args.comment + 'R&K_'

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    GPU_NUM = args.gpu
    device = 'cuda:{:d}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    print('Current device ', torch.cuda.current_device())  # check
    use_cuda = device
    print(torch.cuda.get_device_name(device))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(device) / 1024 ** 3, 1), 'GB')

    print('\nInput type: {}'.format(args.input_type))
    print('Sequence length: {}'.format(args.seq_len))
    print('CNN learning rate: {}'.format(args.cnn_lr))
    print('CNN weight decay: {}'.format(args.cnn_wd))
    print('CNN batch size: {}'.format(args.cnn_batch_size))

    if not args.cnn_only:
        print('LSTM learning rate: {}'.format(args.lstm_lr))
        print('LSTM weight decay: {}'.format(args.lstm_wd))
        print('LSTM batch size: {}'.format(args.lstm_batch_size))

    if args.default_ce:
        print('Loss function: Cross entropy')
    else:
        print('Loss function: Weighted cross entropy')

    if not args.cnn_only:
        print('Partial data augmentation:', not args.no_data_aug)
        print('Dense LSTM:', not args.no_dense_lstm)
    else:
        print('Using CNN only')

    now = datetime.datetime.now()
    output_path = './output/{}.{}.{}_{}.{}.{}_input({})_sl({})_clr{}_llr{}_cwd{}_lwd{}_{}_SEED{}/'.format(
        now.strftime('%Y'), now.strftime('%m'), now.strftime('%d'),
        now.strftime('%H'), now.strftime('%M'), now.strftime('%S'),
        args.input_type, args.seq_len, args.cnn_lr, args.lstm_lr, args.cnn_wd, args.lstm_wd, args.comment, args.seed)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + '/param/'):
        os.mkdir(output_path + '/param/')

    settings = {}
    settings['output_path'] = output_path
    settings['use_cuda'] = use_cuda

    return settings

def base_train(cv_idx, train_loader, val_loader, criterion, model, opt, scheduler, epoch):
    perfsum = 0

    for e in range(epoch):
        pred_list_tr = []
        true_list_tr = []
        t_loss = 0
        print('Epoch', e + 1)
        for idx, data in enumerate(tqdm(train_loader)):
            model.train()
            train_x, train_y = data
            train_x = train_x.unsqueeze(1).permute(0, 2, 1, 3).contiguous()
            b, _, s, _ = train_x.shape
            train_y = train_y.long()
            train_x = train_x.cuda()
            train_y = train_y.cuda()

            opt.zero_grad()
            train_output = model(train_x)
            train_y = train_y
            train_l = criterion(train_output.view(-1, 5), train_y.view(-1))
            t_loss += train_l.item()
            train_l.backward()
            opt.step()
            scheduler.step()

            expected_train_y = torch.argmax(torch.softmax(train_output, dim=-1), dim=-1)
            true_list_tr.append(train_y.view(-1).detach().cpu().numpy())
            pred_list_tr.append(expected_train_y.view(-1).detach().cpu().numpy())
        true_list_tr = np.array(np.concatenate(true_list_tr))
        pred_list_tr = np.array(np.concatenate(pred_list_tr))

        t_acc = accuracy_score(true_list_tr, pred_list_tr)
        t_f1 = f1_score(true_list_tr, pred_list_tr, average='macro')
        t_kappa = cohen_kappa_score(true_list_tr, pred_list_tr)

        t_loss = t_loss / (idx + 1)
        print("[CV {} Model Train] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f} | loss:{:.4f}"
              .format(cv_idx, e + 1, epoch, t_acc, t_f1, t_kappa, t_loss))

        with torch.no_grad():
            pred_list = []
            true_list = []

            model.eval()
            for j, valid_data in enumerate(val_loader):
                valid_x, valid_y = valid_data
                valid_x = valid_x.unsqueeze(1).permute(0, 2, 1, 3).contiguous()
                b, _, s, _ = valid_x.shape
                valid_y = valid_y.long()

                valid_x = valid_x.cuda()
                valid_y = valid_y.cuda()

                valid_output = model(valid_x)
                expected_valid_y = torch.argmax(torch.softmax(valid_output, dim=-1), dim=-1)
                true_list.append(valid_y.view(-1).detach().cpu().numpy())
                pred_list.append(expected_valid_y.view(-1).detach().cpu().numpy())

        true_list = np.array(np.concatenate(true_list))
        pred_list = np.array(np.concatenate(pred_list))

        acc = accuracy_score(true_list, pred_list)
        F1 = f1_score(true_list, pred_list, average='macro', zero_division=0)
        kappa = cohen_kappa_score(true_list, pred_list)
        if perfsum < (acc + F1 + kappa):
            best_model = model.state_dict()
            perfsum = acc + F1 + kappa

        print("[CV {} Model Valid] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f}"
              .format(cv_idx, e + 1, epoch, acc, F1, kappa))
        print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))

    return best_model

def base_train_lstm(cv_idx, train_loader, val_loader, criterion,
               model, opt, scheduler, epoch, seq_len=23):
    perfsum = 0

    for e in range(epoch):
        t_loss = 0
        pred_list_tr = []
        true_list_tr = []
        print('Epoch', e + 1)

        for idx, data in enumerate(tqdm(train_loader)):
            model.train()
            train_x, train_y = data
            # print(train_x.shape)
            bs, c, _ = train_x.shape
            try:
                train_x = train_x.view(bs // seq_len, seq_len, c, _).permute(0, 2, 1, 3).contiguous()
            except:
                train_x = train_x.view(1, -1, c, _).permute(0, 2, 1, 3).contiguous()
            b, c, s, _ = train_x.shape
            train_y = train_y.long()

            train_x = train_x.cuda()
            train_y = train_y.cuda()

            opt.zero_grad()
            train_output = model(train_x)

            train_l = criterion(train_output.view(-1, 5), train_y.view(-1))
            t_loss += train_l.item()
            train_l.backward()
            opt.step()
            scheduler.step()
            expected_train_y = torch.argmax(torch.softmax(train_output, dim=-1), dim=-1)
            true_list_tr.append(train_y.view(-1).detach().cpu().numpy())
            pred_list_tr.append(expected_train_y.view(-1).detach().cpu().numpy())

        true_list_tr = np.concatenate(true_list_tr)
        pred_list_tr = np.concatenate(pred_list_tr)

        t_acc = accuracy_score(true_list_tr, pred_list_tr)
        t_f1 = f1_score(true_list_tr, pred_list_tr, average='macro')
        t_kappa = cohen_kappa_score(true_list_tr, pred_list_tr)

        t_loss = t_loss / (idx + 1)
        print("[CV {} LSTM Train] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f} | loss:{:.4f}"
              .format(cv_idx, e + 1, epoch, t_acc, t_f1, t_kappa, t_loss))

        with torch.no_grad():
            pred_list = []
            true_list = []

            model.eval()
            for j, valid_data in enumerate(val_loader):
                valid_x, valid_y = valid_data
                try:
                    bs, c, _ = valid_x.shape
                    valid_x = valid_x.view(bs, seq_len, c, _).permute(0, 2, 1, 3).contiguous()
                except:
                    valid_x = valid_x.view(1, -1, c, _).permute(0, 2, 1, 3).contiguous()
                valid_y = valid_y.long()
                b, c, s, _ = valid_x.shape

                valid_x = valid_x.cuda()
                valid_y = valid_y.cuda()

                valid_output = model(valid_x)
                expected_valid_y = torch.argmax(torch.softmax(valid_output, dim=-1), dim=-1)
                true_list.append(valid_y.view(-1).detach().cpu().numpy())
                pred_list.append(expected_valid_y.view(-1).detach().cpu().numpy())

        true_list = np.concatenate(true_list)
        pred_list = np.concatenate(pred_list)

        acc = accuracy_score(true_list, pred_list)
        F1 = f1_score(true_list, pred_list, average='macro', zero_division=0)
        kappa = cohen_kappa_score(true_list, pred_list)

        if perfsum < (acc + F1 + kappa):
            best_model = model.state_dict()
            perfsum = acc + F1 + kappa

        print("[CV {} Model Valid] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f}"
              .format(cv_idx, e + 1, epoch, acc, F1, kappa))
        print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))

    return best_model


def base_test(model, test_loader):
    performance = {}
    all_pred = []
    all_corr = []

    model.eval()
    with torch.no_grad():
        pred_list = []
        true_list = []
        for j, test_data in enumerate(test_loader):
            test_x, test_y = test_data
            test_x = test_x.unsqueeze(1).permute(0, 2, 1, 3).contiguous()
            b, _, s, _ = test_x.shape
            test_y = test_y.long()

            test_x = test_x.cuda()
            test_y = test_y.cuda()

            test_output = model(test_x)
            expected_test_y = torch.argmax(torch.softmax(test_output, dim=-1), dim=-1)
            true_list.append(test_y.view(-1).detach().cpu().numpy())
            pred_list.append(expected_test_y.view(-1).detach().cpu().numpy())

    true_list = np.concatenate(true_list)
    pred_list = np.concatenate(pred_list)
    all_corr.append(true_list)
    all_pred.append(pred_list)
    acc = accuracy_score(true_list, pred_list)
    F1 = f1_score(true_list, pred_list, average='macro', zero_division=0)
    kappa = cohen_kappa_score(true_list, pred_list)
    test_confusion = confusion_matrix(true_list, pred_list)
    performance['acc'] = acc
    performance['F1'] = F1
    performance['kappa'] = kappa

    return performance, test_confusion, true_list, pred_list