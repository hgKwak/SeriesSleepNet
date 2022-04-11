import torch
import numpy as np
import os
import datetime
from tqdm import tqdm
from sklearn.metrics import *

def cnn_train(cv_idx, train_loader, val_loader, criterion, cnn_model, cnn_opt, args, settings):
    Acc, Kappa, Mean = [], [], []
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
            if (epoch < args.cnn_epoch // 3):
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
        t_cls_rpt = classification_report(true_list_tr, pred_list_tr, zero_division=0,
                                          target_names=['W', 'N1', 'N2', 'N3', 'R'], output_dict=True)

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
        f1_confusion.append(t_cls_rpt['R']['f1-score'])

        print("[CV {} CNN Valid] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f}"
              .format(cv_idx, epoch + 1, args.cnn_epoch, acc, F1, kappa))
        print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))

        Mean.append(F1)
        Acc.append(acc)
        Kappa.append(kappa)

    return best_cnn

def lstm_train(cv_idx, train_loader, val_loader, criterion,
               cnn_model, lstm_model, lstm_opt, args, settings):
    Acc, Kappa, Mean = [], [], []
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
            train_output = lstm_model(in_feature, state)
            train_output = train_output.view(-1, settings['n_classes'])

            if (epoch < args.lstm_epoch // 3):
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
        t_acc = accuracy_score(true_list_tr, pred_list_tr)
        t_f1 = f1_score(true_list_tr, pred_list_tr, average='macro')
        t_kappa = cohen_kappa_score(true_list_tr, pred_list_tr)
        t_cls_rpt = classification_report(true_list_tr, pred_list_tr, zero_division=0,
                                          target_names=['W', 'N1', 'N2', 'N3', 'R'], output_dict=True)

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
                valid_output = lstm_model(in_feature, state)
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
        f1_confusion.append(t_cls_rpt['R']['f1-score'])

        print("[CV {} LSTM Valid] epoch: {}/{} | Acc.: {:.4f} | "
              "F1 score: {:.4f} | kappa: {:.4f}"
              .format(cv_idx, epoch + 1, args.lstm_epoch, acc, F1, kappa))
        print(classification_report(true_list, pred_list, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))

        Mean.append(F1)
        Acc.append(acc)
        Kappa.append(kappa)

    return best_lstm

def test(cnn_model, lstm_model, test_loader, args, settings):
    performance = {}
    all_pred = []
    all_corr = []

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
            test_output = lstm_model(in_feature, state)
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
    GPU_NUM = args.gpu
    device = 'cuda:{:d}'.format(GPU_NUM) if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    print('Current device ', torch.cuda.current_device())
    use_cuda = device
    print(torch.cuda.get_device_name(device))
    print('\nInput type: {}'.format(args.input_type))
    print('Sequence length: {}'.format(args.seq_len))
    print('CNN learning rate: {}'.format(args.cnn_lr))
    print('CNN weight decay: {}'.format(args.cnn_wd))
    print('CNN batch size: {}'.format(args.cnn_batch_size))

    now = datetime.datetime.now()
    output_path = './output/{}.{}.{}_{}.{}.{}_input({})_sl({})_clr{}_llr{}_cwd{}_lwd{}/'.format(
        now.strftime('%Y'), now.strftime('%m'), now.strftime('%d'),
        now.strftime('%H'), now.strftime('%M'), now.strftime('%S'),
        args.input_type, args.seq_len, args.cnn_lr, args.lstm_lr, args.cnn_wd, args.lstm_wd)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + '/param/'):
        os.mkdir(output_path + '/param/')

    settings = {}
    settings['output_path'] = output_path
    settings['use_cuda'] = use_cuda

    return settings
