import torch.optim as optim
import argparse
import network.model as mdl
import network.dataset as DS
import sys
from torch.utils.data import DataLoader
from train import *
from network.loss import Weighted_LablesSmoothing_Loss

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--data_dir', type=str, default="./data/",
                    help='Location of preprocessed input data.')
parser.add_argument('-intyp', '--input_type', type=str,
                    help='Type of input dataset: [SleepEDF or SHHS], (default: SleepEDF)', default='SleepEDF')
parser.add_argument('-sl', '--seq_len', type=int, default=30,
                    help='Input sequence length (default: 30) \n * Warning: value higher than 40 may cause the system freezing.')
parser.add_argument('-clr', '--cnn_lr', type=float, default=1e-3,
                    help='Learning rate for cnn training (default: 1e-3)')
parser.add_argument('-llr', '--lstm_lr', type=float, default=1e-4,
                    help='Learning rate for lstm training (default: 1e-4)')
parser.add_argument('-cwd', '--cnn_wd', type=float, default=1e-4,
                    help='Weight decay for cnn training (default: 1e-4)')
parser.add_argument('-lwd', '--lstm_wd', type=float, default=1e-5,
                    help='Weight decay for lstm training (default: 1e-5)')
parser.add_argument('-cepo', '--cnn_epoch', type=int, default=30,
                    help='Training epoch for cnn (default: 30)')
parser.add_argument('-lepo', '--lstm_epoch', type=int, default=30,
                    help='Training epoch for lstm (default: 30)')
parser.add_argument('-cbs', '--cnn_batch_size', type=int, default=128,
                    help='Training batch size for cnn (default: 128)')
parser.add_argument('-lbs', '--lstm_batch_size', type=int, default=64,
                    help='Training batch size for lstm (default: 64)')
parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Label smoothing hyper parameter alpha (default: 0.1)')
parser.add_argument('-g', '--gamma', type=float, default=3, help='Hyper parameter for adaptive loss function (default: 3)')
parser.add_argument('-k', '--k', type=int, default=10, help='Hyper parameter for adaptive loss function (default: 10)')
parser.add_argument('-tcv', '--target_cv', type=int, default=None,
                    help='Use this argument if you want train and evaluate on specific subject. (default: None)')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parser.parse_args()
    settings = prepare_training(args)

    cnn_batch_size = args.cnn_batch_size
    lstm_batch_size = args.lstm_batch_size
    all_corr, all_pred = [], []
    if args.target_cv != None:
        scv = args.target_cv
        tcv = scv + 1
    else:
        scv = 1
        tcv = 21

    for cv_idx in range(scv, tcv):
        print('\n************ Starting CV{} ************'.format(cv_idx))
        train_data, val_data, test_data, fs, n_classes = DS.load_dataset(args.data_dir, args.input_type, cv_idx)
        settings['n_classes'] = n_classes

        trainCNN = DS.CNNDataset(train_data[0], train_data[1])
        trainLSTM = DS.LSTMDataset(train_data[0], train_data[1], seq_len=args.seq_len)
        val_data = DS.CNNDataset(val_data[0], val_data[1])
        test_data = DS.CNNDataset(test_data[0], test_data[1])

        train_dataloader_CNN = DataLoader(trainCNN, batch_size=cnn_batch_size, shuffle=True)
        train_dataloader_LSTM = DataLoader(trainLSTM, batch_size=lstm_batch_size, shuffle=True, drop_last=False)
        valid_dataloader = DataLoader(val_data, batch_size=args.seq_len, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=args.seq_len, shuffle=False, drop_last=False)

        criterion = Weighted_LablesSmoothing_Loss(k=args.k, n_classes=n_classes, alpha=args.alpha, gamma=args.gamma).cuda()
        cnn_model = mdl.CNN(fs=fs, n_classes=n_classes).cuda()
        lstm_model = mdl.LSTM(n_classes=n_classes).cuda()
        cnn_opt = optim.Adam(cnn_model.parameters(), lr=args.cnn_lr, weight_decay=args.cnn_wd)
        lstm_opt = optim.Adam(lstm_model.parameters(), lr=args.lstm_lr, weight_decay=args.lstm_wd)

        #################### CNN TRAINING ####################
        best_cnn = cnn_train(cv_idx, train_dataloader_CNN, valid_dataloader,
                            criterion, cnn_model, cnn_opt, args, settings)
        cnn_model.load_state_dict(best_cnn)
        ##################### LSTM TRAINING #####################
        print('CNN stage complete, starting Bi-LSTM training')
        best_lstm = lstm_train(cv_idx, train_dataloader_LSTM, valid_dataloader,
                                criterion, cnn_model, lstm_model, lstm_opt, args, settings)
        lstm_model.load_state_dict(best_lstm)

        #################### TEST SESSION ####################
        performance, test_confusion, true_list, pred_list = test(cnn_model, lstm_model,
                                                               test_dataloader, args, settings)
        all_corr.append(true_list)
        all_pred.append(pred_list)
        corr_ = np.concatenate(all_corr)
        pred_ = np.concatenate(all_pred)
        cv_mean_acc = accuracy_score(corr_, pred_)
        cv_mean_f1 = f1_score(corr_, pred_, average='macro', zero_division=0)
        cv_mean_kappa = cohen_kappa_score(corr_, pred_)

        print('******* [CV {}] Test performance *******'.format(cv_idx))
        print('[CV {}] Acc.: {:.4f} | F1 score: {:.4f} | Kappa: {:.4f}'
              .format(cv_idx, performance['acc'], performance['F1'], performance['kappa']))
        print(classification_report(true_list, pred_list))
        print(test_confusion)
        print('[Mean CV] Acc.: {:.4f} | F1 score: {:.4f} | Kappa: {:.4f}'
              .format(cv_mean_acc, cv_mean_f1, cv_mean_kappa))

        with open(settings['output_path'] + 'Result.txt', 'a') as f:
            f.write('\n[CV {}] Acc.: {:.4f} | F1 score: {:.4f} | Kappa: {:.4f}\n'
              .format(cv_idx, performance['acc'], performance['F1'], performance['kappa']))
            f.write(classification_report(true_list, pred_list))
            f.write(str(test_confusion))
            f.write('\n[Mean CV] Acc.: {:.4f} | F1 score: {:.4f} | Kappa: {:.4f}\n'
              .format(cv_mean_acc, cv_mean_f1, cv_mean_kappa))
            f.close()

    print('******* CV completed *******')
    all_corr = np.concatenate(all_corr)
    all_pred = np.concatenate(all_pred)

    acc = accuracy_score(all_corr, all_pred)
    F1 = f1_score(all_corr, all_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_corr, all_pred)
    print(classification_report(all_corr, all_pred, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))
    print(confusion_matrix(all_corr, all_pred))
    print('[Mean CV performance] Acc.: {:.4f} | F1 score: {:.4f} | Kappa: {:.4f}'
          .format(acc, F1, kappa))

    sys.stdout = open(settings['output_path'] + 'Result.txt', 'a')
    print('******* CV completed *******')
    print(classification_report(all_corr, all_pred, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))
    print(confusion_matrix(all_corr, all_pred))
    print('[Mean CV performance] Acc.: {:.4f} | F1 score: {:.4f} | Kappa: {:.4f}'
          .format(acc, F1, kappa))
    sys.stdout.close()