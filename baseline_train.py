import torch.optim as optim
import argparse
import network.baseline as mdl
import network.dataset as DS
import sys
from torch.utils.data import DataLoader
from train import *
from network.loss import temporal_crossentropy_loss

parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument('--data_dir', type=str, default="./data/",
                    help='pre-processed data dir')
parser.add_argument('-intyp', '--input_type', type=str,
                    help='SleepEDF, SHHS, male_SHHS, female_SHHS, R&K', default='SleepEDF')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('-c', '--comment', type=str, default='')

if __name__ == '__main__':
    args = parser.parse_args()
    all_corr, all_pred = [], []

    now = datetime.datetime.now()
    output_path = './output/{}.{}.{}_{}.{}.{}_baseline_{}/'.format(
        now.strftime('%Y'), now.strftime('%m'), now.strftime('%d'),
        now.strftime('%H'), now.strftime('%M'), now.strftime('%S'), args.comment)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + '/param/'):
        os.mkdir(output_path + '/param/')

    batch_size = 256

    for cv_idx in range(1, 21):
        print('\n************ Starting CV{} ************'.format(cv_idx))
        train_data, val_data, test_data, fs, n_classes = DS.load_dataset(args.data_dir, args.input_type, cv_idx)

        train = DS.CNNDataset(train_data[0], train_data[1])
        valid = DS.CNNDataset(val_data[0], val_data[1])
        test_data = DS.CNNDataset(test_data[0], test_data[1])

        train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)

        model1 = mdl.RnnModel().cuda()
        model1_opt = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        scheduler = optim.lr_scheduler.CyclicLR(mode='triangular', base_lr=0.01, max_lr=0.05, step_size_up=500, optimizer=model1_opt)

        criterion = temporal_crossentropy_loss()
        best_model = base_train(cv_idx, train_dataloader, valid_dataloader,
                            criterion, model1, model1_opt, scheduler, 20)
        model1.load_state_dict(best_model)

        # best_model = base_train_lstm(cv_idx, train_dataloader, valid_dataloader,
        #                     criterion, model1, model1_opt, scheduler, 20)

        performance, test_confusion, true_list, pred_list = base_test(model1, test_dataloader)
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

        with open(output_path + 'Result.txt', 'a') as f:
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

    sys.stdout = open(output_path + 'Result.txt', 'a')
    print('******* CV completed *******')
    print(classification_report(all_corr, all_pred, zero_division=0, target_names=['W', 'N1', 'N2', 'N3', 'R']))
    print(confusion_matrix(all_corr, all_pred))
    print('[Mean CV performance] Acc.: {:.4f} | F1 score: {:.4f} | Kappa: {:.4f}'
          .format(acc, F1, kappa))
    sys.stdout.close()