import copy
import logging
from sys import stderr
import warnings
warnings.filterwarnings("ignore")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("training_log_pre2e-5.txt"),  # 日志写入文件
                        logging.StreamHandler()  # 日志输出到控制台
                    ])
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
from process.utils.debug import debug
from torch.cuda.amp import GradScaler, autocast
from encoder import encode_input

def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets.long())
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), accuracy_score(all_targets, all_predictions) * 100
    pass

def evaluate_metrics(model, loss_function, dataset):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for input, targets in dataset:
            targets = targets.cuda()
            # input_ids, attention_mask = encode_input(feat)
            # predictions = model(input, input_ids, attention_mask)
            # predictions = model(input_ids, attention_mask)
            # predictions = predictions.logits
            predictions = model(input)
            batch_loss = loss_function(predictions, targets.long())
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), \
               accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass


def train(model, dataset, epoches, dev_every, loss_function, optimizer, save_path, log_every=5, max_patience=20, best_path=None, test=False):
    train_dataset, test_dataset, val_dataset = dataset[0], dataset[1], dataset[2]
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    log_flag = 0
    all_train_acc = []
    all_train_loss = []
    all_valid_acc = []
    all_valid_loss = []
    train_processer = tqdm(train_dataset, total=len(train_dataset))
    test_processer = tqdm(test_dataset, total=len(test_dataset))
    valid_processer = tqdm(val_dataset, total=len(val_dataset))
    scaler = GradScaler()
    
    if best_path is not None:
        model.load_state_dict(torch.load(best_path))
        logging.info('#' * 100)
        logging.info("Test result")
        loss, acc, pr, rc, f1 = evaluate_metrics(model, loss_function, test_processer)
        debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
        logging.info('%s\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (save_path, loss, acc, pr, rc, f1))
        exit()
    if test:
        logging.info('#' * 100)
        logging.info("Test result")
        loss, acc, pr, rc, f1 = evaluate_metrics(model, loss_function, test_processer)
        debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
        logging.info('%s\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (save_path, loss, acc, pr, rc, f1))
        exit()
    try:
        for e in range(epoches):
            _loss = []
            all_predictions, all_targets = [], []
            for input, targets in train_processer:
                model.train()
                model.zero_grad()
                targets = targets.cuda()
                # input_ids, attention_mask = encode_input(feat)
                # predictions = model(input, input_ids, attention_mask)
                # predictions = model(input_ids, attention_mask)
                # predictions = predictions.logits
                predictions = model(input)
                batch_loss = loss_function(predictions, targets.long())
                train_losses.append(batch_loss.detach().item())
                batch_loss.backward()
                optimizer.step()
                
                predictions = predictions.detach().cpu()
                
                if predictions.ndim == 2:
                    all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
                else:
                    all_predictions.extend(
                        predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                            dtype=torch.int32).numpy().tolist()
                    )
                all_targets.extend(targets.detach().cpu().numpy().tolist())
                _loss.append(batch_loss.detach().cpu().item())
            # print(model.weight)   
            train_loss, train_acc, train_pr, train_rc, train_f1 = np.mean(_loss).item(), \
            accuracy_score(all_targets, all_predictions) * 100, \
            precision_score(all_targets, all_predictions) * 100, \
            recall_score(all_targets, all_predictions) * 100, \
            f1_score(all_targets, all_predictions) * 100

            # train_loss, train_acc, train_pr, train_rc, train_f1 = evaluate_metrics(model, loss_function, train_dataset)
            all_train_acc.append(train_acc)
            all_train_loss.append(train_loss)

            logging.info('-' * 100)
            logging.info('Epoch %d\t---Train--- Average Loss: %10.4f\t Patience %d\t Loss: %10.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tf1: %5.3f\t' % (
                e, np.mean(train_losses).item(), patience_counter, train_loss, train_acc, train_pr, train_rc, train_f1))
            loss, acc, pr, rc, valid_f1 = evaluate_metrics(model, loss_function, valid_processer)
            logging.info('Epoch %d\t----Valid---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (e, loss, acc, pr, rc, valid_f1))
            
            test_loss, test_acc, test_pr, test_rc, test_f1 = evaluate_metrics(model, loss_function, test_processer)
            logging.info('Epoch %d\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (e, test_loss, test_acc, test_pr, test_rc, test_f1))
            all_valid_acc.append(acc)
            all_valid_loss.append(loss)
            if test_f1 > best_f1:
                patience_counter = 0
                best_f1 = test_f1
                best_model = copy.deepcopy(model.state_dict())
                _save_file = open('diverse_temp{}.bin'.format(e), 'wb')
                torch.save(model.state_dict(), _save_file)
                _save_file.close()
            else:
                patience_counter += 1
            train_losses = []
            if patience_counter == max_patience:
                break
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')
        logging.info('Training Interrupted by user!')
    logging.info('Finish training!')

    if best_model is not None:
        model.load_state_dict(best_model)
    _save_file = open('diverse_all.bin', 'wb')
    torch.save(model.state_dict(), _save_file)
    _save_file.close()


    logging.info('#' * 100)
    logging.info("Test result")
    loss, acc, pr, rc, f1 = evaluate_metrics(model, loss_function, test_processer)
    debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    logging.info('%s\t----Test---- Loss: %0.4f\tAccuracy: %0.4f\tPrecision: %0.4f\tRecall: %0.4f\tF1: %0.4f' % (save_path, loss, acc, pr, rc, f1))
    