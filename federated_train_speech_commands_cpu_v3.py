# 1. data loading: split trainning data to clients
# 2. In each training epoch, do #clients training on each small dataset. then combine and compute the gradients
# 3. what need to be compared?
# 4. considering using gpu for acceleration
import argparse
import time
import scipy, math
from scipy.linalg import null_space
from tqdm import *

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

import torchvision
from torchvision.transforms import *

from tensorboardX import SummaryWriter

import models
from datasets import *
from transforms import *
from mixup import *
from federated_utils_cpu import Federated

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--clients", type = int, default = 5, help= 'number of clients')
parser.add_argument("--num-threads", type = int, default = 10, help= 'number of threads')
parser.add_argument("--matrix-size", type = int, default = 1000, help= 'size of randomization matrix')
parser.add_argument("--train-dataset", type=str, default='datasets/speech_commands/train', help='path of train dataset')
parser.add_argument("--valid-dataset", type=str, default='datasets/speech_commands/valid', help='path of validation dataset')
parser.add_argument("--background-noise", type=str, default='datasets/speech_commands/train/_background_noise_', help='path of background noise')
parser.add_argument("--comment", type=str, default='', help='comment in tensorboard title')
parser.add_argument("--batch-size", type=int, default=128, help='batch size')
parser.add_argument("--dataload-workers-nums", type=int, default=6, help='number of workers for dataloader')
parser.add_argument("--weight-decay", type=float, default=1e-2, help='weight decay')
parser.add_argument("--optim", choices=['sgd', 'adam'], default='sgd', help='choices of optimization algorithms')
parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate for optimization')
parser.add_argument("--lr-scheduler", choices=['plateau', 'step'], default='plateau', help='method to adjust learning rate')
parser.add_argument("--lr-scheduler-patience", type=int, default=5, help='lr scheduler plateau: Number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument("--lr-scheduler-step-size", type=int, default=50, help='lr scheduler step: number of epochs of learning rate decay.')
parser.add_argument("--lr-scheduler-gamma", type=float, default=0.1, help='learning rate is multiplied by the gamma to decrease it')
parser.add_argument("--max-epochs", type=int, default=70, help='max number of epochs')
parser.add_argument("--resume", type=str, help='checkpoint file to resume')
parser.add_argument("--model", choices=models.available_models, default=models.available_models[0], help='model of NN')
parser.add_argument("--input", choices=['mel32'], default='mel32', help='input of NN')
parser.add_argument('--mixup', action='store_true', help='use mixup')
args = parser.parse_args()

use_gpu = torch.cuda.is_available()
use_gpu = 'True'
print('use_gpu', use_gpu)
print('num of clients', args.clients)
if use_gpu:
    torch.backends.cudnn.benchmark = True

n_mels = 32

def build_dataset(n_mels = n_mels, train_dataset = args.train_dataset, valid_dataset = args.valid_dataset, background_noise = args.background_noise):
    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    bg_dataset = BackgroundNoiseDataset(background_noise, data_aug_transform)
    add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=n_mels), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_dataset = SpeechCommandsDataset(train_dataset,
                                Compose([LoadAudio(),
                                         data_aug_transform,
                                         add_bg_noise,
                                         train_feature_transform]))

    valid_feature_transform = Compose([ToMelSpectrogram(n_mels=n_mels), ToTensor('mel_spectrogram', 'input')])
    valid_dataset = SpeechCommandsDataset(valid_dataset,
                                Compose([LoadAudio(),
                                         FixAudioLength(),
                                         valid_feature_transform]))
    return train_dataset, valid_dataset

def main():
    # 1. load dataset, train and valid
    train_dataset, valid_dataset = build_dataset(n_mels = n_mels, train_dataset = args.train_dataset, valid_dataset = args.valid_dataset, background_noise = args.background_noise)
    print('train ',len(train_dataset), 'val ', len(valid_dataset))

    weights = train_dataset.make_weights_for_balanced_classes()
    sampler = WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                              pin_memory=use_gpu, num_workers=args.dataload_workers_nums)
    # a name used to save checkpoints etc.
    # 2. prepare the model, checkpoint
    full_name = '%s_%s_%s_bs%d_lr%.1e_wd%.1e' % (args.model, args.optim, args.lr_scheduler, args.batch_size, args.learning_rate, args.weight_decay)
    if args.comment:
        full_name = '%s_%s' % (full_name, args.comment)

    model = models.create_model(model_name=args.model, num_classes=len(CLASSES), in_channels=1)

    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    start_timestamp = int(time.time()*1000)
    start_epoch = 0
    best_accuracy = 0
    best_loss = 1e100
    global_step = 0

    if args.resume:
        print("resuming getShapeLista checkpoint '%s'" % args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model.float()
        optimizer.load_state_dict(checkpoint['optimizer'])

        best_accuracy = checkpoint.get('accuracy', best_accuracy)
        best_loss = checkpoint.get('loss', best_loss)
        start_epoch = checkpoint.get('epoch', start_epoch)
        global_step = checkpoint.get('step', global_step)

        del checkpoint  # reduce memory

    if args.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_scheduler_patience, factor=args.lr_scheduler_gamma)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma, last_epoch=start_epoch-1)

    def get_lr():
        return optimizer.param_groups[0]['lr']

    writer = SummaryWriter(comment=('_speech_commands_' + full_name))

    #3. train and validation
    print("training %s for Google speech commands..." % args.model)
    since = time.time()
    #grad_client_list = [[]] * args.clients
    federated = Federated(args.clients, args.matrix_size, args.num_threads)
    for epoch in range(start_epoch, args.max_epochs):
        print("epoch %3d with lr=%.02e" % (epoch, get_lr()))
        phase = 'train'
        writer.add_scalar('%s/learning_rate' % phase,  get_lr(), epoch)

        model.train()  # Set model to training mode

        running_loss = 0.0
        it = 0
        correct = 0
        total = 0

        #compute for each client
        current_client = 0
        pbar = tqdm(train_dataloader, unit="audios", unit_scale=train_dataloader.batch_size, disable=False)
        for batch in pbar:
            inputs = batch['input']
            inputs = torch.unsqueeze(inputs, 1)
            targets = batch['target']
            #print(inputs.shape, targets.shape)
            if args.mixup:
                inputs, targets = mixup(inputs, targets, num_classes=len(CLASSES))

            inputs = Variable(inputs, requires_grad=True)
            targets = Variable(targets, requires_grad=False)
            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda(async=True)

            outputs = model(inputs)
            if args.mixup:
                loss = mixup_cross_entropy_loss(outputs, targets)
            else:
                loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            current_client_grad = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    #print(name, param.grad.shape, param.grad.type())#, param.grad)
                    current_client_grad.append(param.grad)
            #randomize the gradient, if in a new batch, generate the randomization matrix
            if (current_client == 0):
                federated.init(current_client_grad)
            #print("client ", current_client, " start")
            start_time = time.time()
            federated.work_for_client(current_client, current_client_grad)
            #print("client", current_client, " complete")
            end_time = time.time()
            #print("work for client ", current_client, " cost ", end_time - start_time)
            if (current_client == args.clients - 1):
                recover_start = time.time()
                recovered_grad = federated.recoverGradient()
                ind = 0
                recover_end = time.time()
                #print("recover gradient cost ", recover_end - recover_start)
                #print(recovered_grad_in_cuda, recovered_grad_in_cuda[0].shape, r)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.grad = recovered_grad[ind]
                        ind+=1
                assert(ind == len(recovered_grad))
                optimizer.step()
                #print("all clients finished")
                current_client = 0
            else :
                current_client += 1

            # only update the parameters when current_client == args.clients - 1

            # statistics
            it += 1
            global_step += 1
            #running_loss += loss.data[0]
            running_loss += loss.item()
            pred = outputs.data.max(1, keepdim=True)[1]
            if args.mixup:
                targets = batch['target']
                targets = Variable(targets, requires_grad=False).cuda(async=True)
            correct += pred.eq(targets.data.view_as(pred)).sum()
            total += targets.size(0)

            writer.add_scalar('%s/loss' % phase, loss.item(), global_step)

            # update the progress bar
            pbar.set_postfix({
                'loss': "%.05f" % (running_loss / it),
                'acc': "%.02f%%" % (100*float(correct)/total)
            })
            
            #print("[batch]\t", it, " [loss]\t ", running_loss / it, " [acc] \t", 100 * float(correct)/total)
            #print('------------------------------------------------------------------')
            #break

        accuracy = float(correct)/total
        epoch_loss = running_loss / it
        writer.add_scalar('%s/accuracy' % phase, 100*accuracy, epoch)
        writer.add_scalar('%s/epoch_loss' % phase, epoch_loss, epoch)
        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            checkpoint = {
                'epoch': epoch,
                'step': global_step,
                'state_dict': model.state_dict(),
                'loss': epoch_loss,
                'accuracy': accuracy,
                'optimizer' : optimizer.state_dict(),
            }
            torch.save(checkpoint, 'checkpoints/federated-best-loss-speech-commands-checkpoint-%s.pth' % full_name)
            torch.save(model, '%d-%s-federated-best-loss.pth' % (start_timestamp, full_name))
            del checkpoint
if __name__ == '__main__':
    main()
