# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author wenjing
"""

import os
import sys
import argparse
import time
from datetime import datetime
#import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DataSet
from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, \
    get_torch_network, get_mean_std, modify_output,get_train_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args,logger,epoch):

    start = time.time()
    net.train()
    loss_train = 0.0
    acc_train = 0.0
    correct_prediction = 0.0
    # total = 0.0
    for batch_index, (images, labels) in enumerate(tqdm(training_loader)):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
        if args.single_input:
            outputs = net(images)
        else:
            outputs,_ = net(images,labels)
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        loss.backward()
        optimizer.step()
        correct_prediction += (predicted == labels).sum().item()
        # total += labels.size(0)
        #####
        if epoch <= args.warm:
            warmup_scheduler.step()
        #######
        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        '''logger.info('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))'''
    # total_batch = len(training_loader)
    train_loss = loss_train /len(training_loader)
    train_acc = correct_prediction / len(train_datasets)

    finish = time.time()
    logger.info('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
# validation
    if epoch % 1 == 0:
        start = time.time()
        net.eval()
        test_loss = 0.0 # cost function error
        correct = 0.0
    
        for batch_idx, (images, labels) in enumerate(val_loader):

            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                if args.single_input:
                    outputs = net(images)
                else:
                    outputs,_ = net(images,labels)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum()

        finish = time.time()
        test_loss = test_loss / len(val_loader)
        test_acc = correct.float() / len(val_datasets)
   
        '''if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary())
        print('Evaluating Network.....')
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            test_loss,
            test_acc,
            finish - start
        ))
        print()'''
    
    logger.info("train Acc:"+ str(train_acc))
    logger.info("train Loss:"+ str(train_loss))
    
    logger.info("val Acc:"+ str(test_acc))
    logger.info("val Loss:"+ str(test_loss))

    return test_acc
def setup_logger(path):
    # 创建一个logger对象
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)  # 设置最低的日志级别

    # 创建一个文件处理器，并设置其日志格式
    file_handler = logging.FileHandler(os.path.join(path, 'logger.txt'), mode='a', encoding='utf-8')  # 'a' 表示追加模式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将处理器添加到logger
    logger.addHandler(file_handler)

    return logger
from tqdm import tqdm
import logging

if __name__ == '__main__':
    #dataset = 'pic-day-cam1'
    #net_name = 'my_ca'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', default='mobilenet_v2', type=str, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=2, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-pretrain', default=True, help='wether the pretrain model is used')
    parser.add_argument("-dataset", default='pic-day-cam3', type=str)
    parser.add_argument("-num_class", default=22, type=int)
    parser.add_argument("-single_input", default=False)
    
    args = parser.parse_args()
    
    if args.dataset == 'pic-day-cam4':
        teacher_root = 'checkpoint/pic-day-cam4-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-81-best.pth'
    elif args.dataset == 'pic-day-cam3':
        teacher_root = 'checkpoint/pic-day-cam3-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-91-best.pth'
    elif args.dataset == 'pic-day-cam2':
        teacher_root = 'checkpoint/pic-day-cam2-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-96-best.pth'
    elif args.dataset == 'pic-day-cam1':
        teacher_root = 'checkpoint/pic-day-cam1-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-99-best.pth'
    elif args.dataset == 'pic-night-cam4':
        teacher_root = 'checkpoint/pic-night-cam4-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-88-best.pth'
    elif args.dataset == 'pic-night-cam3':
        teacher_root = 'checkpoint/pic-night-cam3-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-82-best.pth'
    elif args.dataset == 'pic-night-cam2':
        teacher_root = 'checkpoint/pic-night-cam2-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-94-best.pth'
    elif args.dataset == 'pic-night-cam1':
        teacher_root = 'checkpoint/pic-night-cam1-teacher/split_by_driver/mobilenet_v2/mobilenet_v2-97-best.pth'

    args.teacher_root = teacher_root
    print(args)
    #args.dataset = dataset
    #args.net = net_name
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH,args.dataset,'split_by_driver', args.net)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    logger = setup_logger(checkpoint_path)
    

# bulid the backbone model
    net = get_torch_network(args)
    #net = modify_output(args, net)
    if args.pretrain:
        logger.info("load pretrain model successful")

    for name,parameters in net.named_parameters():
        logger.info(name+':'+str(parameters.size()))
    

# the config for wandb output
    config = dict(
    architecture = args.net,
    dataset_id = args.dataset,
    batch_size = args.b,
    pretrain = args.pretrain,
    lr = 'StepLR50'
     )
     
    #wandb.init(project="pic-car-157", entity = 'wenjing', config=config, name=args.dataset + '_' + args.net)


    if args.gpu:
        net = net.cuda()

# prepare the data
    trainloader, trainloadertxt, valloader, valloadertxt = get_train_split(args.dataset)
    mean, std = get_mean_std(args.dataset)
    train_datasets = DataSet(trainloader, trainloadertxt, mean, std, flag ='train')#get data
    val_datasets = DataSet(valloader, valloadertxt, mean, std, flag = 'val')#
    training_loader = get_training_dataloader(
        dataset = train_datasets,
        num_workers=4,
        batch_size=args.b,
        shuffle=True  
    )
    val_loader = get_test_dataloader(
        dataset = val_datasets,
        num_workers=4,
        batch_size=args.b,
        shuffle=True  
    )

# define loss function
    loss_function = nn.CrossEntropyLoss()

# prepare the optimizer
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

# set detail of the checkpoint storage path
    
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    best_path = ''
# start training the model
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH+1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        '''if args.resume:
            if epoch <= resume_epoch:
                continue'''
        acc = train(args,logger,epoch)

        # save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            logger.info("best model! save...")
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            best_acc = acc
            logger.info("best_acc: {:.4f}".format(best_acc))
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
    if best_path == '' and settings.EPOCH > 0:#如果没有保存过best model，则保存最后一个epoch的模型
        logger.info("save last model as the best one...")
        torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
        best_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
    
    import argparse
    from dataset import DataSet
    import torch
    from utils import get_test_dataloader, get_torch_network, modify_output, get_test_split, get_mean_std
    import time
    #parser = argparse.ArgumentParser()
    #parser.add_argument('-net', type=str, default='mobilenet_v2', help='net type')
    #parser.add_argument('-dataset', type=str, default='', help='the name of the dataset')
    #parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    #parser.add_argument('-pretrain', type=bool, default=False, help='use pretrain model or not')
    #parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    #parser.add_argument('-num_class', type=int, default=22, help='the number of the classes')
    #parser.add_argument('-model_root', type=str, default='', help='the weights file you want to test')
    test_args = parser.parse_args()
    test_args.dataset = args.dataset
    test_args.net = args.net
    test_args.model_root = best_path
    test_args.teacher_root = args.teacher_root
    test_args.gpu = args.gpu
    test_args.b = args.b
    test_args.warm = args.warm
    test_args.lr = args.lr
    test_args.resume = False
    test_args.pretrain = True
    test_args.num_class = args.num_class
    test_args.single_input = args.single_input
    print(test_args)
    print("best model path:", best_path)
    net = get_torch_network(test_args)

#load model from the checkpoint path
    #net = modify_output(test_args, net)
    
    net = net.cuda()
 
    checkpoint = torch.load(best_path)
    net.load_state_dict(checkpoint)

    net.eval()

    test_root, test_split = get_test_split(test_args.dataset)
    mean, std = get_mean_std(test_args.dataset)
# prepare test data
    val_datasets = DataSet(test_root, test_split, mean, std, flag='val')
    test_loader = get_test_dataloader(
        dataset = val_datasets,
        num_workers=4,
        batch_size=test_args.b,
        shuffle=True  
    )


# start testing
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if test_args.gpu:
                image = image.cuda()
                label = label.cuda()
                
            time0 = time.time()
            if test_args.single_input:
                output = net(image)
            else:
                output,_ = net(image,label)
            time1 = time.time()
            timed = time1 - time0
            #print('time', timed)

            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()


    #print()
    logger.info("Top 1 acc: "+ str(correct_1 / len(test_loader.dataset)))
    logger.info("Top 5 acc: "+ str(correct_5 / len(test_loader.dataset)))
    logger.info("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))


