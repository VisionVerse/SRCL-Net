import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
from datetime import datetime
from utils import dataloader
from utils.metrics import Evaluator
from utils.tools import adjust_lr, AvgMeter
import argparse
import logging
from network.CDRNet import CDRNet
from network.tools import *
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

#train
def Train(train_loader, BCD_Model, EUE_Model, BCD_Model_optimizer, EUE_Model_optimizer, epoch, Eva):
    BCD_Model.train()
    EUE_Model.train()
    loss_record_cdrnet = AvgMeter()
    loss_record_eue = AvgMeter()
    print('CDRNet Learning Rate: {}'.format(BCD_Model_optimizer.param_groups[0]['lr']))
    for i, sample in enumerate(tqdm(train_loader), start=1):
        BCD_Model_optimizer.zero_grad()
        EUE_Model_optimizer.zero_grad()
        A, B, mask = sample['A'], sample['B'], sample['label']
        A = Variable(A).to(device, non_blocking=True)
        B = Variable(B).to(device, non_blocking=True)
        Y = Variable(mask).to(device, non_blocking=True)
        gts = Y.unsqueeze(1)

        # train CDRNet
        Refined_out, edge_out = BCD_Model(A, B, gts)
        bcd_module = unwrap_model(BCD_Model)
        reg_loss = l2_regularisation(bcd_module.SCC_1) + l2_regularisation(bcd_module.SCC_2) + \
                   l2_regularisation(bcd_module.SCC_3) + l2_regularisation(bcd_module.SCC_4) + \
                   l2_regularisation(bcd_module.edge_head) + l2_regularisation(bcd_module.decoder)
        reg_loss = opt.reg_weight * reg_loss
        edge_gt = mask_to_edge(gts)

        loss_cd = BCE_loss(Refined_out, gts)
        loss_edge = BCE_loss(edge_out, edge_gt) + dice_loss_with_logits(edge_out, edge_gt)
        total_loss = loss_cd + opt.edge_weight * loss_edge + reg_loss

        total_loss.backward()
        BCD_Model_optimizer.step()

        # get variance map (entropy)
        preds = [torch.sigmoid(Refined_out)]
        with torch.no_grad():
            for ff in range(4):
                ff_m = BCD_Model(A,B, gts)[0]
                preds.append(torch.sigmoid(ff_m))
        preds = torch.cat(preds, dim=1)
        mean_preds = torch.mean(preds, 1, keepdim=True)
        var_map = -1 * mean_preds * torch.log(mean_preds + 1e-8)
        var_map = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
        var_map = var_map.detach()

        # train EUE
        output_D = EUE_Model(torch.cat((A, B, torch.sigmoid(Refined_out.detach())), 1))
        output_D = F.interpolate(output_D, size=(opt.trainsize, opt.trainsize), mode='bilinear', align_corners=True)
        approximation_loss = BCE_loss(output_D, var_map)
        approximation_loss.backward()
        EUE_Model_optimizer.step()


        loss_record_cdrnet.update(total_loss.data, opt.batchsize)
        loss_record_eue.update(approximation_loss.data, opt.batchsize)

        if i % 100 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, Edge Loss: {:.4f}, Reg Loss: {:.4f}, CD Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record_cdrnet.show(), loss_edge.item(), reg_loss.item(), loss_cd.item()))
            logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total Loss: {:.4f}, Edge Loss: {:.4f}, Reg Loss: {:.4f}, CD Loss: {:.4f}'.
                         format(epoch, opt.epoch, i, total_step, loss_record_cdrnet.show(), loss_edge.item(), reg_loss.item(), loss_cd.item()))

        output = Refined_out.sigmoid().data.cpu().numpy().squeeze()
        output[output>=0.5] = 1
        output[output<0.5] = 0
        target = Y.cpu().numpy()
        # Add batch sample into evaluator
        # print(target.shape, output.shape)
        Eva.add_batch(target, output.astype(np.int64))
    IoU = Eva.Intersection_over_Union()[1]
    F1 = Eva.F1()[1]
    print('Epoch [{:03d}/{:03d}], \n[Training] IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))

    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], IoU: {:.4f}, F1: {:.4f}'.format(epoch, opt.epoch, IoU, F1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight for regularization term')
    parser.add_argument('--edge_weight', type=float, default=0.1, help='weight for edge supervision')
    parser.add_argument('--data_name', type=str, default='foggy-LEVIR-CD', help='the test rgb images root')
    parser.add_argument('--segclass', type=int, default=1, help='number of segmentation classes')
    parser.add_argument('--save_path', type=str, default='./train_output/CDRNet/')
    opt = parser.parse_args()

    save_path = opt.save_path + opt.data_name + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('CDRNet Learning Rate: {}'.format(opt.lr))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')
        print('Using CUDA device(s): {} (visible: {})'.format(gpu_count, visible_devices))
    else:
        print('CUDA is not available, using CPU.')

    # build models
    BCD_Model = CDRNet(num_classes=opt.segclass).to(device)
    if gpu_count > 1:
        BCD_Model = nn.DataParallel(BCD_Model)
    BCD_Model_params = BCD_Model.parameters()
    BCD_Model_optimizer = torch.optim.Adam(BCD_Model_params, opt.lr)

    EUE_Model = UncertaintyHead(channel=64).to(device)
    if gpu_count > 1:
        EUE_Model = nn.DataParallel(EUE_Model)
    EUE_Model_params = EUE_Model.parameters()
    EUE_Model_optimizer = torch.optim.Adam(EUE_Model_params, 1e-4)

    # set path
    if opt.data_name == 'foggy-LEVIR-CD':
        opt.train_root = "/data/BCD-datasets/LEVIR-CD-foggy/train/"
        opt.val_root = "/data/BCD-datasets/LEVIR-CD-foggy/test/"
        palatte = [[0, 0, 0], [255, 255, 255]]
    elif opt.data_name == 'foggy-LEVIR-CD+':
        opt.train_root = "/data/BCD-datasets/LEVIR-CD+foggy/train/"
        opt.val_root = "/data/BCD-datasets/LEVIR-CD+foggy/test/"
        palatte = [[0, 0, 0], [255, 255, 255]]
    elif opt.data_name == 'foggy-SYSU-CD':
        opt.train_root = "/data/BCD-datasets/SYSU-CD-foggy/train/"
        opt.val_root = "/data/BCD-datasets/SYSU-CD-foggy/val/"
        palatte = [[0, 0, 0], [255, 255, 255]]
    elif opt.data_name == 'HH':
        opt.train_root = "/data/BCD-datasets/HH/train/"
        opt.val_root = "/data/BCD-datasets/HH/val/"
        palatte = [[0, 0, 0], [255, 255, 255]]


    train_loader = dataloader.get_loader(img_A_root = opt.train_root + 'A/', img_B_root = opt.train_root + 'B/', gt_root = opt.train_root + 'label/', trainsize = opt.trainsize, palatte = palatte, mode ='train', batchsize = opt.batchsize, mosaic_ratio=0.25, num_workers=4, shuffle=True, pin_memory=True)
    test_loader = dataloader.get_loader(img_A_root = opt.val_root + 'A/', img_B_root = opt.val_root + 'B/', gt_root = opt.val_root + 'label/', trainsize = opt.trainsize, palatte = palatte, mode ='val', batchsize = opt.batchsize, mosaic_ratio=0, num_workers=4, shuffle=False, pin_memory=True)
    total_step = len(train_loader)

    logging.basicConfig(filename=save_path+'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO,filemode='a',datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("CDRNet-Train")
    logging.info("Config")
    logging.info('epoch:{}; lr:{}; batchsize:{}; trainsize:{}; save_path:{}\
                edge_weight:{} reg_loss_weight:{}'.
                format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, save_path,\
                        opt.edge_weight, opt.reg_weight))

    # loss function
    BCE_loss = torch.nn.BCEWithLogitsLoss().to(device)
    print("Let's gooooo!")
    best_f1 = 0
    best_epoch = 0
    Eva_tr = Evaluator(2)
    for epoch in range(1, (opt.epoch+1)):
        Eva_tr.reset()
        lr = adjust_lr(BCD_Model_optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        Train(train_loader, BCD_Model, EUE_Model, BCD_Model_optimizer, EUE_Model_optimizer, epoch, Eva_tr)



