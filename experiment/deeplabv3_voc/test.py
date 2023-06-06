# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2

from config import cfg
# from datasets.generateData import generate_dataset
from datasets.VOCDataset import VOCDataset
from net.generateNet import generate_net
import torch.optim as optim
from net.sync_batchnorm.replicate import patch_replication_callback

from torch.utils.data import DataLoader
from models.segmentation.unet import UNetResnet18
from models.segmentation.deeplab_v3 import DeeplabV3Resnet50, DeeplabV3Resnet101
from models.segmentation.deeplab_v3plus import DeeplabV3Plus

MODELS = {
    "UNetResnet18": UNetResnet18,
    "DeeplabV3Resnet50": DeeplabV3Resnet50,
    "DeeplabV3Resnet101": DeeplabV3Resnet101,
    "DeeplabV3Plus": DeeplabV3Plus,
}


def test_net():
    # dataset = generate_dataset(cfg.DATA_NAME, cfg, 'val')
    dataset = VOCDataset('VOC2012', cfg, 'val', False, run_test=True)

    dataloader = DataLoader(dataset,
                            batch_size=cfg.TEST_BATCHES,
                            shuffle=False,
                            num_workers=cfg.DATA_WORKERS)

    model_name = 'DeeplabV3Plus'
    model_kwargs = {}
    model_kwargs['max_epochs'] = 1
    model_kwargs['dataloader_length'] = 1
    print('dataloader_length: ', model_kwargs['dataloader_length'])
    net = MODELS[model_name](class_names=dataset.classes,
                             ignore_index=dataset.ignore_index,
                             visualizer_kwargs=dataset.visualizer_kwargs,
                             **model_kwargs
                             )
    # model.load_state_dict(torch.load(cfg.TEST_CKPT)['state_dict'])
    # model.eval()
    # net = generate_net(cfg)
    print('net initialize')
    if cfg.TEST_CKPT is None:
        raise ValueError('test.py: cfg.MODEL_CKPT can not be empty in test period')

    print('Use %d GPU' % cfg.TEST_GPUS)
    device = torch.device('cuda')
    # if cfg.TEST_GPUS > 1:
    # 	net = nn.DataParallel(net)
    # 	patch_replication_callback(net)
    net.to(device)

    print('start loading model %s' % cfg.TEST_CKPT)
    model_dict = torch.load(cfg.TEST_CKPT, map_location=device)
    state_dict = model_dict['state_dict']
    net.load_state_dict(model_dict)

    net.eval()
    result_list = []
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(dataloader):
            name_batched = sample_batched['name']
            row_batched = sample_batched['row']
            col_batched = sample_batched['col']

            [batch, channel, height, width] = sample_batched['image'].size()
            multi_avg = torch.zeros((batch, cfg.MODEL_NUM_CLASSES, height, width), dtype=torch.float32).to(0)
            for rate in cfg.TEST_MULTISCALE:
                inputs_batched = sample_batched['image_%f' % rate].to(0)
                predicts = net(inputs_batched).to(0)
                predicts_batched = predicts.clone()
                del predicts
                if cfg.TEST_FLIP:
                    inputs_batched_flip = torch.flip(inputs_batched, [3])
                    predicts_flip = torch.flip(net(inputs_batched_flip), [3]).to(0)
                    predicts_batched_flip = predicts_flip.clone()
                    del predicts_flip
                    predicts_batched = (predicts_batched + predicts_batched_flip) / 2.0

                predicts_batched = F.interpolate(predicts_batched, size=None, scale_factor=1 / rate, mode='bilinear',
                                                 align_corners=True)
                multi_avg = multi_avg + predicts_batched
                del predicts_batched

            multi_avg = multi_avg / len(cfg.TEST_MULTISCALE)
            result = torch.argmax(multi_avg, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(batch):
                row = row_batched[i].item()
                col = col_batched[i].item()
                #	max_edge = max(row,col)
                #	rate = cfg.DATA_RESCALE / max_edge
                #	new_row = row*rate
                #	new_col = col*rate
                #	s_row = (cfg.DATA_RESCALE-new_row)//2
                #	s_col = (cfg.DATA_RESCALE-new_col)//2

                #	p = predicts_batched[i, s_row:s_row+new_row, s_col:s_col+new_col]
                p = result[i, :, :]
                p = cv2.resize(p, dsize=(col, row), interpolation=cv2.INTER_NEAREST)
                result_list.append({'predict': p, 'name': name_batched[i]})

            print('%d/%d' % (i_batch, len(dataloader)))
    dataset.save_result(result_list, cfg.MODEL_NAME)
    dataset.do_python_eval(cfg.MODEL_NAME)
    print('Test finished')


if __name__ == '__main__':
    test_net()
