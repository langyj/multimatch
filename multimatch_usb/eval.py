# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset
from pathlib import Path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--strong_n', type=int, default=2)
    parser.add_argument('--stronger_n', type=int, default=4)
    parser.add_argument('--num_labels', type=int, default=100)
    parser.add_argument(
        "--ulb_num_labels",
        type=int,
        default=None,
        help="number of labels for unlabeled data, used for determining the maximum "
             "number of labels in imbalanced setting",
    )
    parser.add_argument('--attack', type=bool, default=False)
    parser.add_argument('--attack_target', type=bool, default=False)

    args = parser.parse_args()

    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path)
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''

    net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
    keys = net.load_state_dict(load_state_dict)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # specify these arguments manually
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    dataset_dict = get_dataset(args, 'fixmatch', args.dataset, args.num_labels, args.num_classes, args.data_dir, False)
    eval_dset = dataset_dict['eval']
    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=4)

    # acc = 0.0
    # test_feats = []
    # test_preds = []
    # test_probs = []
    # test_labels = []
    # with torch.no_grad():
    #     for data in eval_loader:
    #         image = data['x_lb']
    #         target = data['y_lb']
    #
    #         image = image.type(torch.FloatTensor).cuda()
    #         feat = net(image, only_feat=True)
    #         logit = net(feat, only_fc=True)
    #         prob = logit.softmax(dim=-1)
    #         pred = prob.argmax(1)
    #
    #         acc += pred.cpu().eq(target).numpy().sum()

    #         test_feats.append(feat.cpu().numpy())
    #         test_preds.append(pred.cpu().numpy())
    #         test_probs.append(prob.cpu().numpy())
    #         test_labels.append(target.cpu().numpy())
    # test_feats = np.concatenate(test_feats)
    # test_preds = np.concatenate(test_preds)
    # test_probs = np.concatenate(test_probs)
    # test_labels = np.concatenate(test_labels)

    # print(f"Test Accuracy: {acc / len(eval_dset) * 100}")

    if args.attack:
        from torchattacks import *

        # attacks = ["PGD", "RFGSM", "TPGD"]
        attacks = ["PGD"]

        for attack in attacks:
            func = getattr(sys.modules[__name__], attack)
            if args.attack_target:
                atk = PGD(net, eps=4 / 255, alpha=2 / 225, steps=1, random_start=True, targeted=True)
                atk.set_mode_targeted_by_function(target_map_function=lambda images, labels: (labels + 1) % args.num_classes)
            else:
                atk = PGD(net, eps=4 / 255, alpha=2 / 225, steps=1, random_start=True)
            atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            correct = 0
            total = 0
            for data in eval_loader:
                image = data['x_lb']
                target = data['y_lb']

                image = image.type(torch.FloatTensor).cuda()
                target = target.type(torch.FloatTensor).cuda()
                with torch.no_grad():
                    feat = net(image, only_feat=True)
                    logit = net(feat, only_fc=True)
                    prob = logit.softmax(dim=-1)
                    pred = prob.argmax(1)
                adv_image = atk(image, pred)
                with torch.no_grad():
                    adv_feat = net(adv_image, only_feat=True)
                    adv_logit = net(adv_feat, only_fc=True)
                    adv_prob = adv_logit.softmax(dim=-1)
                    adv_pred = adv_prob.argmax(1)
                total += target.size(0)
                correct += adv_pred.eq(target).cpu().numpy().sum()
            print(
                f"Success rate of {type(atk).__name__}{'-' + atk.attack_class.__name__ if hasattr(atk, 'attack_class') else ''}: {100 - 100 * float(correct) / total}%\n")
