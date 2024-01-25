import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse

import numpy as np
from tensorboardX import SummaryWriter

import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam
from jittor.lr_scheduler import MultiStepLR

jt.flags.use_cuda = 1

from tqdm import tqdm

from data import SegmentationDataset
from network import MeshDeepLab
from mesh_tensor import to_mesh_tensor
from utils import show_mesh
from utils import save_results
from utils import update_label_accuracy
from utils import compute_original_accuracy
from utils import SegmentationMajorityVoting


def train(net, optim, dataset, writer, epoch):
    net.train()
    acc = 0
    for meshes, labels, _ in tqdm(dataset, desc=str(epoch)):
        mesh_tensor = to_mesh_tensor(meshes)
        mesh_labels = jt.int32(labels)
        outputs, _, _ = net(mesh_tensor)
        loss = nn.cross_entropy_loss(outputs.unsqueeze(dim=-1), mesh_labels.unsqueeze(dim=-1), ignore_index=-1)
        optim.step(loss)

        preds = np.argmax(outputs.data, axis=1)
        acc += np.sum((labels == preds).sum(axis=1) / meshes['Fs'])
        writer.add_scalar('loss', loss.data[0], global_step=train.step)
        train.step += 1
    acc /= dataset.total_len

    print(f'Epoch #{epoch}: train acc = {acc}')
    writer.add_scalar('train-acc', acc, global_step=epoch)


@jt.single_process_scope()
def test(net, dataset, writer, epoch):
    net.eval()
    acc = 0
    oacc = 0
    parts = 4
    label_acc = np.zeros(parts)
    name = "coseg-aliens"
    voted = SegmentationMajorityVoting(parts, name)

    with jt.no_grad():
        for meshes, labels, mesh_infos in tqdm(dataset, desc=str(epoch)):
            mesh_tensor = to_mesh_tensor(meshes)
            mesh_labels = jt.int32(labels)
            outputs = net(mesh_tensor)[0]
            preds = np.argmax(outputs.data, axis=1)

            batch_acc = (labels == preds).sum(axis=1) / meshes['Fs']
            batch_oacc = compute_original_accuracy(mesh_infos, preds, mesh_labels)
            acc += np.sum(batch_acc)
            oacc += np.sum(batch_oacc)
            update_label_accuracy(preds, mesh_labels, label_acc)
            voted.vote(mesh_infos, preds, mesh_labels)
            save_results(mesh_infos, preds, mesh_labels, name)

    acc /= dataset.total_len
    oacc /= dataset.total_len
    voacc = voted.compute_accuracy(save_results=True)
    writer.add_scalar('test-acc', acc, global_step=epoch)
    writer.add_scalar('test-oacc', oacc, global_step=epoch)
    writer.add_scalar('test-voacc', voacc, global_step=epoch)

    # Update best results
    if test.best_oacc < oacc:
        if test.best_oacc > 0:
            os.remove(os.path.join('checkpoints', name, f'oacc-{test.best_oacc:.4f}.pkl'))
        net.save(os.path.join('checkpoints', name, f'oacc-{oacc:.4f}.pkl'))
        test.best_oacc = oacc

    if test.best_voacc < voacc:
        if test.best_voacc > 0:
            os.remove(os.path.join('checkpoints', name, f'voacc-{test.best_voacc:.4f}.pkl'))
        net.save(os.path.join('checkpoints', name, f'voacc-{voacc:.4f}.pkl'))
        test.best_voacc = voacc

    print('test acc = ', acc)
    print('test acc [original] =', oacc, ', best =', test.best_oacc)
    print('test acc [original] [voted] =', voacc, ', best =', test.best_voacc)
    print('test acc per label =', label_acc / dataset.total_len)


if __name__ == '__main__':
    mode = 'train'
    name = "coseg-aliens"
    batch_size = 8
    dataroot = 'E:/Segmentation/SubdivNet-master/data/coseg-aliens-MAPS-256-3/'
    save_frequency = 50

    parts = 4  # segmentation parts
    backbone = 'resnet50'
    global_pool = 'mean'

    lr = 2e-2
    lr_gamma = 0.1
    lr_milestones = [50, 100, 150]
    weight_decay = 0

    net = MeshDeepLab(13, parts, backbone, global_pool=global_pool)
    optim = Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepLR(optim, milestones=lr_milestones, gamma=lr_gamma)

    writer = SummaryWriter("logs/" + name)
    print('name:', name)

    augments = ['scale', 'orient']

    train_dataset = SegmentationDataset(dataroot=dataroot, batch_size=batch_size, train=True,
                                        shuffle=True, num_workers=0, augments=augments)
    test_dataset = SegmentationDataset(dataroot=dataroot, batch_size=batch_size, shuffle=False,
                                       train=False, num_workers=0)

    checkpoint_path = os.path.join('checkpoints', name)
    checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint = "C:/Users/vcc/PycharmProjects/SubDivNet-reproduction/checkpoints/coseg-aliens/coseg-aliens-latest.pkl"
    if checkpoint is not None:
        print('parameters: loaded from ', checkpoint)
        net.load(checkpoint)

    train.step = 0
    test.best_oacc = 0
    test.best_voacc = 0

    if mode == 'train':
        for epoch in range(500):
            train(net, optim, train_dataset, writer, epoch)
            test(net, test_dataset, writer, epoch)
            scheduler.step()
            if epoch % save_frequency == 0:
                net.save(checkpoint_name)
    else:
        test(net, test_dataset, writer, 0)


