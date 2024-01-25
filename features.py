import os

import trimesh

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from tensorboardX import SummaryWriter

import jittor as jt

jt.flags.use_cuda = 1

from data import SegmentationDataset
from network import MeshDeepLab
from mesh_tensor import MeshTensor
from utils import show_mesh, save_mesh


#  可视化特征图
def show_feature_mesh(conv_mesh):
    conv_feat = np.array(conv_mesh.feats[0][0:7])
    for i in range(conv_feat.shape[-1]):
        mu = np.sqrt(
            conv_feat[1][i] * conv_feat[1][i] + conv_feat[2][i] * conv_feat[2][i] + conv_feat[3][i] * conv_feat[3][i])
        conv_feat[1][i] = conv_feat[1][i] / mu
        conv_feat[2][i] = conv_feat[2][i] / mu
        conv_feat[3][i] = conv_feat[3][i] / mu
    conv_feat[4] = (conv_feat[4] - conv_feat[4].min()) / (conv_feat[4].max() - conv_feat[4].min())
    conv_feat[5] = (conv_feat[5] - conv_feat[5].min()) / (conv_feat[5].max() - conv_feat[5].min())
    conv_feat[6] = (conv_feat[6] - conv_feat[6].min()) / (conv_feat[6].max() - conv_feat[6].min())
    conv_feat = np.expand_dims(conv_feat, 0)
    conv_feat = jt.float32(conv_feat)
    conv_mesh = MeshTensor(conv_mesh.faces, conv_feat, conv_mesh.Fs)
    show_mesh(faces=np.array(conv_mesh.faces[0]), vertices=mesh_tensor.compute_vertices()[0])


if __name__ == '__main__':
    mode = 'test'
    name = "coseg-aliens"
    dataroot = 'E:/Segmentation/SubdivNet-master/data/coseg-aliens-MAPS-256-3/'

    backbone = 'resnet50'
    global_pool = 'mean'

    net = MeshDeepLab(13, 4, backbone, global_pool=global_pool)

    writer = SummaryWriter("logs/" + name)
    print('name:', name)

    augments = ['scale', 'orient']

    N = 1
    dataset = SegmentationDataset(dataroot=dataroot, batch_size=N, shuffle=False,
                                  train=False, num_workers=0)[0]
    faces, feats, Fs = dataset[:3]
    np_faces = np.zeros((N, Fs, 3), dtype=np.int32)
    for i in range(N):
        np_faces[i, :] = faces
    feats = np.expand_dims(feats, 0)
    faces = jt.int32(np_faces)
    feats = jt.float32(feats)
    Fs = jt.int32(Fs)
    mesh_tensor = MeshTensor(faces, feats, Fs)
    vertices = mesh_tensor.compute_vertices()[0]
    show_mesh(faces=np.array(mesh_tensor.faces[0]), vertices=vertices)

    # checkpoint_path = os.path.join('checkpoints', name)
    # checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')
    # os.makedirs(checkpoint_path, exist_ok=True)
    #
    # checkpoint = "C:/Users/vcc/PycharmProjects/SubDivNet-reproduction/checkpoints/coseg-aliens/coseg-aliens-latest.pkl"
    # if checkpoint is not None:
    #     print('parameters: loaded from ', checkpoint)
    #     net.load(checkpoint)
    #
    # net.eval()
    # outputs, conv1_mesh, conv2_mesh = net(mesh_tensor)
    #
    # save_path = r'E:\Segmentation\SubdivNet-master\data\coseg-aliens-MAPS-256-3\save_mesh'
    # show_feature_mesh(conv1_mesh)
    # show_feature_mesh(conv2_mesh)
    #
    # mesh1 = trimesh.Trimesh(faces=np.array(conv1_mesh.faces[0]), vertices=vertices)
    # mesh1.export(save_path + '\\conv1.ply')
    # mesh2 = trimesh.Trimesh(faces=np.array(conv2_mesh.faces[0]), vertices=vertices)
    # mesh2.export(save_path + '\\conv2.ply')


