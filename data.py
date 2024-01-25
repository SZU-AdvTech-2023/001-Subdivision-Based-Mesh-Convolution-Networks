import json
import trimesh

import jittor as jt
from jittor.dataset import Dataset

jt.flags.use_cuda = 1

from tqdm import tqdm

import numpy as np
from utils import *


def random_scale(mesh: trimesh.Trimesh):
    mesh.vertices = mesh.vertices * np.random.normal(1, 0.1, size=(1, 3))
    return mesh


def mesh_normalize(mesh: trimesh.Trimesh):
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices = vertices / vertices.max()
    mesh.vertices = vertices
    return mesh


def load_mesh(path, normalize=False, augments=[], request=[]):
    mesh = trimesh.load_mesh(path, process=False)

    for method in augments:
        if method == 'scale':
            mesh = random_scale(mesh)

    if normalize:
        mesh = mesh_normalize(mesh)

    F = mesh.faces
    V = mesh.vertices
    Fs = mesh.faces.shape[0]
    Vs = mesh.vertices.shape[0]

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    # corner = V[F.flatten()].reshape(-1, 3, 3) - face_center[:, np.newaxis, :]
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([    # 每个面三个顶点的顶点法向与该面法向内积
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)  # n_faces * 3, 0~3.14
    if 'curves' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)

    return mesh.faces, feats, Fs, mesh.vertices, Vs


def load_segment(path):
    with open(path) as f:
        segment = json.load(f)
    raw_labels = np.array(segment['raw_labels']) - 1
    sub_labels = np.array(segment['sub_labels']) - 1
    raw_to_sub = np.array(segment['raw_to_sub'])

    return raw_labels, sub_labels, raw_to_sub


class SegmentationDataset(Dataset):
    def __init__(self, dataroot, batch_size, train=True, shuffle=False, num_workers=0, augments=None):
        super(SegmentationDataset, self).__init__(batch_size=batch_size, shuffle=shuffle,
                                                  num_workers=num_workers, keep_numpy_array=True,
                                                  buffer_size=134217728)
        self.batch_size = batch_size
        self.dataroot = dataroot
        self.augments = []
        if train and augments:
            self.augments = augments
        self.mode = 'train' if train else 'test'
        self.feats = ['area', 'face_angles', 'curves', 'center', 'normal']
        self.mesh_paths = []
        self.raw_paths = []
        self.seg_paths = []
        self.browse_dataroot()
        self.set_attrs(total_len=len(self.mesh_paths))  # 需要调用self.set_attrs来指定数据集加载所需的参数batch_size，total_len、shuffle

    def browse_dataroot(self):
        # dir_path.iterdir() 可以扫描某个目录下的所有路径（文件和子目录)， 打印的会是处理过的绝对路径。
        # pathlib 支持用 / 拼接路径。
        for dataset in (Path(self.dataroot) / self.mode).iterdir():
            if not dataset.is_dir():        # 判断是否为文件夹
                continue
            for obj_path in dataset.iterdir():  # 遍历该文件夹下所有文件
                if obj_path.suffix == '.obj':
                    obj_name = obj_path.stem
                    seg_path = obj_path.parent / (obj_name + '.json')   # 对应的json文件, 获取分割的label
                    raw_name = obj_name.rsplit('-', 1)[0]
                    raw_path = list(Path(self.dataroot).glob(f'raw/{raw_name}.*'))[0]   # 对应的off文件

                    self.mesh_paths.append(str(obj_path))
                    self.seg_paths.append(str(seg_path))
                    self.raw_paths.append(str(raw_path))
        self.mesh_paths = np.array(self.mesh_paths)
        self.seg_paths = np.array(self.seg_paths)
        self.raw_paths = np.array(self.raw_paths)

    def __getitem__(self, idx):
        faces, feats, Fs, vertices, Vs = load_mesh(self.mesh_paths[idx],
                                                   normalize=True,
                                                   augments=self.augments,
                                                   request=self.feats)
        raw_labels, sub_labels, raw_to_sub = load_segment(self.seg_paths[idx])
        return faces, feats, Fs, vertices, Vs, \
            raw_labels, sub_labels, raw_to_sub, self.mesh_paths[idx], self.raw_paths[idx]

    def collate_batch(self, batch):
        faces, feats, Fs, vertices, Vs, \
            raw_labels, sub_labels, raw_to_sub, mesh_paths, raw_paths = zip(*batch)
        N = len(batch)
        max_f = max(Fs)
        max_v = max(Vs)

        np_faces = np.zeros((N, max_f, 3), dtype=np.int32)
        np_feats = np.zeros((N, feats[0].shape[0], max_f), dtype=np.float32)
        np_Fs = np.int32(Fs)
        np_vertices = np.zeros((N, max_v, 3), dtype=np.float32)
        np_sub_labels = np.ones((N, max_f), dtype=np.int32) * -1

        for i in range(N):
            np_faces[i, :Fs[i]] = faces[i]
            np_feats[i, :, :Fs[i]] = feats[i]
            np_sub_labels[i, :Fs[i]] = sub_labels[i]
            np_vertices[i, :Vs[i]] = vertices[i]

        meshes = {'faces': np_faces,
                  'feats': np_feats,
                  'Fs': np_Fs,
                  'vertices': np_vertices,
                  # 'vertex_neighbor': vertex_neighbor,
                  }
        labels = np_sub_labels
        mesh_infos = {'raw_labels': raw_labels,
                      'raw_to_sub': raw_to_sub,
                      'mesh_paths': mesh_paths,
                      'raw_paths': raw_paths}
        return meshes, labels, mesh_infos


def test(dataset):
    for meshes, labels, mesh_infos in tqdm(dataset, desc='epoch'):
        show_mesh(faces=meshes['faces'][0], vertices=meshes['vertices'][0], colors=labels[0])
        break


if __name__ == '__main__':
    root = 'E:/Segmentation/SubdivNet-master/data/coseg-aliens-MAPS-256-3/'
    dataset = SegmentationDataset(dataroot=root, batch_size=8, train=True,
                                  shuffle=True, num_workers=0)
    test(dataset)
    # for epoch in range(1):
    #     for meshes, labels, mesh_infos in tqdm(dataset, desc=str(epoch)):
    #         show_mesh(faces=meshes['faces'][0], vertices=meshes['vertices'][0], colors=labels[0])
    #         save_results(mesh_infos, preds=None, labels=labels, name='coseg-alien')
    #         break
