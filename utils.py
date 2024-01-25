import os
import json
from pathlib import Path

import jittor as jt

import numpy as np
import trimesh

segment_colors = np.array([
    [0, 114, 189],
    [217, 83, 26],
    [238, 177, 32],
    [126, 47, 142],
    [117, 142, 48],
    [76, 190, 238],
    [162, 19, 48],
    [240, 166, 202],
])


def save_results(mesh_infos, preds, labels, name):
    if not os.path.exists('results'):
        os.mkdir('results')
    if isinstance(labels, jt.Var):
        labels = labels.data

    results_path = Path('results') / name
    results_path.mkdir(parents=True, exist_ok=True)

    for i in range(labels.shape[0]):
        mesh_path = mesh_infos['mesh_paths'][i]
        mesh_name = Path(mesh_path).stem
        mesh = trimesh.load_mesh(mesh_path, process=False)
        mesh.visual.face_colors[:, :3] = segment_colors[preds[i, :mesh.faces.shape[0]]]
        mesh.export(results_path / f'pred-{mesh_name}.ply')
        # mesh.visual.face_colors[:, :3] = segment_colors[labels[i, :mesh.faces.shape[0]]]
        # mesh.export(results_path / f'ft-{mesh_name}.ply')


def show_mesh(faces, vertices, colors=None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=None)
    if colors is not None:
        if isinstance(colors, jt.Var):
            colors = colors.data
        mesh.visual.face_colors[:, :3] = segment_colors[colors[:faces.shape[0]]]
    mesh.show()
    mesh.export(r'E:\Segmentation\SubdivNet-master\data\coseg-aliens-MAPS-256-3\save_mesh\orig.ply')


def save_mesh(save_path, faces, vertices):
    with open(save_path, 'w') as f:
        for vertex in vertices:
            f.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))
        for face in faces:
            f.write("f {} {} {}\n".format(face[0], face[1], face[2]))
    f.close()

def update_label_accuracy(preds, labels, acc):
    if isinstance(preds, jt.Var):
        preds = preds.data
    if isinstance(labels, jt.Var):
        labels = labels.data

    for i in range(preds.shape[0]):
        for k in range(len(acc)):
            if (labels[i] == k).sum() > 0:
                acc[k] += ((preds[i] == labels[i]) * (labels[i] == k)).sum() / (labels[i] == k).sum()


def compute_original_accuracy(mesh_infos, preds, labels):
    if isinstance(preds, jt.Var):
        preds = preds.data
    if isinstance(labels, jt.Var):
        labels = labels.data

    accs = np.zeros(preds.shape[0])
    for i in range(preds.shape[0]):
        raw_labels = mesh_infos['raw_labels'][i]
        raw_to_sub = mesh_infos['raw_to_sub'][i]
        accs[i] = np.mean((preds[i])[raw_to_sub] == raw_labels)

    return accs


class SegmentationMajorityVoting:
    def __init__(self, nclass, name=''):
        self.votes = {}
        self.nclass = nclass
        self.name = name

    def vote(self, mesh_infos, preds, labels):
        if isinstance(preds, jt.Var):
            preds = preds.data
        if isinstance(labels, jt.Var):
            labels = labels.data

        for i in range(preds.shape[0]):
            name = (Path(mesh_infos['mesh_paths'][i]).stem)[:-4]
            nfaces = mesh_infos['raw_labels'][i].shape[0]
            if not name in self.votes:
                self.votes[name] = {
                    'polls': np.zeros((nfaces, self.nclass), dtype=int),
                    'label': mesh_infos['raw_labels'][i],
                    'raw_path': mesh_infos['raw_paths'][i],
                }
            polls = self.votes[name]['polls']
            raw_to_sub = mesh_infos['raw_to_sub'][i]
            raw_pred = (preds[i])[raw_to_sub]
            polls[np.arange(nfaces), raw_pred] += 1

    def compute_accuracy(self, save_results=False):
        if save_results:
            if self.name:
                results_path = Path('results') / self.name
            else:
                results_path = Path('results')
            results_path.mkdir(parents=True, exist_ok=True)

        sum_acc = 0
        all_acc = {}
        for name, vote in self.votes.items():
            label = vote['label']
            pred = np.argmax(vote['polls'], axis=1)
            acc = np.mean(pred == label)
            sum_acc += acc
            all_acc[name] = acc

            if save_results:
                mesh_path = vote['raw_path']
                mesh = trimesh.load_mesh(mesh_path, process=False)
                mesh.visual.face_colors[:, :3] = segment_colors[pred[:mesh.faces.shape[0]]]
                mesh.export(results_path / f'pred-{name}.ply')
                mesh.visual.face_colors[:, :3] = segment_colors[label[:mesh.faces.shape[0]]]
                mesh.export(results_path / f'gt-{name}.ply')

        if save_results:
            with open(results_path / 'acc.json', 'w') as f:
                json.dump(all_acc, f, indent=4)
        return sum_acc / len(self.votes)


if __name__ == '__main__':
    mesh_dir = r'E:\Segmentation\SubdivNet-master\data\coseg-aliens-MAPS-256-3\test\alien'
    mesh_name = '166-000'
    mesh_path = mesh_dir + '\\{}.obj'.format(mesh_name)
    mesh = trimesh.load_mesh(mesh_path, process=False)
    seg_path = mesh_dir + '\\{}.json'.format(mesh_name)
    with open(seg_path) as f:
        segment = json.load(f)
    sub_labels = np.array(segment['sub_labels']) - 1
    mesh.visual.face_colors[:, :3] = segment_colors[sub_labels[:mesh.faces.shape[0]]]
    mesh.export(mesh_dir + '\\pred-{}.ply'.format(mesh_name))



