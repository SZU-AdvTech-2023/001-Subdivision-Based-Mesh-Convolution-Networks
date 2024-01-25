import jittor as jt
import numpy as np
import time


def to_mesh_tensor(meshes):
    return MeshTensor(jt.int32(meshes['faces']),
                      jt.float32(meshes['feats']),
                      jt.int32(meshes['Fs']),
                      )


class MeshTensor:
    """
    A MeshTensor object stores a batch of triangular meshes with
    multidimensional arrays.

    All faces are stored in a 3-dimensional tensor. To support a batch of
    variable number of faces, an additional array Fs is used to hold every mesh's
    number of faces.
    """

    def __init__(self, faces: jt.Var, feats: jt.Var, Fs: jt.Var = None,
                 # vertices: jt.Var = None,
                 cache=None):
        """
        Parameters
        ------------
        faces: (Batch_size, Face_num, 3) int32
            Array of triangular faces.
        feats: (Batch_size, Channel_num, Face_num) float32
            Array of face features.
        Fs: (Batch_size,) int32, optional
            Array of number of faces in each mesh.
            If not specified, Fs is set to n.
        cache: dict, optional
            things calculated from faces to avoid repeated calculation for the
            same mesh.
        """
        self.faces = faces
        self.feats = feats
        # self.vertices = vertices

        self.N, self.C, self.F = feats.shape

        if Fs is not None:
            self.Fs = Fs
            assert self.F == self.Fs.max().data[0]
        else:
            self.Fs = jt.ones(self.N, dtype="int32") * self.F

        self._cache = cache if cache is not None else {}

    def updated(self, new_feats):
        """
        Return a new MeshTensor with its feats updated.

        A shortcut to obtain a new MeshTensor with new features.
        """
        assert new_feats.shape[0] == self.N
        assert new_feats.shape[2] == self.F
        return MeshTensor(self.faces, new_feats, self.Fs, self._cache)

    @property
    def shape(self):
        return self.feats.shape

    @property
    def V(self) -> int:
        """ Maximum number of vertices in the mini-batch """
        if not 'V' in self._cache:
            self._cache['V'] = int((self.faces.max() + 1).data[0])
        return self._cache['V']

    @property
    def Vs(self) -> jt.Var:
        """
        Number of vertices in each mesh.

        Returns
        ------------
        (N,) int32
        """
        if not 'Vs' in self._cache:
            self._cache['Vs'] = self.faces.max(dim=1).max(dim=1) + 1
        return self._cache['Vs']

    # @property
    # def new_vertices(self) -> jt.Var:
    #     if not 'vertices' in self._cache:
    #         self._cache['vertices'] = self.compute_vertices()
    #     return self._cache['vertices']

    @property
    def degrees(self) -> jt.Var:
        """
        Degrees of vertices.

        Return:
        ------------
        (N, V) int32
        """
        if not 'degrees' in self._cache:
            face_degrees = jt.ones((self.N, self.F, 3), dtype=jt.int32)
            self._cache['degrees'] = face_degrees.reindex_reduce(
                # 统计各数字在 faces 中出现的次数
                op='add',
                shape=[self.N, self.V],
                indexes=[
                    # 预定义变量 i0, i1, ..., in 是输入变量的下标索引。
                    'i0', '@e0(i0, i1, i2)'
                ],
                extras=[self.faces, self.Fs],
                overflow_conditions=['i1 >= @e1(i0)']  # i1 不大于该网格面片个数
            )
        return self._cache['degrees']

    @property
    def FAF(self) -> jt.Var:
        """
        FAF (Face-adjacent-faces) indexs the adjacencies.

        Returns:
        ------------
        (N, F, 3) int32
        """
        if not 'FAF' in self._cache:
            self._cache['FAF'] = self.compute_face_adjacency_faces()
        return self._cache['FAF']

    @property
    def FAFP(self) -> jt.Var:
        """ The previous face of current face's adjacent faces """
        if not 'FAFP' in self._cache:
            self._cache['FAFP'], self._cache['FAFN'] = self.compute_face_adjacency_reordered()
        return self._cache['FAFP']

    @property
    def FAFN(self) -> jt.Var:
        """ The next face of current face's adjacent faces """
        if not 'FAFN' in self._cache:
            self._cache['FAFP'], self._cache['FAFN'] = self.compute_face_adjacency_reordered()
        return self._cache['FAFN']

    # @property
    # def VAF(self) -> list[list[list[int]]]:
    #     """ Vertex-adjacent-faces
    #
    #     Return :
    #     ------------
    #     (N, Vs, n_ad) int32
    #     """
    #     if not 'VAF' in self._cache:
    #         self._cache['VAF'] = self.compute_vertex_adjacency_faces()
    #     return self._cache['VAF']

    # @property
    # def VAV(self) -> list[list[list[int]]]:
    #     """ Vertex-adjacent-vertices"""
    #     if not 'VAV' in self._cache:
    #         self._cache['VAV'] = self.compute_vertex_adjacency_vertices()
    #     return self._cache['VAV']

    def __add__(self, other: jt.Var) -> jt.Var:
        new_feats = self.feats + other.feats
        return self.updated(new_feats)

    def __radd__(self, other: jt.Var) -> jt.Var:
        return self.__add__(other)

    def __sub__(self, other: jt.Var) -> jt.Var:
        new_feats = self.feats - other.feats
        return self.updated(new_feats)

    def __rsub__(self, other: jt.Var) -> jt.Var:
        new_feats = other.feats - self.feats
        return self.updated(new_feats)

    def __repr__(self):
        return f'MeshTensor: N={self.N}, C={self.C}, F={self.F}'

    def compute_face_adjacency_faces(self) -> jt.Var:
        """
        Compute face adjacency faces.

        Returns:
        ------------
        (N, F, 3) int32
        """
        FAF = jt.zeros_like(self.faces)
        for i in range(self.N):
            F = self.Fs[i].data[0]
            E = jt.concat([
                self.faces[i, :F, [1, 2]],
                self.faces[i, :F, [2, 0]],
                self.faces[i, :F, [0, 1]],
            ], dim=0)

            # 给 edges 哈希值, 使得顶点相同的两个边(即两个半边)的值相等, 顶点不同两个边哈希值不等
            E_hash = E.min(dim=1).astype('int64') * E.max() + E.max(dim=1)

            # S is index of sorted E_hash.
            S, _ = jt.argsort(E_hash)  # 将数组升序，返回其索引, 这样哈希值相同的两个边将会配对

            # S[:, 0] and S[:, 1] are pairs of half-edge
            S = S.reshape(-1, 2)

            # Based on the construction rule of E,
            #   1. S % F is the face id
            #   2. S // F is the order of edge in F
            FAF[i, S[:, 0] % F, S[:, 0] // F] = S[:, 1] % F
            FAF[i, S[:, 1] % F, S[:, 1] // F] = S[:, 0] % F

        return FAF

    def compute_face_adjacency_reordered(self) -> tuple[jt.Var, jt.Var]:
        """
        Compute previous and next face of current face's adjacency faces.
        See Fig.3 k=5, d=1 in paper

        Returns:
        ------------
        FAFP: (N, F, 3) int32
        FAFN: (N, F, 3) int32
        """
        FAF = self.FAF

        FAF_ext = FAF.reindex(
            shape=[self.N, self.F, 3, 3],
            indexes=[
                'i0', '@e0(i0, i1, i2)', 'i3',
            ],
            extras=[FAF],
        )

        # shift adjacency so that
        for _ in range(2):
            FAF_ext = FAF_ext.reindex(
                shape=[self.N, self.F, 3, 3],
                indexes=[
                    'i0', 'i1', 'i2', '@e0(i0, i1, i2, 0) == i1 ? i3 : (i3 > 0 ? i3 - 1 : 2)'
                ],
                extras=[FAF_ext]
            )

        FAFP = FAF_ext[:, :, :, 2]
        FAFN = FAF_ext[:, :, :, 1]
        return FAFP, FAFN

    def compute_vertex_adjacency_faces(self) -> list[list[list[int]]]:
        """
        Compute three of vertex adjacency faces.

        Returns:
        ------------
        (N, V, Deg) int32
        """
        # t1 = time.time()
        # VAF = [[[] for _ in range(self.Vs.data[i])] for i in range(self.N)]
        # for i0 in range(self.N):
        #     for i1 in range(self.F):    # 遍历网格所有面
        #         for i2 in range(3):     # 遍历面上3个顶点
        #             VAF[i0][self.faces[i0][i1][i2].data[0]].append(i1)  #
        # t2 = time.time()  # 60s
        # print("查找邻接边用时", (t2-t1), "s\n")

        # t1 = time.time()
        v_index = np.arange(3 * self.F).reshape((self.F, 3))
        VAF = [[[] for _ in range(self.Vs.data[i])] for i in range(self.N)]
        for i0 in range(self.N):
            for v in range(self.V):
                idx = list(v_index[self.faces[i0] == v] // 3)  # 顶点v的所有邻接边的索引
                VAF[i0][v] = idx
        # t2 = time.time()    # 28s
        # print("查找邻接边用时", (t2-t1), "s\n")
        return VAF

    # def compute_vertex_adjacency_vertices(self) -> list[list[list[int]]]:
    #     """
    #     Compute three of vertex adjacency faces.
    #
    #     Returns:
    #     ------------
    #     (N, V, Deg) int32
    #     """
    #     VAV = [[[] for _ in range(self.Vs.data[i])] for i in range(self.N)]
    #     for i in range(self.N):
    #         F = self.Fs[i].data[0]
    #
    #         E = jt.concat([
    #             self.faces[i, :F, [0, 1]],
    #             self.faces[i, :F, [1, 2]],
    #             self.faces[i, :F, [2, 0]]
    #         ], dim=0)
    #
    #         for e in E:
    #             VAV[i][e[0]].append(e[1].data[0])
    #     return VAV

    def compute_vertices(self) -> np.array:
        """ compute vertices position
            the feats should contain face center and face normal.
            the position of vertices is the solution of
            Ax=b

            Returns:
            -------------
            (N, V, 3) int32
        """
        self.VAF = self.compute_vertex_adjacency_faces()
        vertices = np.zeros((self.N, self.V, 3))
        for i0 in range(self.N):
            nF = self.Fs.data[i0]
            b_arr = np.zeros(nF)
            b_visited = [False for _ in range(nF)]
            n_arr = np.array(self.feats[i0][1:4])   # 面法向量
            p_arr = np.array(self.feats[i0][4:7])   # 面中心点坐标
            for v in range(self.Vs.data[i0]):
                n_vaf = len(self.VAF[i0][v])
                A = np.zeros((n_vaf, 3))
                b = np.zeros(n_vaf)
                for i, f in enumerate(self.VAF[i0][v]):
                    A[i] = n_arr[:, f]
                    if not b_visited[f]:
                        b_arr[f] = np.dot(n_arr[:, f], p_arr[:, f])
                        b_visited[f] = True
                    b[i] = b_arr[f]
                U, S, V = np.linalg.svd(A)
                if S[1] < 1e-1:  # adjacency faces are almost on a same plane
                    # 不妨直接取邻接面的重心的平均
                    v_adjacency = np.zeros((1, 3))
                    for i, f_ad in enumerate(self.VAF[i0][v]):
                        v_adjacency += p_arr[:, f_ad]
                    vertices[i0][v] = v_adjacency / (i + 1)
                    continue

                elif S[2] < 1e-2:  # adjacency faces intersect with a line
                    # 取邻接面重心的平均到交线的垂足
                    v_adjacency = np.zeros((3))
                    for i, f_ad in enumerate(self.VAF[i0][v]):
                        v_adjacency += p_arr[:, f_ad]
                    v_adjacency = v_adjacency / (i + 1)
                    newA, newb = np.zeros((3, 3)), np.zeros((3))
                    newA[0], newb[0] = A[0], b[0]
                    for ai, bi in zip(A[1:], b[1:]):
                        if abs(ai[0] - A[0][0]) > 1e-2 or abs(ai[1] - A[0][1]) > 1e-2 or abs(ai[2] - A[0][2]) > 1e-2:
                            newA[1], newb[1] = ai, bi
                            break
                    newA[2] = np.cross(newA[0], newA[1])
                    newb[2] = np.dot(newA[2], v_adjacency)
                    U, S, V = np.linalg.svd(newA)
                    b = newb
                x = np.matmul(U.T, b)
                vertices[i0][v] = np.matmul(V.T, x[:3] / S)
        return vertices

    def inverse_loop_pool(self, op='max', pooled_feats=None):
        """
        Pooling with the inverse loop scheme.

        Parameters:
        ------------
        op: {'max', 'mean'}, optional
            Reduction method of pooling. The default is 'max'.
        pooled_feats: (N, C, F) float32, optional
            Specifying the feature after pooling.

        Returns:
        ------------
        MeshTensor after 4-to-1 face merge.
        """
        pooled_Fs = self.Fs // 4

        pooled_faces = self.faces.reindex(
            shape=[self.N, self.F // 4, 3],
            indexes=[
                'i0',
                'i1 + @e0(i0) * i2',
                '0',
            ],
            extras=[pooled_Fs],
            overflow_conditions=['i1 >= @e0(i0)'],
            overflow_value=0
        )

        if pooled_feats is None:
            pooled_feats = self.feats.reindex(
                shape=[self.N, self.C, self.F // 4, 4],  # 将4个待池化点切片
                indexes=[
                    'i0',
                    'i1',
                    'i2 + @e0(i0) * i3'
                ],
                extras=[pooled_Fs],
                overflow_conditions=['i2 >= @e0(i0)'],
                overflow_value=0
            )

            if op == 'max':
                pooled_feats = jt.argmax(pooled_feats, dim=3)[1]
            elif op == 'mean':
                pooled_feats = jt.mean(pooled_feats, dim=3)
            else:
                raise Exception('Unsupported pooling operation')
        else:
            assert pooled_feats.shape[0] == self.N
            assert pooled_feats.shape[2] == self.F // 4

        return MeshTensor(pooled_faces, pooled_feats, pooled_Fs)

    def loop_subdivision(self):
        """
        Computes the faces of meshes after one time of loop subdivision.
        """
        subdiv_faces = jt.zeros([self.N, self.F * 4, 3], dtype=jt.float32)
        for i in range(self.N):
            V = self.faces[i].max() + 1
            F = self.Fs[i].data[0]

            E = jt.concat([
                self.faces[i, :F, [0, 1]],
                self.faces[i, :F, [1, 2]],
                self.faces[i, :F, [2, 0]]
            ], dim=0)
            E_hash = E.min(dim=1).astype('int64') * E.max() + E.max(dim=1)
            E2F, _ = jt.argsort(E_hash)
            F2E = jt.zeros_like(E2F)
            F2E[E2F] = jt.index((E.shape[0],), 0) // 2  # 边的编号, 两个半边编号相同

            # 更新边点的索引
            E2 = V + F2E[:F]
            E0 = V + F2E[F:F * 2]
            E1 = V + F2E[F * 2:]
            # 更新面的索引
            subdiv_faces[i, :F * 4] = jt.concat([
                jt.stack([self.faces[i, :F, 0], E2, E1], dim=-1),
                jt.stack([self.faces[i, :F, 1], E0, E2], dim=-1),
                jt.stack([self.faces[i, :F, 2], E1, E0], dim=-1),
                jt.stack([E0, E1, E2], dim=-1)
            ], dim=0)
        return subdiv_faces

    def loop_unpool(self, mode, ref_faces=None, ref_cache=None):
        """
        Unpooling with the loop subdivision scheme.

        See Sec 3.5 and Fig.5.

        Parameters:
        ------------
        mode: {'nearest', 'bilinear'}
            Algorithm used for unpooling.
        ref_faces: (N, F, 3) int32, optional
            If specified, the returned MeshTensor uses the reference faces
            instead of computing by loop subdivision. This parameter can speed
            up dense prediction networks with pairs of pooling and unpooling.
            The default is None.
        ref_cache: dict, optional
            If specified, the returned MeshTensor uses the reference cache. The
            default is None.

        Returns:
        ------------
        MeshTensor after 1-to-4 face split.
        """
        unpooled_Fs = self.Fs * 4

        if ref_faces is not None:
            unpooled_faces = ref_faces
            unpooled_cache = ref_cache
        else:
            unpooled_faces = self.loop_subdivision()
            unpooled_cache = None

        if mode == 'nearest':
            unpooled_feats = jt.concat([self.feats] * 4, dim=2)
        elif mode == 'bilinear':
            neighbor_feats = self.feats.reindex(
                shape=[self.N, self.C, self.F, 3],
                indexes=[
                    'i0', 'i1', '@e0(i0, i2, i3)'
                ],
                extras=[self.FAF]
            )
            unpooled_feats = jt.concat([
                # 注意与 loop_subdivision 中面的细分得到的索引对应
                (self.feats * 2 + neighbor_feats[..., 1] + neighbor_feats[..., 2]) / 4,
                (self.feats * 2 + neighbor_feats[..., 2] + neighbor_feats[..., 0]) / 4,
                (self.feats * 2 + neighbor_feats[..., 0] + neighbor_feats[..., 1]) / 4,
                self.feats
            ], dim=2)
        else:
            raise Exception(f'Unsupported unpool mode: {mode}')

        return MeshTensor(unpooled_faces, unpooled_feats, unpooled_Fs, unpooled_cache)

    def dilated_face_adjacencies(self, dilation: int):
        if dilation <= 1:
            raise Exception('dilation must be greater than zero')

        DFA = jt.code(
            shape=[self.N, self.F, 3],
            dtype=jt.int32,
            inputs=[self.FAF, jt.zeros((dilation, 0), dtype=jt.int32)],
            cpu_src="""
                @alias(FAF, in0)
                int dilation = in1_shape0;

                for (int bs = 0; bs < out_shape0; ++bs)
                    for (int f = 0; f < out_shape1; ++f)
                        for (int k = 0; k < out_shape2; ++k) {
                            int a = f;
                            int b = @FAF(bs, f, k);
                            for (int d = 1; d < dilation; ++d) {
                                int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                                a = b;
                                if ((d & 1) == 0) {       // go to next
                                    b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                                } else {                // go to previous
                                    b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                                }
                            }
                            @out(bs, f, k) = b;
                        }
            """,
            cuda_src="""
                __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
                    @PRECALC
                    @alias(FAF, in0)
                    int dilation = in1_shape0;
                    int N = in0_shape0;
                    int F = in0_shape1;

                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int bs = idx / (F * 3);
                    int f = idx / 3 % F;
                    int k = idx % 3;

                    if (bs >= N)
                        return;

                    int a = f;
                    int b = @FAF(bs, f, k);
                    for (int d = 1; d < dilation; ++d) {
                        int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                        a = b;
                        if ((d & 1) == 0) {     // go to next
                            b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                        } else {                // go to previous
                            b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                        }
                    }
                    @out(bs, f, k) = b;
                }

                dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/512+1, 512>>>(@ARGS);
            """
        )

        return DFA

    def convolution_kernel_pattern(self, kernel_size=3, dilation=1):
        if kernel_size == 1:
            raise Exception(f'kernel size 1 does not have convolution pattern')

        if kernel_size == 3:
            if dilation == 1:
                return self.FAF
            else:
                return self.dilated_face_adjacencies(dilation)
        elif kernel_size == 5:
            if dilation == 1:
                return jt.stack([
                    self.FAFN[:, :, 0],
                    self.FAF[:, :, 0],
                    self.FAFP[:, :, 0],
                    self.FAFN[:, :, 1],
                    self.FAF[:, :, 1],
                    self.FAFP[:, :, 1],
                    self.FAFN[:, :, 2],
                    self.FAF[:, :, 2],
                    self.FAFP[:, :, 2],
                ], dim=-1)
            else:
                raise Exception('Not support dilation with kernel size larger than 3 yet')
        else:
            DFA = jt.code(
                shape=[self.N, self.F, 3],  # the output shape, an integer array
                dtype=jt.int32,
                inputs=[self.FAF, jt.zeros(kernel_size, 0), jt.zeros((dilation, 0), dtype=jt.int32)],
                cpu_src="""
                    @alias(FAF, in0)
                    int kernel_size = in1_shape0;
                    int dilation = in2_shape0;

                    for (int bs = 0; bs < out_shape0; ++bs)
                        for (int f = 0; f < out_shape1; ++f)
                            for (int k = 0; k < out_shape2; ++k) {
                                int a = f;
                                int b = @FAF(bs, f, k);
                                for (int d = 1; d < 0; ++d) {
                                    int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                                    a = b;
                                    if ((d & 1) == 0) {       // go to next
                                        b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                                    } else {                // go to previous
                                        b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                                    }
                                }
                                @out(bs, f, k) = b;
                            }
                """,
                cuda_src="""
                    __global__ void dilated_face_adjacencies_kernel(@ARGS_DEF) {
                        @PRECALC
                        @alias(FAF, in0)
                        int dilation = in1_shape0;
                        int N = in0_shape0;
                        int F = in0_shape1;

                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        int bs = idx / (F * 3);
                        int f = idx / 3 % F;
                        int k = idx % 3;

                        if (bs >= N)
                            return;

                        int a = f;
                        int b = @FAF(bs, f, k);
                        for (int d = 1; d < dilation; ++d) {
                            int i = @FAF(bs, b, 0) == a ? 0 : (@FAF(bs, b, 1) == a ? 1 : 2);
                            a = b;
                            if ((d & 1) == 0) {     // go to next
                                b = @FAF(bs, b, i < 2 ? i + 1 : 0);
                            } else {                // go to previous
                                b = @FAF(bs, b, i > 0 ? i - 1 : 2);
                            }
                        }
                        @out(bs, f, k) = b;
                    }

                    dilated_face_adjacencies_kernel<<<(in0_shape0*in0_shape1*3-1)/512+1, 512>>>(@ARGS);
                """
            )

            return DFA


if __name__ == '__main__':
    from data import *

    root = 'E:/Segmentation/SubdivNet-master/data/coseg-aliens-MAPS-256-3/'
    dataset = SegmentationDataset(dataroot=root, batch_size=8, train=False,
                                  shuffle=True, num_workers=4)
    for meshes, labels, mesh_infos in tqdm(dataset):
        meshes_tensor = to_mesh_tensor(meshes)
        show_mesh(faces=np.array(meshes_tensor.faces[0]),
                  vertices=meshes_tensor.compute_vertices()[0], colors=labels[0])
        break
