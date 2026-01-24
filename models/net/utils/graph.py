import numpy as np

class Graph:
    """
    MediaPipe-based gait graph (14 joints)
    Node order must match GAIT_JOINTS
    """

    def __init__(self, labeling_mode='spatial'):
        self.num_node = 14
        self.edge = self._get_edge()
        self.A = self._get_adjacency_matrix()

    def _get_edge(self):
        self_link = [(i, i) for i in range(self.num_node)]

        neighbor_link = [
            (0, 1),          # eyes
            (2, 3),          # shoulders
            (2, 4), (3, 5),  # shoulders → hips
            (4, 6), (6, 8), (8,10), (10,12),  # left leg
            (5, 7), (7, 9), (9,11), (11,13),  # right leg
        ]

        return self_link + neighbor_link

    def _get_adjacency_matrix(self):
        A = np.zeros((1, self.num_node, self.num_node))
        for i, j in self.edge:
            A[0, i, j] = 1
            A[0, j, i] = 1
        return A
