import numpy as np

class Graph:
    """
    Gait graph for MediaPipe-based lower-body + trunk skeleton (14 joints)
    Node ordering must match your GAIT_JOINTS order
    """

    def __init__(self, labeling_mode='spatial'):
        self.num_node = 14
        self.edge = self._get_edge()
        self.A = self._get_adjacency_matrix(labeling_mode)

    def _get_edge(self):
        # self-links
        self_link = [(i, i) for i in range(self.num_node)]

        # anatomical connections (indices correspond to GAIT_JOINTS order)
        neighbor_link = [
            # head
            (0, 1),          # left eye – right eye

            # shoulders
            (2, 3),

            # torso
            (2, 4), (3, 5),  # shoulders → hips

            # left leg
            (4, 6), (6, 8), (8,10), (10,12),

            # right leg
            (5, 7), (7, 9), (9,11), (11,13),
        ]

        return self_link + neighbor_link

    def _get_adjacency_matrix(self, labeling_mode):
        A = np.zeros((1, self.num_node, self.num_node))
        for i, j in self.edge:
            A[0, i, j] = 1
            A[0, j, i] = 1
        return A
