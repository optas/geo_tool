'''
Created on Feb 14, 2018

@author: optas
'''

# soligs.Mesh
def sample_faces(self, n_samples, at_least_one=True, seed=None):
        """Generates a point cloud representing the surface of the mesh by sampling points
        proportionally to the area of each face.

        Args:
            n_samples (int) : number of points to be sampled in total
            at_least_one (int): Each face will have at least one sample point (TODO: broken fix)
        Returns:
            numpy array (n_samples, 3) containing the [x,y,z] coordinates of the samples.

        Reference :
          http://chrischoy.github.in_out/research/barycentric-coordinate-for-mesh-sampling/
          [1] Barycentric coordinate system

          \begin{align}
            P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
          \end{align}
        """

        face_areas = self.area_of_triangles()
        face_areas = face_areas / np.sum(face_areas)

        n_samples_per_face = np.round(n_samples * face_areas)

        if at_least_one:
            n_samples_per_face[n_samples_per_face == 0] = 1

        n_samples_per_face = n_samples_per_face.astype(np.int)
        n_samples_s = int(np.sum(n_samples_per_face))

        if seed is not None:
            np.random.seed(seed)

        # Control for float truncation (breaks the area analogy sampling)
        diff = n_samples_s - n_samples
        indices = np.arange(self.num_triangles)
        if diff > 0:    # we have a surplus.
            rand_faces = np.random.choice(indices[n_samples_per_face >= 1], abs(diff), replace=False)
            n_samples_per_face[rand_faces] = n_samples_per_face[rand_faces] - 1
        elif diff < 0:
            rand_faces = np.random.choice(indices, abs(diff), replace=False)
            n_samples_per_face[rand_faces] = n_samples_per_face[rand_faces] + 1

        # Create a vector that contains the face indices
        sample_face_idx = np.zeros((n_samples, ), dtype=int)

        acc = 0
        for face_idx, _n_sample in enumerate(n_samples_per_face):
            sample_face_idx[acc: acc + _n_sample] = face_idx
            acc += _n_sample

        r = np.random.rand(n_samples, 2)
        A = self.vertices[self.triangles[sample_face_idx, 0], :]
        B = self.vertices[self.triangles[sample_face_idx, 1], :]
        C = self.vertices[self.triangles[sample_face_idx, 2], :]
        P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
            np.sqrt(r[:, 0:1]) * r[:, 1:] * C
        return P, sample_face_idx
