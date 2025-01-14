import tensorflow as tf
from torchstain.tf.utils import cov, percentile, solveLS

class TensorFlowMultiMacenkoNormalizer:
    def __init__(self, norm_mode="avg-post"):
        self.norm_mode = norm_mode
        self.HERef = tf.constant([[0.5626, 0.2159],
                                  [0.7201, 0.8012],
                                  [0.4062, 0.5581]], dtype=tf.float32)
        self.maxCRef = tf.constant([1.9705, 1.0308], dtype=tf.float32)

    def __convert_rgb2od(self, I, Io, beta):
        I = tf.transpose(I, perm=[1, 2, 0])  # Shape: (height, width, 3)
        OD = -tf.math.log((tf.reshape(I, [-1, tf.shape(I)[-1]]) + 1) / Io)
        ODhat = tf.boolean_mask(OD, ~tf.reduce_any(OD < beta, axis=1))

        if tf.size(ODhat) == 0:
            raise ValueError("ODhat is empty. Check image values and beta threshold.")
        
        return OD, ODhat

    def __find_phi_bounds(self, ODhat, eigvecs, alpha):
        That = tf.matmul(ODhat, eigvecs)
        phi = tf.math.atan2(That[:, 1], That[:, 0])

        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)
        return minPhi, maxPhi

    def __find_HE_from_bounds(self, eigvecs, minPhi, maxPhi):
        # Expand minPhi and maxPhi to have a second dimension
        vMin = tf.matmul(eigvecs, tf.stack([tf.cos(minPhi), tf.sin(minPhi)])[:, tf.newaxis])
        vMax = tf.matmul(eigvecs, tf.stack([tf.cos(maxPhi), tf.sin(maxPhi)])[:, tf.newaxis])

        # Concatenate along the last dimension and return
        HE = tf.where(vMin[0] > vMax[0],
                    tf.concat([vMin, vMax], axis=1),
                    tf.concat([vMax, vMin], axis=1))
        return HE

    def __find_HE(self, ODhat, eigvecs, alpha):
        minPhi, maxPhi = self.__find_phi_bounds(ODhat, eigvecs, alpha)
        return self.__find_HE_from_bounds(eigvecs, minPhi, maxPhi)

    def __find_concentration(self, OD, HE):
        # Solve linear system using the provided solveLS function
        Y = tf.transpose(OD)
        C = solveLS(HE, Y)
        return C

    def __compute_matrices_single(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io, beta)

        cov_matrix = cov(tf.transpose(ODhat))  # cov expects shape (dims, samples)
        eigvals, eigvecs = tf.linalg.eigh(cov_matrix)
        eigvecs = tf.gather(eigvecs, [1, 2], axis=1)

        HE = self.__find_HE(ODhat, eigvecs, alpha)
        C = self.__find_concentration(OD, HE)
        maxC = tf.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])
        return HE, C, maxC

    def fit(self, Is, Io=240, alpha=1, beta=0.15):
        if not isinstance(Is, list) or len(Is) == 0:
            raise ValueError("Input images should be a non-empty list of tensors.")

        for i, I in enumerate(Is):
            if not isinstance(I, tf.Tensor):
                raise ValueError(f"Image at index {i} is not a TensorFlow tensor.")
            if I.ndim != 3 or I.shape[0] != 3:
                raise ValueError(f"Image at index {i} should have shape (3, height, width).")

        if self.norm_mode == "avg-post":
            HEs, _, maxCs = zip(*[self.__compute_matrices_single(I, Io, alpha, beta) for I in Is])
            self.HERef = tf.reduce_mean(tf.stack(HEs), axis=0)
            self.maxCRef = tf.reduce_mean(tf.stack(maxCs), axis=0)
        elif self.norm_mode == "concat":
            ODs, ODhats = zip(*[self.__convert_rgb2od(I, Io, beta) for I in Is])
            OD = tf.concat(ODs, axis=0)
            ODhat = tf.concat(ODhats, axis=0)

            cov_matrix = cov(tf.transpose(ODhat))  # cov expects shape (dims, samples)
            eigvals, eigvecs = tf.linalg.eigh(cov_matrix)
            eigvecs = tf.gather(eigvecs, [1, 2], axis=1)

            HE = self.__find_HE(ODhat, eigvecs, alpha)
            C = self.__find_concentration(OD, HE)
            maxCs = tf.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

            self.HERef = HE
            self.maxCRef = maxCs
        else:
            raise ValueError("Unsupported normalization mode.")

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices_single(I, Io, alpha, beta)

        # Ensure maxCRef and maxC are broadcastable
        scaling_factors = (self.maxCRef / maxC)[:, tf.newaxis]  # Shape: [2, 1]
        C = scaling_factors * C[:2, :]  # Use only the first two rows of C

        # Reconstruct the normalized image
        Inorm = Io * tf.exp(-tf.linalg.matmul(self.HERef, C))
        Inorm = tf.clip_by_value(Inorm, 0, 255)
        Inorm = tf.transpose(tf.reshape(Inorm, [c, h, w]), perm=[1, 2, 0])  # Convert back to HWC format
        Inorm = tf.cast(Inorm, tf.int32)

        H, E = None, None

        if stains:
            # Extract the H and E components
            H = Io * tf.exp(-tf.linalg.matmul(self.HERef[:, 0:1], C[0:1, :]))
            H = tf.clip_by_value(H, 0, 255)
            H = tf.transpose(tf.reshape(H, [c, h, w]), perm=[1, 2, 0])
            H = tf.cast(H, tf.int32)

            E = Io * tf.exp(-tf.linalg.matmul(self.HERef[:, 1:2], C[1:2, :]))
            E = tf.clip_by_value(E, 0, 255)
            E = tf.transpose(tf.reshape(E, [c, h, w]), perm=[1, 2, 0])
            E = tf.cast(E, tf.int32)

        return Inorm, H, E