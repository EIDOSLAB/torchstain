import jax
from jax import lax
from jax import numpy as jnp
from torchstain.base.normalizers import HENormalizer
from functools import partial


class JaxMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = jnp.array([[0.5626, 0.2159],
                          [0.7201, 0.8012],
                          [0.4062, 0.5581]])
        self.maxCRef = jnp.array([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io=240, beta=0.15):
        # calculate optical density
        OD = -jnp.log((I.astype(jnp.float32) + 1) / Io)

        # remove transparent pixels
        #ODhat = OD[~jnp.any(OD < beta, axis=1)]

        # jax dont support dynamic shapes, but this: https://stackoverflow.com/a/71694754
        # @FIXME: Not identical to numpy approach above!
        mask = ~jnp.any(OD < beta, axis=1)
        indices = jnp.where(mask, size=len(mask), fill_value=255)
        ODhat = OD.at[indices].get()  # mode="fill", fill_value=0)

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])

        phi = jnp.arctan2(That[:, 1], That[:, 0])

        minPhi = jnp.percentile(phi, alpha)
        maxPhi = jnp.percentile(phi, 100 - alpha)

        vMin = eigvecs[:, 1:3].dot(jnp.array([(jnp.cos(minPhi), jnp.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(jnp.array([(jnp.cos(maxPhi), jnp.sin(maxPhi))]).T)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        HE = lax.cond(
            vMin[0, 0] > vMax[0, 0],
            lambda x: jnp.array((x[0], x[1])).T,
            lambda x: jnp.array((x[0], x[1])).T,
            (vMin[:, 0], vMax[:, 0])
        )

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = jnp.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = jnp.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1, 3))

        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)

        # compute eigenvectors
        _, eigvecs = jnp.linalg.eigh(jnp.cov(ODhat.T))

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = jnp.array([jnp.percentile(C[0, :], 99), jnp.percentile(C[1, :],99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    @partial(jax.jit, static_argnums=(0,))
    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        h, w, c = I.shape
        I = I.reshape((-1, 3))

        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta)

        maxC = jnp.divide(maxC, self.maxCRef)
        C2 = jnp.divide(C, maxC[:, jnp.newaxis])

        # recreate the image using reference mixing matrix
        Inorm = jnp.multiply(Io, jnp.exp(-self.HERef.dot(C2)))
        Inorm = jnp.clip(Inorm, 0, 255)
        Inorm = jnp.reshape(Inorm.T, (h, w, c)).astype(jnp.uint8)

        H, E = None, None

        if False:
            # unmix hematoxylin and eosin
            H = jnp.multiply(Io, jnp.exp(jnp.expand_dims(-self.HERef[:, 0], axis=1).dot(jnp.expand_dims(C2[0, :], axis=0))))
            H = jnp.clip(H, 0, 255)
            H = jnp.reshape(H.T, (h, w, c)).astype(jnp.uint8)

            E = jnp.multiply(Io, jnp.exp(jnp.expand_dims(-self.HERef[:, 1], axis=1).dot(jnp.expand_dims(C2[1, :], axis=0))))
            E = jnp.clip(E, 0, 255)
            E = jnp.reshape(E.T, (h, w, c)).astype(jnp.uint8)

        return Inorm, H, E
