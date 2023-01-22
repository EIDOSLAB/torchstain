import jax
from jax import lax
from jax import numpy as jnp
from torchstain.base.normalizers import HENormalizer
from jax import tree_util

class JaxMacenkoNormalizer(HENormalizer):
    def __init__(self):
        super().__init__()

        self.HERef = jnp.array([[0.5626, 0.2159],
                                [0.7201, 0.8012],
                                [0.4062, 0.5581]])
        self.maxCRef = jnp.array([1.9705, 1.0308])

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = jnp.reshape(OD, (-1, 3)).T

        # determine concentrations of the individual stains
        C = jnp.linalg.lstsq(HE, Y, rcond=None)[0]

        return C

    def __compute_matrices(self, I, Io, alpha, beta):
        I = I.reshape((-1, 3))

        # calculate optical density
        OD = -jnp.log((I.astype(jnp.float32) + 1) / Io)

        # compute eigenvectors
        mask = ~jnp.any(OD < beta, axis=1)
        cov = jnp.cov(OD.T, fweights=mask.astype(jnp.int32))
        _, eigvecs = jnp.linalg.eigh(cov)

        Th = OD.dot(eigvecs[:, 1:3])

        phi = jnp.arctan2(Th[:, 1], Th[:, 0])

        phi = jnp.where(mask, phi, jnp.inf)
        pvalid = mask.mean() # proportion that is valid and not masked

        minPhi = jnp.percentile(phi, alpha * pvalid)
        maxPhi = jnp.percentile(phi, (100 - alpha) * pvalid)

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
        C = self.__find_concentration(OD, HE)

        # normalize stain concentrations
        maxC = jnp.array([jnp.percentile(C[0, :], 99), jnp.percentile(C[1, :],99)])

        return HE, C, maxC

    @jax.jit
    def fit(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        self.HERef = HE
        self.maxCRef = maxC

    @jax.jit
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

        if stains:
            # unmix hematoxylin and eosin
            H = jnp.multiply(Io, jnp.exp(jnp.expand_dims(-self.HERef[:, 0], axis=1).dot(jnp.expand_dims(C2[0, :], axis=0))))
            H = jnp.clip(H, 0, 255)
            H = jnp.reshape(H.T, (h, w, c)).astype(jnp.uint8)

            E = jnp.multiply(Io, jnp.exp(jnp.expand_dims(-self.HERef[:, 1], axis=1).dot(jnp.expand_dims(C2[1, :], axis=0))))
            E = jnp.clip(E, 0, 255)
            E = jnp.reshape(E.T, (h, w, c)).astype(jnp.uint8)

        return Inorm, H, E
    
    def _tree_flatten(self):
        children = ()  # arrays / dynamic values
        aux = ()  # static values
        return (), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

tree_util.register_pytree_node(
    JaxMacenkoNormalizer, JaxMacenkoNormalizer._tree_flatten,JaxMacenkoNormalizer._tree_unflatten
)
