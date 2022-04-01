import tensorflow as tf
import tensorflow_probability as tfp

class HSCIC() :

    def __init__(self,
                 amplitude = 1.0,
                 length_scale = 0.1,
                 regularization = 0.01,
                 **kwargs):

        # kernel model
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
            amplitude = amplitude,
            length_scale = length_scale
        )
        self.regularization = regularization

    # get loss givne a single instance z
    def __call__(self,
                 Y : tf.Tensor,
                 A : tf.Tensor,
                 X : tf.Tensor,
                 H : tf.Tensor):

        # reshape arrays
        if tf.rank(A) == 1 : A = tf.reshape(A, [tf.shape(A)[0], 1])
        if tf.rank(X) == 1 : X = tf.reshape(X, [tf.shape(X)[0], 1])
        if tf.rank(Y) == 1 : Y = tf.reshape(Y, [tf.shape(Y)[0], 1])
        if tf.rank(H) == 1 : H = tf.reshape(H, [tf.shape(H)[0], 1])
        A = tf.cast(A, dtype='float32')
        X = tf.cast(X, dtype='float32')
        Y = tf.cast(Y, dtype='float32')
        H = tf.cast(H, dtype='float32')

        X = tf.concat([X, H], axis = 1)

        # get Kernel matrices
        gram_A  = self.kernel.matrix(A, A)
        gram_X  = self.kernel.matrix(X, X)
        gram_Y  = self.kernel.matrix(Y, Y)
        gram_A = tf.cast(gram_A, dtype='float32')
        gram_X = tf.cast(gram_X, dtype='float32')
        gram_Y = tf.cast(gram_Y, dtype='float32')

        # get HSCIC loss
        res = tf.map_fn(fn=lambda X: self.inner_loss(X, gram_A, gram_X, gram_Y),
                        elems=gram_X)
        res = tf.math.reduce_mean(res)

        return res

    # get loss givne a single instance z
    def inner_loss(self, X, gram_A, gram_X, gram_Y) :

        # get number of samples and make matrix W
        n_samples = tf.cast(tf.shape(gram_Y)[0], dtype = 'float32')
        identity  = tf.eye(n_samples, dtype = 'float32')
        W = gram_X + n_samples * self.regularization * identity

        # solve linear system
        if tf.rank(X) == 1: X = tf.reshape(X, [tf.shape(X)[0], 1])
        f = tf.linalg.solve(tf.transpose(W), X)
        fT = tf.transpose(f)

        # get distributions
        res = tf.einsum('ij,jk,kl', fT, gram_A * gram_Y, f)
        M = tf.einsum('ij,jk', gram_A, f)
        N = tf.einsum('ij,jk', gram_Y, f)
        res = res - 2 * tf.einsum('ij,jk', fT, M * N)
        P = tf.einsum('ij,jk,kl', fT, gram_A, f)
        Q = tf.einsum('ij,jk,kl', fT, gram_Y, f)
        res = res + P * Q

        return res