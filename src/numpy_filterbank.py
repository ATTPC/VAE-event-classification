import numpy as np

H = 128
W = 128
eps = 1e-8

def attn_params(gx, gy, logsigma_sq, delta, loggamma, N):

    sigma_sq = np.exp(logsigma_sq)
    gamma = np.exp(loggamma)

    gx = (H + 1)/2 * (gx + 1)
    gy = (W + 1)/2 * (gy + 1)
    #delta = (H - 1)/(N - 1) * delta

    return gx, gy, sigma_sq, delta, gamma


def filters( gx, gy, sigma_sq, delta, gamma, N):
    i = np.arange(N, dtype=np.float32)

    mu_x = gx + (i - N/2 - 0.5) * delta  # batch_size, N
    mu_y = gy + (i - N/2 - 0.5) * delta
    # print(mu_x.get_shape(), gx.get_shape(), i.get_shape())
    a = np.arange(H, dtype=np.float32)
    b = np.arange(W, dtype=np.float32)

    A, MU_X = np.meshgrid(a, mu_x)  # batch_size, N * H
    B, MU_Y = np.meshgrid(b, mu_y)

    A = np.reshape(A, [1, N, H])
    B = np.reshape(B, [1, N, W])

    MU_X = np.reshape(MU_X, [1, N, H])
    MU_Y = np.reshape(MU_Y, [1, N, W])

    sigma_sq = np.reshape(sigma_sq, [1, 1, 1])

    Fx = np.exp(- np.square(A - MU_X)/(2*sigma_sq))
    Fy = np.exp(- np.square(B - MU_Y)/(2*sigma_sq))

    Fx = Fx / np.maximum(np.sum(Fx, 1, keepdims=True), eps)
    Fy = Fy / np.maximum(np.sum(Fy, 1, keepdims=True), eps)

    return Fx, Fy

def read_attn( x, xhat, h_dec_prev, Fx, Fy, gamma, N):

    Fx_t = np.transpose(Fx, perm=[0, 2, 1])

    x = np.reshape(x, [1, 128, 128])
    xhat = np.reshape(xhat, [1, 128, 128])

    FyxFx_t = np.reshape(np.matmul(Fy, np.matmul(x, Fx_t)), [-1, N*N])
    FyxhatFx_t = np.reshape(np.matmul(Fy, np.matmul(x, Fx_t)), [-1, N*N])

    return gamma * np.concat([FyxFx_t, FyxhatFx_t], 1)
