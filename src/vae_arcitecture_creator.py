

def conv_output(W, K, P, S):
    return (W - K + 2*P)/S + 1


def deconv_output(W, K, P, S):
    return S*(W - 1) + K - 2*P
