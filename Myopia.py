import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def gaussian_psf(size=64, sigma=3):
    '''
    :param size: PSF이미지의 크기
    :param sigma: 가우시안 분포의 표준편차
    :return:
    '''
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    x, y = np.meshgrid(x,y)

    pos = np.dstack((x,y))
    rv = multivariate_normal([0,0],[[sigma**2,0],[0,sigma**2]])
    psf = rv.psf(pos)

    return psf/psf.sum()

psf = gaussian_psf()