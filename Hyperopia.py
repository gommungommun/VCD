# airy_disk
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def airy_psf(size=64, na=0.65,wavelength=550e-9):
    '''
    :param size: PSF 이미지 크기
    :param na: 안구와 디스플레이 거리
    :param wavelength: 파장
    '''
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    x, y = np.meshgrid(x, y)

    r = np.sqrt(x**2+y**2)
    k = 2*np.pi*na/wavelength

    # r=0 인 지점의 거리
    psf = np.zeros_like(r)
    mask = r!=0
    psf[mask] = (2*special.j1(k*r*[mask])/(k*r[mask]))**2
    psf[~mask] = 1

    return psf/psf.sum()