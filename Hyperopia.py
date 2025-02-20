# airy_disk
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def airy_psf(size=64, na=0.65,gamma=550e-9):
    '''
    :param size: PSF 이미지 크기
    :param na: 안구와 디스플레이 거리
    :param gamma: 파장
    blur정도를 감소시키려면 gamma값을 낮춤
    na값은 블러 정도와 반비례 -> 베셀 인자 k 값은 회절 인자와 반비례함
    '''
    x = np.linspace(-size/2, size/2, size)
    y = np.linspace(-size/2, size/2, size)
    x, y = np.meshgrid(x, y)

    r = np.sqrt(x**2+y**2)
    # vessel eq. parameter
    k = 2*np.pi*na/gamma

    # r=0 인 지점의 거리
    psf = np.zeros_like(r)
    mask = r!=0
    # r!=0 인 지점들
    psf[mask] = (2*special.j1(k*r*[mask])/(k*r[mask]))**2
    psf[~mask] = 1

    return psf/psf.sum()