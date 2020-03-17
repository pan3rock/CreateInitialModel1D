import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='interpolate model')
    parser.add_argument('model_raw')
    parser.add_argument('--zc', default=40.0, type=float,
                        help='depth of the crust bottom')
    parser.add_argument('--nc', default=10, type=int,
                        help='number of layers for the curst')
    parser.add_argument('--zm', default=200.0, type=float,
                        help='depth of the interested mantle bottom')
    parser.add_argument('--nm', default=10, type=int,
                        help='number of layers for the mantle')
    parser.add_argument('--dz', nargs=2, type=float,
                        help='dz for crust and dz_max for mantle')
    args = parser.parse_args()
    file_model_raw = args.model_raw
    zc = args.zc
    zm = args.zm
    dz = args.dz

    model_raw = np.loadtxt(file_model_raw)
    z = model_raw[:, 1]
    ind_c = np.argwhere(z > zc)[0][0]
    z1 = np.arange(0, zc+dz[0], dz[0])
    nc = len(z1)
    intp1 = [interp1d(z[:ind_c+1], model_raw[:ind_c+1, i])
             for i in range(2, 5)]
    mc_new = [np.arange(1, nc+1, dtype=int), z1]
    mc_new.extend([f(z1) for f in intp1])
    mc_new = np.asarray(mc_new).T

    ind_m = np.argwhere(z < zc)[-1][0]
    zm2 = zm - zc
    nm = round(2.0 * zm2 / (dz[0] + dz[1])) + 1
    dz = np.linspace(dz[0], dz[1], nm)
    z2 = []
    z_cum = zc
    for i in range(nm):
        z_cum += dz[i]
        z2.append(z_cum)
    z2 = np.asarray(z2)
    intp2 = [interp1d(z[ind_m:], model_raw[ind_m:, i], fill_value='extrapolate')
             for i in range(2, 5)]
    mm_new = [np.arange(nc+1, nm+nc+1), z2]
    mm_new.extend([f(z2) for f in intp2])
    mm_new = np.asarray(mm_new).T

    model_new = np.vstack([mc_new, mm_new])

    with open('mi_new.txt', 'w') as fp:
        for n, *para in model_new:
            contents = ('{:5d}' + '{:10.4f}'*4).format(int(n), *para)
            fp.write(contents + '\n')
            print(contents)

    plt.figure()
    plt.step(model_new[:, 3], model_new[:, 1])
    plt.xlabel('Vs (km/s)')
    plt.ylabel('Depth (km)')
    plt.gca().invert_yaxis()
    plt.show()
