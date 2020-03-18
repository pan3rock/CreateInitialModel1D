from netCDF4 import Dataset
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from obspy.geodetics.base import gps2dist_azimuth
from scipy.interpolate import interp1d

NP = 9
NLO = 360
NLA = 180


def extend_model(model):
    model = np.asarray(model)
    ret = []
    for i in range(model.shape[0]-1):
        ret.append(model[i, :])
        line = model[i, :].copy()
        line[0] = model[i+1, 0]
        ret.append(line)
    ret.append([ret[-1][0] + (ret[-1][0] - ret[-2][0])/2.0,  *(ret[-1][1:])])
    return np.asarray(ret)


def load_crust1():
    crust_model = dict(
        vp=np.zeros([NP, NLA, NLO]),
        vs=np.zeros([NP, NLA, NLO]),
        rho=np.zeros([NP, NLA, NLO]),
        bnds=np.zeros([NP, NLA, NLO]))

    path = os.path.dirname(__file__)
    for s in ['vp', 'vs', 'rho', 'bnds']:
        fname = path + '/models/crust1.' + s
        fp = open(fname, 'r')
        for j in range(NLA):
            for i in range(NLO):
                line = np.array(fp.readline().split()).astype(np.float)
                crust_model[s][:, j, i] = line
        fp.close()

    return crust_model


def interp_model(model, num_layer, ice_exist=False):
    model = np.asarray(model)
    ret = []
    if ice_exist:
        ind_start = 1
    else:
        ind_start = 0
    for i in range(ind_start, model.shape[0]-1):
        if i % 2 == 0:
            line = [model[i:i+2, 0].sum()/2.0, *model[i, 1:]]
            ret.append(line)
        if i % 2 == 1:
            line = [model[i, 0], *(model[i:i+2, 1:].sum(axis=0)/2.0)]
            ret.append(line)
    model2 = np.asarray(ret)

    z_new = np.linspace(model[ind_start, 0], model[-1, 0], num_layer)
    mintp = [interp1d(model2[:, 0], model2[:, i+1], fill_value='extrapolate')
             for i in range(3)]
    model_new = np.asarray([f(z_new) for f in mintp]).T
    model_new = np.hstack([z_new.reshape(-1, 1), model_new])
    if ice_exist:
        model_new = np.vstack([model[0, :].reshape(1, -1), model_new])
    return model_new


def find_crust(crust_model, lat, lon, num_layer):
    if lon > 180:
        lon = lon - 360
    if lon < -180:
        lon = lon + 360
    ilon = int(round(180 + lon))
    ilat = int(round(90 - lat))

    cm = dict()
    for k, v in crust_model.items():
        cm[k] = v[:, ilat, ilon]
    if cm['bnds'][0] != cm['bnds'][1]:
        # water exists
        print("Water layer exists at ({:9.2f}, {:9.2f})".format(lat, lon))
        return None, False

    ret = []
    depth = 0.
    ice_exist = False
    if cm['bnds'][2] != cm['bnds'][1]:
        ret.append([depth, cm['rho'][1], cm['vs'][1], cm['vp'][1]])
        depth = cm['bnds'][1] - cm['bnds'][2]
        ice_exist = True
    for i in range(3):
        if cm['vp'][2+i] > 1.0e-10:
            ret.append([depth, cm['rho'][2+i], cm['vs'][2+i], cm['vp'][2+i]])
            depth += cm['bnds'][2+i] - cm['bnds'][3+i]
    for i in range(3):
        ret.append([depth, cm['rho'][5+i], cm['vs'][5+i], cm['vp'][5+i]])
        depth += cm['bnds'][5+i] - cm['bnds'][6+i]

    return np.asarray(ret), ice_exist


def load_mean_model():
    path = os.path.dirname(__file__)
    x = Dataset(path + '/models/MEAN.nc', 'r')
    mean_model = np.vstack(
        [6371-x['radius'][()], x['rho'], x['vs'], x['vp']]).T
    return mean_model[::-1, :]


def main():
    parser = argparse.ArgumentParser(description='Create a 1D initial model'
                                     ' based on CRUST 1.0 and MEAN reference model.')
    parser.add_argument('file_station',
                        help='file for stations in the subarray')
    parser.add_argument('--zmax', default=200.0, type=float,
                        help='zmax of target model')
    parser.add_argument('--nc', default=10, type=int,
                        help='number of layers for the crust')
    parser.add_argument('--nm', default=10, type=int,
                        help='number of layers for the mantle')
    args = parser.parse_args()
    file_station = args.file_station
    zmax = args.zmax
    nc = args.nc
    nm = args.nm

    crust_model = load_crust1()
    mean_model = load_mean_model()
    stations = []
    with open(file_station, 'r') as fp:
        for line in fp:
            stations.append(np.array(line.split()[1:]).astype(np.float))
    stations = np.asarray(stations)

    c_lat, c_lon = np.mean(stations, axis=0)
    # plt.figure()
    # plt.plot(c_lon, c_lat, 'ro')
    # plt.plot(stations[:, 1], stations[:, 0], 'k.')
    # plt.xlabel('latitude')
    # plt.ylabel('longitude')
    # plt.tight_layout()
    # plt.show()

    dists = []
    for lat, lon in stations:
        dist, _, _ = gps2dist_azimuth(lat, lon, c_lat, c_lon)
        dists.append(dist)
    weight1 = np.asarray(dists) / np.sum(dists)

    cms = []
    weight = []
    for i, (lat, lon) in enumerate(stations):
        cm, ice_exist = find_crust(crust_model, lat, lon, nc)
        if cm is None:
            continue
        weight.append(weight1[i])
        plt.step(cm[:, 1], cm[:, 0])
        dmax_crust = cm[-1, 0]
        itm = np.argwhere(mean_model[:, 0] > dmax_crust)[0][0]
        ibm = np.argwhere(mean_model[:, 0] <= 50.0)[-1][0]
        if ibm > itm:
            cm_ext = mean_model[itm:ibm+1, :]
            cm = np.vstack([cm, cm_ext])
        cm2 = extend_model(cm)
        cm = interp_model(cm2, nc, ice_exist)
        cms.append(cm)
    plt.gca().invert_yaxis()
    cms = np.asarray(cms)
    cm = np.zeros((cms.shape[1], cms.shape[2]))
    cm[:, 0] = cms[0, :, 0]
    for i in range(3):
        cm[:, 1+i] = np.average(cms[:, :, 1+i], weights=weight, axis=0)

    itm = np.argwhere(mean_model[:, 0] <= 50.0)[-1][0]
    ibm = np.argwhere(mean_model[:, 0] < zmax*1.2)[-1][0]
    mm = mean_model[itm:ibm, :]
    mm = extend_model(mm)
    mm = interp_model(mm, nm)
    model_new = np.vstack([cm, mm])

    with open('model_init.txt', 'w') as fp:
        for i, row in enumerate(model_new):
            fp.write(('{:5d}'+'{:9.2f}'*4+'\n').format(i+1, *row))

    plt.figure()
    vs = model_new[:, 2]
    z = model_new[:, 0]
    plt.step(vs, z)
    plt.ylim([0, zmax])
    plt.xlabel('Vs (km/s)')
    plt.ylabel('Depth (km)')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
