import pickle
import sys


def main():
    ind_sub = int(sys.argv[1]) - 1
    with open('subarray.pkl', 'rb') as fp:
        sub = pickle.load(fp)

    _, stations = sub[ind_sub]

    sta_loc = dict()
    with open('station.txt', 'r') as fp:
        for line in fp:
            sta, lat, lon = line.split()
            sta_loc[sta] = (lat, lon)

    with open('station_subarray.txt', 'w') as fp:
        for sta in stations:
            fp.write(('{:12s}'*3).format(sta, *sta_loc[sta]) + '\n')


if __name__ == '__main__':
    main()
