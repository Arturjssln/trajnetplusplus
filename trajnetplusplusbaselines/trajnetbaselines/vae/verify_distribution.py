import argparse
import datetime
import json

import torch
import numpy as np
import pysparkling
from scipy import stats
import matplotlib.pyplot as plt


def read_log(path):
    sc = pysparkling.Context()
    return (sc
            .textFile(path)
            .filter(lambda line: line.startswith("{'z_val':"))
            .map(lambda line: json.loads(line.replace("\'", "\"").strip('json:'))['z_val'])
            .collect())

def shapiro_test(data):
    p_values = []
    for dim in range(data.shape[1]):
        shapiro_test = stats.shapiro(data[:, dim])
        p_values.append(shapiro_test.pvalue)
    return np.array(p_values)

def plot_histogram(data):
    for dim in range(data.shape[1]):
        plt.figure()
        plt.hist(data[:, dim])
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='/Users/arturjesslen/Documents/GitHub/trajnet++/trajnetplusplusbaselines/z_val.log',
                        type=str, help='path to log file')
    args = parser.parse_args()
    data = np.array(read_log(args.log_file))
    p_values = shapiro_test(data)
    P_VALUE_THRESHOLD = 0.05 / data.shape[1]
    print(p_values)
    print(p_values > P_VALUE_THRESHOLD)
    plot_histogram(data[:, 5:10])


if __name__ == '__main__':
    main()
