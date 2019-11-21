
from __future__ import print_function
import sys, time, os
import random
import copy
import datetime

import numpy as np
# import pandas as pd
import json

import orio.main.tuner.search.search as orio_search
from orio.main.util.globals import *


class epmal(orio_search.Search):
    '''
    Specific arguments:
        init_size
        batch_size
        max_runs
    '''
    __INIT_SIZE = 'init_size'
    __BATCH_SIZE = 'batch_size'
    __MAX_RUNS = 'max_runs'

    def __init__(self, params):
        orio_search.Search.__init__(self, params)

        # set default values
        self.pool_size = 6000
        self.file_data = 'data.csv'
        self.file_perfs = 'perfs.csv'
        self.file_metadata = 'metadata.json'

    
    def CI(self, x):
        x = np.sort(x)
        n = x.shape[0]
        z = 1.96
        median = (x[np.floor(n/2).astype(int)] + x[np.floor((n-1)/2).astype(int)]) / 2
        lower_bound = x[np.floor((n - z * np.sqrt(n)) / 2).astype(int)]
        upper_bound = x[min(np.ceil(1 + (n + z * np.sqrt(n)) / 2).astype(int), n-1)]
        return median, lower_bound, upper_bound

    def check_ci(self, x):
        ci = 1.0 / 100
        x = np.array(x)
        c = 200 # times of repetition
        n = x.shape[0]

        n_samples = [] # number of samples
        medians = []
        lower_bounds = []
        upper_bounds = []

        x = np.sort(x)
        for s in range(10, n+1):
            n_samples.append(s)

            median = []
            lower_bound = []
            upper_bound = []
            # repeat `c` times
            for i in range(c):
                # select `s` samples from `x`, randomly
                xs = x[np.random.permutation(n)[:s]]
                res = self.CI(xs)
                median.append(res[0])
                lower_bound.append(res[1])
                upper_bound.append(res[2])
            
            if np.mean(median) * (1+ci) > np.mean(upper_bound) and np.mean(median) * (1-ci) < np.mean(lower_bound):
                return True
        return False


    def searchBestCoord(self, startCoord=None):
        info('---- SEARCH METHOD: alsearch ----')
        info('  -- START... ')
        # info('  ## PARAMS:\n', self.params)


        # Generate coords pool randomly
        iterations = 0
        while iterations <  min(self.pool_size, self.space_size):
            coord = self.getRandomCoord()
            # test if the performance params are valid
            perf_params = self.coordToPerfParams(coord)
            perf_params_t = copy.copy(perf_params)
            try:
                is_valid = eval(self.constraint, perf_params_t, dict(self.input_params))
            except Exception as e:
                err('failed to evaluate the constraint expression: "%s"\n%s %s' \
                    % (self.constraint, e.__class__.__name__, e))
            if not is_valid:
                continue
            max_rounds = 10
            rounds = 0
            t1 = time.time()
            perf_samples = [] 
            next_iteration = False
            while True:
                rounds += 1
                try:
                    perf_cost = self.getPerfCosts([coord])
                    perf_samples.extend(perf_cost.items()[0][1][0])
                except IndexError:
                    next_iteration = True
                    break
                except:
                    exit(-1)
                if self.check_ci(perf_samples) or rounds >= max_rounds:
                    break
            self.ptdriver.cleanup()
            if next_iteration == True:
                continue

            t2 = time.time()
            iterations += 1
            
            compile_time = t2-t1 - np.sum(perf_samples)
            perf_median = np.median(perf_samples)
            perf_std = np.std(perf_samples)

            print('[%4d/%4d] %s Compile-Transform time: %4.fs, Execution time [%3d](mean, std): (%.4f, %.4f)' % ( \
                    iterations, self.pool_size,
                    datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S'),
                    compile_time, len(perf_samples), np.mean(perf_samples), perf_std), end=' ')

            perf_params = self.coordToPerfParams(coord)
            if not os.path.exists(self.file_data):
                with open(self.file_data, 'w') as f:
                    columns = list(perf_params.keys())
                    columns_str = reduce(lambda x, y: x + ',' + y, columns)
                    columns_str += ',start_time,end_time,compile_time,exe_time'
                    f.write(columns_str + '\n')
                    print('File Created, ', end='')
                # open(self.file_perfs, 'w').close()

            perf_str = ('%s' % list(perf_samples))[1:-1].replace(' ', '')
            row_str = ('%s' % (perf_params.values() + [t1, t2, compile_time, perf_median]))[1:-1].replace(' ', '')

            with open(self.file_data, 'a') as f:
                f.write(row_str + '\n')
                print('Stored.', end=' ')

            with open(self.file_perfs, 'a') as f:
                f.write(perf_str + '\n')
                print('Stored.')

        info('  END.')
        exit()

