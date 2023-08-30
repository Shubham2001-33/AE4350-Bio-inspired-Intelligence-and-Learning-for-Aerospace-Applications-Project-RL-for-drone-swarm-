#!/usr/bin/env python3
'''
Generates plots for QL training
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import yaml

def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

class Plotter:
    def __init__(self):
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

        self.cfg = cfg

    def gen_title(self):
        return 'MiniGridEnv: {} x {} / Agents: {}'.format(
                                                        self.cfg['env']['grid_size'],
                                                        self.cfg['env']['grid_size'],
                                                        self.cfg['env']['agents']
                                                        )

    def gen_text(self):
        txt_keys = ''
        txt_vals = ''
        for k, v in self.cfg['env'].items():
            # txt += '{:<13} {:>4}\n'.format(k,v)
            txt_keys += '{}\n'.format(k)
            txt_vals += '{}\n'.format(v)

        txt_keys += '\n'
        txt_vals += '\n'

        for k, v in self.cfg['ql'].items():
            if k == 'episodes':
                continue

            txt_keys += '{}\n'.format(k)
            txt_vals += '{}\n'.format(v)

        # print(txt)
        return txt_keys, txt_vals

    def save_plot(self):
        filename = "st_plt_{}x{}_o{}_a{}_r{}_t{}.png".format(
                                                        self.cfg['env']['grid_size'],
                                                        self.cfg['env']['grid_size'],
                                                        self.cfg['env']['obstacles'],
                                                        self.cfg['env']['agents'],
                                                        self.cfg['env']['obs_radius'],
                                                        self.cfg['env']['reward_type']
                                                        )
        plt.savefig(filename)

    def plot_steps(self, steps_list):
        if len(steps_list) >= 2:
            x = np.arange(0, len(steps_list))

            plt.clf()

            regression_type = self.cfg['plt']['regression_type']
            if regression_type == None:
                plt.plot(x,steps_list, 'yo')
            elif regression_type == 'lin':
                fit = np.polyfit(x,steps_list,1) # linear regression
                fit_fn = np.poly1d(fit)
                plt.plot(x,steps_list, 'yo', x, fit_fn(x), '--k')
            elif regression_type == 'quad':
                fit = np.polyfit(x,steps_list,2) # quadratic regression
                fit_fn = np.poly1d(fit)
                plt.plot(x,steps_list, 'yo', x, fit_fn(x), '--k')
            elif regression_type == 'exp':
                popt, pcov = curve_fit(exp_func, x, steps_list) # exponential regression
                fit_fn = exp_func(x, *popt)
                plt.plot(x,steps_list, 'yo', x, exp_func(x, *popt), '--k')
            else:
                assert False, "invalid regression type"

            plt.title(self.gen_title())
            plt.text(0.80, 0.6, self.gen_text()[0], transform=plt.gca().transAxes, fontsize=8, ha='left')
            plt.text(0.97, 0.6, self.gen_text()[1], transform=plt.gca().transAxes, fontsize=8, ha='right')
            plt.xlabel('Episode')
            plt.ylabel('Steps to completion')

            plt.ion()
            plt.ioff()
            self.save_plot()
            plt.show()
