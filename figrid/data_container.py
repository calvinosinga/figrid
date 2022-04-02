#!/usr/bin/env python3

import numpy as np
import copy
    
class DataContainer():

    def __init__(self, data):
        self.attrs = {}
        self.data = data
        self.default_key = 'no key found'

        self.plotArgs = {}
        self.plotFunc = self._defaultPlot
        return
    
    ##### ATTR ACCESS/MANAGEMENT ####################################

    def get(self, key):
        try:
            return self.attrs[key]
        except KeyError:
            return self.default_key
    
    def add(self, key, attr_value, overwrite = True):
        if not key in self.attrs or overwrite:
            self.attrs[key] = attr_value
        return
    
    def update(self, new_dict, overwrite = True):
        for k,v in new_dict.items():
            self.add(k, v, overwrite)
        return
    
    ##### DATA ACCESS/MANAGEMENT ####################################

    def getData(self):
        return self.data
    
    def setData(self, data):
        self.data = data
        return
    
    ##### USED FOR ORGANIZING FIGURE ################################

    def isMatch(self, desired_attrs):
        isMatch = True

        for k, v in desired_attrs.items():
            self_val = self.get(k)
            if isinstance(v, list):
                isMatch = (isMatch and self_val in v)
            else:
                isMatch = (isMatch and self_val == v)

        return isMatch
    
    ##### PLOT DATA ACCESS/MANAGEMENT ###############################

    def setFunc(self, func):
        self.plotFunc = func
        return
    
    def plot(self, ax):
        self.plotFunc(ax, self.data, self.plotArgs)
        return
    
    def _defaultPlot(self, ax, data, kwargs):
        ax.plot(data[0], data[1], **kwargs)
        return
    
    def setArgs(self, kwargs):
        self.plotArgs.update(kwargs)
        return
    
    def getArgs(self):
        return self.plotArgs
