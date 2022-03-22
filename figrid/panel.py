#!/usr/bin/env python3

import matplotlib as mpl
import numpy as np
from data_container import DataContainer

class Panel():

    def __init__(self, dataList, panelAttr):
        self.attr = panelAttr
        self.panelList = dataList
        self.axis = None
        return

    def setAxis(self, axis):
        self.axis = axis
        return
        
    def plot(self, ax):
        for dc in self.panelList:
            kwargs = self.plotArgs[dc.get(self.attr)]
            dc.plot(ax, kwargs)

        # set ticks

        # set legend
        return
    
    def setTicks(self, axis, which, kwargs):
        
        if axis == 'x' or axis == 'both':
            self.xtickArgs[which].update(kwargs)
        if axis == 'y' or axis == 'both':
            self.ytickArgs[which].update(kwargs)
        
        return
    
    def setLegend(self, kwargs):
        self.legendArgs.update(kwargs)
        return
        
