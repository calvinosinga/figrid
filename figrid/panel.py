#!/usr/bin/env python3

import matplotlib as mpl
import numpy as np
from figrid.data_container import DataContainer

class Panel():

    def __init__(self, dataList, axis):
        self.panelList = dataList
        self.axis = axis

        self.setFunc({})
        return
    
    ##### INTERFACE WITH DATALIST ###################################

    def setArgs(self, attrs, kwargs):
        self.panelList.setArgs(attrs, kwargs)
        return

    def setFunc(self, attrs, func = None):
        if func is None:

            def _defaultPlot(data, kwargs):
                self.axis.plot(data[0], data[1], kwargs)
                return
            
            self.panelList.setFunc(attrs, _defaultPlot)
        
        else:
            self.panelList.setFunc(attrs, func)
        return

    def makeFill(self, attrs, kwargs):
        filldc = self.panelList.makeFill(attrs)
        
        def _fillFunc(data, fill_kwargs):
            self.axis.fill_between(data[0], data[1], data[2], 
                    **fill_kwargs)
            return

        filldc.setFunc(_fillFunc)
        filldc.setArgs(kwargs)
        return

    def plot(self):
        for dc in self.panelList.getData():
            dc.plot()
        return
    
        
