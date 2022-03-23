#!/usr/bin/env python3

import numpy as np
import copy

class DataList():
    def __init__(self, dclist = []):
        self.dclist = dclist
        return
    
    ##### I/O METHODS ###############################################

    def loadHdf5(self):
        return
    
    def loadResults(self, results):
        for r in results:
            data = [r.xvalues, r.yvalues, r.zvalues]
            dc = DataContainer(data)

            dc.update(r.props)
            self.dclist.append(dc)
        return

    ##### DATA ACCESS/MANAGEMENT ####################################

    def append(self, dataContainer):
        self.dclist.append(dataContainer)
        return
    
    def getAttrs(self):
        unique_attr = set()
        for dc in self.dclist:
            unique_attr.update(list(dc.attrs.keys()))
        return unique_attr

    def getAttrVals(self, key):
        unique_vals = []
        for dc in self.dclist:
            attrVal = dc.get(key)

            if not attrVal in unique_vals:
                
                unique_vals.append(attrVal)
        
        return unique_vals

    def getData(self):
        return self.dclist
        
    def removeMatching(self, desired_attrs):
        rmidx = []
        for dc in range(len(self.dclist)):
            if self.dclist[dc].isMatch(desired_attrs):
                rmidx.append(dc)
        rmidx = np.array(rmidx)
        for rm in range(len(rmidx)):

            self.dclist.pop(rmidx[rm])
            rmidx = rmidx - 1
            
        return

    def getMatching(self, desired_attrs, return_as = 'list'):
        matches = []
        for dc in self.dclist:
            #print(desired_attrs['color'])
            print(dc.attrs['color'])
            if dc.isMatch(desired_attrs):
                matches.append(dc)
        
        if return_as == 'list':
            return copy.deepcopy(matches)
        elif return_as == 'DataList':
            return DataList(copy.deepcopy(matches))
        else:
            msg = "return_as not supported (list, DataList)"
            raise ValueError(msg)
    
    ##### POST-PROCESS DATA #########################################

    def makeFill(self, attrs):
        print('datalist make fill')
        # print(attrs)
        # print(self.getAttrVals('color'))
        dl = self.dclist
        matches = self.getMatching(attrs)
        print(matches)
        data = matches[0].getData()
        x = data[0]
        y = data[1]
        ymins = np.ones_like(y) * y
        ymaxs = np.ones_like(y) * y
        for m in matches:
            data = m.getData()
            y = data[1]
            ymins = np.minimum(y, ymins)
            ymaxs = np.maximum(y, ymaxs)
        
        #self.removeMatching(attrs)
        filldc = DataContainer([x, ymins, ymaxs])
        attrs['post_process'] = 'fill'
        filldc.update(attrs)
        self.append(filldc)
        return filldc
    
    ##### INTERFACING WITH DATA CONTAINERS ##########################

    def setArgs(self, attrs, kwargs):
        matches = self.getMatching(attrs)
        for m in matches:
            m.setArgs(kwargs)
        return matches
        
    def setFunc(self, attrs, func):
        matches = self.getMatching(attrs)
        for m in matches:
            m.setFunc(func)
        return matches
    
class DataContainer():

    def __init__(self, data):
        self.attrs = {}
        self.data = data
        self.default_key = 'no key found'

        self.plotArgs = {}
        self.plotFunc = None
        return
    
    ########### DATA ACCESS/MANAGEMENT ##############################

    def get(self, key):
        try:
            return self.attrs[key]
        except KeyError:
            return self.default_key
    
    def getData(self):
        return self.data
    
    def setData(self, data):
        self.data = data
        return
    
    def add(self, key, attr_value, overwrite = True):
        if not key in self.attrs or overwrite:
            self.attrs[key] = attr_value
        return
    
    def update(self, new_dict, overwrite = True):
        for k,v in new_dict.items():
            self.add(k, v, overwrite)
        return
    
    ########### USED FOR ORGANIZING FIGURE ###########################

    def isMatch(self, desired_attrs):
        isMatch = True

        for k, v in desired_attrs.items():
            self_val = self.get(k)
            if isinstance(v, list):
                isMatch = (isMatch and self_val in v)
            else:
                isMatch = (isMatch and self_val == v)

        print(isMatch)
        return isMatch
    
    ######### PLOT DATA ############################################

    def setFunc(self, func):
        self.plotFunc = func
        return
    
    def plot(self):
        self.plotFunc(self.data, self.plotArgs)
        return
    
    def setArgs(self, kwargs):
        self.plotArgs.update(kwargs)
        return
