#!/usr/bin/env python3

import numpy as np

class DataList():
    def __init__(self, dclist = []):
        self.dclist = dclist
        return
    
    def loadHdf5(self):
        return
    
    def loadResults(self, results):
        for r in results:
            data = [r.xvalues, r.yvalues, r.zvalues]
            dc = DataContainer(data)

            dc.update(r.props)
            self.dclist.append(dc)
        return
    
    def append(self, dataContainer):
        self.dclist.append(dataContainer)
        return
    
    def getAttrVals(self, key):
        unique_vals = []
        for dc in self.dclist:
            attrVal = dc.get(key)

            if not attrVal == dc.default_key or \
                    not attrVal in unique_vals:
                
                unique_vals.append(attrVal)
        
        return unique_vals

    def removeMatching(self, desired_attrs):
        for dc in self.dclist:
            if dc.isMatch(desired_attrs):
                self.dclist.remove(dc)
        return

    def getMatching(self, desired_attrs, return_as = 'list'):
        matches = []
        for dc in self.dclist:
            if dc.isMatch(desired_attrs):
                matches.append(dc)
        
        if return_as == 'list':
            return matches
        elif return_as == 'DataList':
            return DataList(matches)
        else:
            msg = "return_as not supported (list, DataList)"
            raise ValueError(msg)
    
    def makeFill(self, attrs):

        dl = self.dclist
        matches = dl.getMatching(attrs)
        x, y = matches[0].getData()
        ymins = np.ones_like(y)
        ymaxs = np.ones_like(y)
        for m in matches:
            _, y = m.getData()
            ymins = np.minimum(y, ymins)
            ymaxs = np.maximum(y, ymaxs)
            dl.remove(m)

        filldc = DataContainer([x, ymins, ymaxs])
        attrs['post_process'] = 'fill'
        filldc.update(attrs)
        dl.append(filldc)
        return
    
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

        
        return isMatch
    
    ######### PLOT DATA ############################################

    def setFunc(self, func):
        self.plotFunc = func
        return
    
    def plot(self):
        self.plotFunc(self.data, self.plotArgs)
        return