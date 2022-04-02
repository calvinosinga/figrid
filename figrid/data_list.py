from figrid.data_container import DataContainer
import numpy as np

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
        unique_attr = []
        for dc in self.dclist:
            keys = list(dc.attrs.keys())
            for k in keys:
                if k not in unique_attr:
                    unique_attr.append(k)
                
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

    def getMatching(self, desired_attrs):
        matches = []
        for dc in self.dclist:
            if dc.isMatch(desired_attrs):
                matches.append(dc)
        
        return matches
    
    ##### POST-PROCESS DATA #########################################

    def makeFill(self, attrs, fillkwargs):
        attrs['figrid_process'] = 'no key found'
        matches = self.getMatching(attrs)
        if len(matches) >= 1:
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
                
                args = {'visible':False, 'zorder' : -1, 
                        'label':'_nolegend_'}
                m.setArgs(args)
            
            filldc = DataContainer([x, ymins, ymaxs])
            attrs['figrid_process'] = 'fill'
            filldc.update(attrs)

            def _plotFill(ax, data, kwargs):
                ax.fill_between(data[0], data[1], data[2], **kwargs)
                return
            
            filldc.setFunc(_plotFill)
            filldc.setArgs(fillkwargs)
            self.append(filldc)
        return
    
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
    
    def plot(self, ax, attrs = {}):
        if attrs:
            matches = self.getMatching(attrs)
            for m in matches:
                m.plot(ax)
        
        else:
            for dc in self.dclist:
                dc.plot(ax)
        
        return
            
