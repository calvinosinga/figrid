#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from figrid.data_list import DataList
import copy
import matplotlib.gridspec as gspec


class Figrid():

    def __init__(self, dataList):
        self.dl = dataList
        self.rowValues = []
        self.colValues = []
        self.rowOrderFunc = self._defaultOrder
        self.colOrderFunc = self._defaultOrder

        self.panels = None
        self.setrc()
        return
    
    def setrc(self, rcparams = {}):
        if not rcparams:
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['mathtext.fontset'] = 'dejavuserif'
        else:
            plt.rcParams.update(rcparams)
        return

    ########## MAKING FIGURES #####################################

    def setRowOrder(self, ordered_list = [], func = None):
        if ordered_list:
            self.rowValues = ordered_list
        if not func is None:
            self.rowOrderFunc = func
        else:
            self.rowOrderFunc = self._defaultOrder
        return
    
    def _defaultOrder(self, attrList):
        return attrList
    
    def setColOrder(self, ordered_list = [], func = None):
        if ordered_list:
            self.colValues = ordered_list
        if not func is None:
            self.colOrderFunc = func
        else:
            self.colOrderFunc = self._defaultOrder
        return
    
    def arrange(self, rowAttr, colAttr, panel_length = 3, 
            panel_bt = 0.1, xborder = 0.25, yborder = 0.25, 
            height_ratios = None, width_ratios = None, 
            dpi = 100):
        
        panel_bt = panel_bt * panel_length
        xborder = xborder * panel_length
        yborder = yborder * panel_length

        rowValues = self.dl.getAttrVals(rowAttr)
        colValues = self.dl.getAttrVals(colAttr)
        
        rowValues = self.rowOrderFunc(rowValues)
        colValues = self.colOrderFunc(colValues)

        print('The row values for %s: %s'%(rowAttr, str(rowValues)))
        print('The column values for %s: %s'%(colAttr, str(colValues)))

        nrows = len(rowValues)
        ncols = len(colValues)
        
        self.rowValues = rowValues
        self.colValues = colValues
 
        self._makeFig(nrows, ncols, panel_length, panel_bt,
            xborder, yborder, height_ratios, width_ratios, dpi)
        
        self.panels = np.empty((nrows, ncols), dtype = object)
        for i in range(nrows):
            for j in range(ncols):
                
                panelAttr = {}
                panelAttr[rowAttr] = rowValues[i]
                panelAttr[colAttr] = colValues[j]

                dlPanel = self.dl.getMatching(panelAttr)
                
                self.panels[i, j] = DataList(copy.deepcopy(dlPanel))
        
        return


    def _makeFig(self, nrows, ncols, panel_length, panel_bt,
            xborder, yborder, height_ratios, width_ratios, dpi):

        if isinstance(xborder, float) or isinstance(xborder, int):
            xborder = [xborder, xborder]
        if isinstance(yborder, float) or isinstance(yborder, int):
            yborder = [yborder, yborder]
        if isinstance(panel_bt, float) or isinstance(panel_bt, int):
            panel_bt = [panel_bt, panel_bt]
        if height_ratios is None:
            height_ratios = np.ones(nrows) * panel_length
        else:
            # renormalize
            maxval = np.max(height_ratios)
            height_ratios /= maxval
            height_ratios *= panel_length

        if width_ratios is None:
            width_ratios = np.ones(ncols) * panel_length
        else:
            #renormalize
            maxval = np.max(width_ratios)
            width_ratios /= maxval
            width_ratios *= panel_length
        
        # creating Figure object

        figwidth = np.sum(width_ratios) + panel_bt[0] * (ncols - 1) + \
                xborder[0] + xborder[1]
        figheight = np.sum(height_ratios) + panel_bt[1] * (nrows - 1) + \
                yborder[0] + yborder[1]
        
        fig = plt.figure(figsize=(figwidth, figheight), dpi = dpi)

        if height_ratios is None:
            height_ratios = np.ones(nrows)*panel_length
        
        if width_ratios is None:
            width_ratios = np.ones(ncols)*panel_length
        # creating gridspec
        gs = gspec.GridSpec(nrows, ncols, left= xborder[0]/figwidth, right=1-xborder[1]/figwidth,
                top=1-yborder[1]/figheight, bottom=yborder[0]/figheight,
                wspace=panel_bt[0]*ncols/figwidth, hspace=panel_bt[1]*nrows/figheight,
                height_ratios = height_ratios, width_ratios = width_ratios)
        
        # making axes
        self.axes = np.empty((nrows, ncols), dtype = object)

        for i in range(nrows):
            for j in range(ncols):
                idx = (i, j)

                axis = fig.add_subplot(gs[idx])
                self.axes[idx] = axis

        self.fig = fig
        self.panel_length = panel_length
        self.panel_bt = panel_bt
        self.xborder = xborder
        self.yborder = yborder
        self.figsize = [figwidth, figheight]
        self.dpi = dpi
        self.dim = [nrows, ncols]
        return
    
    ##### INTERFACING WITH DATA CONTAINERS ##########################

    def setPlotArgs(self, plotArgs, attrs, slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelArgs(panel):
            panel.setArgs(attrs, plotArgs)
            return
        
        argnp = np.vectorize(_panelArgs)
        argnp(self.panels[slc])
        return
        
    ########## INTERFACING WITH PANELS ##############################

    def setTicks(self, tickParams, xory = 'both', which = 'both', 
            slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelTicks(axis):
            axis.tick_params(axis = xory, which = which, **tickParams)
            return
        
        ticknp = np.vectorize(_panelTicks, cache = True)
        ticknp(self.axes[slc])
        return

    def setAxisParams(self, axisParams, slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelAxis(axis):
            axis.set(**axisParams)
            return

        axisnp = np.vectorize(_panelAxis, cache = True)
        axisnp(self.axes[slc])
        return

    def drawLegend(self, legendParams, slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelLegend(axis):
            axis.legend(**legendParams)
            return

        legnp = np.vectorize(_panelLegend, cache = True)
        legnp(self.axes[slc])
        return

    def matchLimits(self, xory = 'both', slc = []):
        
        def _getlim(ax):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            return xlim[0], xlim[1], ylim[0], ylim[1]

        npget = np.vectorize(_getlim, cache=True)

        xmin, xmax, ymin, ymax = npget(self.axes[slc])

        xlim = [np.min(xmin), np.max(xmax)]
        ylim = [np.min(ymin), np.max(ymax)]

        def _setlim(ax):
            if xory == 'x' or xory == 'both':
                ax.set_xlim(xlim)
            if xory == 'y' or xory == 'both':
                ax.set_ylim(ylim)
            
            return
        
        npset = np.vectorize(_setlim, cache = True)

        npset(self.axes[slc])

        return
    
    def setRowLabels(self, rowlabels, pos, textKwargs = {},
            colidx = 0):
        
        for i in range(self.dim[0]):
            p = self.axes[i, colidx]
            p.text(pos[0], pos[1], rowlabels[i],
                    transform = p.transAxes, **textKwargs)
        return

    def setColLabels(self, collabels, pos, textKwargs = {},
            rowidx = 0):
        
        for i in range(self.dim[1]):
            p = self.axes[rowidx, i]
            p.text(pos[0], pos[1], collabels[i],
                    transform = p.transAxes, **textKwargs)
        return

    def getMask(self, rowVal = '', colVal = ''):
        def _get1DMask(val, dimValues, axis):
            if val == '':
                mask = np.ones(self.dim, dtype = bool)
            else:
                mask = np.zeros(self.dim, dtype = bool)

            try:
                idx = dimValues.index(val)
                if axis == 0:
                    mask[idx, :] = 1
                else:
                    mask[:, idx] = 1
                return mask
            except ValueError:
                return mask
                
        
        rowMask = _get1DMask(rowVal, self.rowValues, 0)
        colMask = _get1DMask(colVal, self.colValues, 1)
        return rowMask & colMask
    
    ##### INTERFACE WITH THE FIGURE #################################

    def annotateFig(self, text, pos, textKwargs = {}):
        self.fig.text(pos[0], pos[1], text, **textKwargs)
        return

    ############ PLOTTING ROUTINES ##################################
    
    def plotPanel(self, rowidx, colidx):
        idx = (rowidx, colidx)
        panel = self.panels[idx]
        panel.plot(self.axes[idx])
        return

    def plot(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.plotPanel(i, j)
        return

    def setFunc(self, attrs, func, slc = []):
        if not slc:
            slc = (slice(None), slice(None))
        
        def _panelFunc(panel):
            panel.setFunc(attrs, func)
            return
        
        funcnp = np.vectorize(_panelFunc, cache = True)
        funcnp(self.panels[slc])
        return
    
    def makeFills(self, attrs, fillKwargs = {}, slc = []):
        
        if not slc:
            slc = (slice(None), slice(None))
        
        def _panelFill(panel):
            panel.makeFill(attrs, fillKwargs)
            return

        fillnp = np.vectorize(_panelFill, cache = True)
        fillnp(self.panels[slc])
        return

    ##### CONVENIENCE METHODS #######################################

    def makeXLabel(self, text, pos = [], txtargs = {}):
        if not pos:
            pos = [0.5, 0]
        
        txtargs['ha'] = 'center'
        txtargs['va'] = 'bottom'

        self.annotateFig(text, pos, txtargs)
        return
    
    def makeYLabel(self, text, pos = [], txtargs = {}):
        if not pos:
            pos = [0, 0.5]
        
        txtargs['ha'] = 'right'
        txtargs['va'] = 'center'
        txtargs['rotation'] = 'vertical'

        self.annotateFig(text, pos, txtargs)
        return

    def setDefaultTicksParams(self):
        # slice of everything but bottom row
        topslc = (slice(0, -1), slice(None))

        # slice of everything but leftmost column
        rightslc = (slice(None), slice(1, None))
        
        if self.dim[0] > 1:
            params = {'labelbottom':False}
            self.setTicks(params, 'x', slc = topslc)
        
        if self.dim[1] > 1:
            params = {'labelleft':False}
            self.setTicks(params, 'y', slc = rightslc)
        return
    
    def matchDefaultLimits(self):
        nrows = self.dim[0]
        ncols = self.dim[1]

        # match y-axis limits in each row
        for i in range(nrows):
            slc = (i, slice(None))
            self.matchLimits('y', slc = slc)
        
        for j in range(ncols):
            slc = (slice(None), j)
            self.matchLimits('x', slc = slc)
        
        return


    ##### POST-PROCESS METHODS ######################################     

    def combineFigrids(self, figrid, loc = 'bottom'):
        nrows = self.dim[0]
        ncols = self.dim[1]

        if loc == 'bottom':
            nrows += figrid.dim[0]
            newslc = (slice(self.dim[0], None), slice(None))
            selfslc = (slice(0, self.dim[0]), slice(None))
        elif loc == 'top':
            nrows += figrid.dim[0]
            newslc = (slice(0, figrid.dim[0]), slice(None))
            selfslc = (slice(figrid.dim[0], None), slice(None))

        elif loc == 'right':
            ncols += figrid.dim[1]
            newslc = (slice(None), slice(self.dim[1], None))
            selfslc = (slice(None), slice(0, self.dim[1]))
        elif loc == 'left':
            ncols += figrid.dim[1]
            newslc = (slice(None), slice(0, figrid.dim[1]))
            selfslc = (slice(None), slice(figrid.dim[1], None))
        else:
            raise ValueError('not accepted location')
        
        height_ratios = np.ones(nrows) * self.panel_length
        width_ratios = np.ones(ncols) * self.panel_length
        if loc == 'bottom' or loc == 'top':

            height_ratios[newslc[0]] = figrid.panel_length
        elif loc == 'left' or loc == 'right':
            height_ratios[newslc[1]] = figrid.panel_length

        self._makeFig(nrows, ncols, self.panel_length, self.panel_bt,
            self.xborder, self.yborder, height_ratios, width_ratios,
            self.dpi)

        newpanels = np.empty((nrows, ncols), dtype = object)
        newpanels[selfslc] = self.panels.copy()
        newpanels[newslc] = figrid.panels.copy()
        self.panels = newpanels
        return

    def clf(self):
        plt.clf()
        plt.close()
        return
    
    def save(self, path):
        self.fig.savefig(path)
        return
