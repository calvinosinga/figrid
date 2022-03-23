#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from figrid.panel import Panel
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
            panel_bt = 0.1, xborder = 1, yborder = 1, 
            height_ratios = None, width_ratios = None, 
            dpi = 100):
        
        rowValues = self.dl.getAttrVals(rowAttr)
        colValues = self.dl.getAttrVals(colAttr)
        
        rowValues = self.rowOrderFunc(rowValues)
        colValues = self.colOrderFunc(colValues)

        print('The row values for %s: %s'%(rowAttr, str(rowValues)))
        print('The column values for %s: %s'%(colAttr, str(colValues)))

        nrows = len(rowValues)
        ncols = len(colValues)
        
        self.dim = [nrows, ncols]
        self.rowValues = rowValues
        self.colValues = colValues

        nrows = self.dim[0]
        ncols = self.dim[1]

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
        
        # making panels list
        self.panels = np.empty((nrows, ncols), dtype = object)

        for i in range(nrows):
            for j in range(ncols):
                idx = (i, j)
                panelAttr = {}
                panelAttr[rowAttr] = rowValues[i]
                panelAttr[colAttr] = colValues[j]

                dlPanel = self.dl.getMatching(panelAttr,
                        return_as = 'DataList')
                axis = fig.add_subplot(gs[idx])
                self.panels[idx] = Panel(dlPanel, axis)

        self.fig = fig
        self.panel_length = panel_length
        self.panel_bt = panel_bt
        self.xborder = xborder
        self.yborder = yborder
        self.figsize = [figwidth, figheight]
        return

    ########## INTERFACING WITH PANELS ##############################
    
    def makeFills(self, attrs, fillKwargs = {}, slc = []):
        
        if not slc:
            slc = (slice(None), slice(None))
        
        def _panelFill(panel):
            
            panel.makeFill(attrs, fillKwargs)
            return panel

        fillnp = np.vectorize(_panelFill)
        self.panels[slc] = fillnp(self.panels[slc])
        print(self.panels)
        return
    
    def tickParams(self, tickParams, xory = 'both', which = 'both', 
            slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelTicks(panel):
            panel.setTicks(tickParams, xory, which)
            return panel
        
        ticknp = np.vectorize(_panelTicks)
        self.panels[slc] = ticknp(self.panels[slc])
        return

    def axisParams(self, axisParams, slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelAxis(panel):
            panel.setAxis(axisParams)
            return panel

        axisnp = np.vectorize(_panelAxis)
        self.panels[slc] = axisnp(self.panels[slc])
        return

    def drawLegend(self, legendParams, slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelLegend(panel):
            panel.drawLegend(legendParams)
            return panel

        legnp = np.vectorize(_panelLegend)
        self.panels[slc] = legnp(self.panels[slc])
        return

    def plotArgs(self, plotArgs, attrs, slc = []):
        if not slc:
            slc = (slice(None), slice(None))

        def _panelArgs(panel):
            panel.setArgs(attrs, plotArgs)
            return panel
        
        argnp = np.vectorize(_panelArgs)
        self.panels[slc] = argnp(self.panels[slc])
        return

    def shareAxis(self, idx, xory, slc = []):
        
        if not slc:
            slc = (slice(None), slice(None))
        
        axisForShare = self.panels[idx].axis

        def _sharex(panel):
            panel.axis.sharex(axisForShare)
            return panel
        def _sharey(panel):
            panel.axis.sharey(axisForShare)
            return panel
        xnp = np.vectorize(_sharex)
        ynp = np.vectorize(_sharey)
        if xory == 'x' or xory == 'both':
            self.panels[slc] = xnp(self.panels[slc])
        if xory == 'y' or xory == 'both':
            self.panels[slc] = ynp(self.panels[slc])
        return

    ############ INTERFACING WITH FIGURE ############################

    def setRowLabels(self, rowlabels, pos, textKwargs = {},
            colidx = 0):
                
        if not slc:
            slc = (slice(None), slice(None))
        
        for i in range(self.dim[0]):
            p = self.panels[i, colidx]
            p.axis.text(pos[0], pos[1], rowlabels[i],
                    transform = p.axis.transAxes, **textKwargs)
        return

    def setColLabels(self, collabels, pos, textKwargs = {},
            rowidx = 0):

        if not slc:
            slc = (slice(None), slice(None))
        
        for i in range(self.dim[1]):
            p = self.panels[rowidx, i]
            p.axis.text(pos[0], pos[1], collabels[i],
                    transform = p.axis.transAxes, **textKwargs)
        return

    def text(self, pos, text, textKwargs = {}):
        self.fig.text(pos[0], pos[1], text, **textKwargs)
        return

    ############ PLOTTING ROUTINES ##################################
    
    
    def plotPanel(self, rowidx, colidx):
        idx = (rowidx, colidx)
        panel = self.panels[idx]
        panel.plot()
        return

    def plot(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.plotPanel(i, j)
        return






    # # def setFills(self, fillAttrs, slc = []):

    # #     if not slc:
    # #         slc = (slice(None), slice(None))

    # #     def _setFill(panel):
    # #         panel.makeFill(
    
    # def setPlotArgs(self, panelVal, kwargs, slc = []):
        
    #     if not slc:
    #         slc = (slice(None), slice(None))
        
    #     def _setArg(panel):
    #         panel.plotArgs[panelVal].update(kwargs)
    #         return panel
        
    #     setArgNumpy = np.vectorize(_setArg)

    #     panelsSubset = copy.deepcopy(self.panels[slc])
    #     self.panels[slc] = setArgNumpy(panelsSubset)
    #     return
    
    # def setTickArgs(self, axis, which, kwargs, slc = []):
        
    #     if not slc:
    #         slc = (slice(None), slice(None))
        
    #     def _setArg(panel):
    #         if axis == 'x' or axis == 'both':
    #             panel.xtickArgs[which].update(kwargs)
    #         if axis == 'y' or axis == 'both':
    #             panel.ytickArgs[which].update(kwargs)
    #         return panel
        
    #     setArgNumpy = np.vectorize(_setArg)

    #     panelsSubset = copy.deepcopy(self.panels[slc])
    #     self.panels[slc] = setArgNumpy(panelsSubset)
    #     return
    
    # def setLegendArgs(self, kwargs, slc = []):
    #     if not slc:
    #         slc = (slice(None), slice(None))
        
    #     def _setArg(panel):
    #         panel.legendArgs.update(kwargs)
    #         return panel
        
    #     setArgNumpy = np.vectorize(_setArg)

    #     panelsSubset = copy.deepcopy(self.panels[slc])
    #     self.panels[slc] = setArgNumpy(panelsSubset)
    #     return
    
    # def fillBetween(self, attrs, slc = []):
        
    #     if not slc:
    #         slc = (slice(None), slice(None))
        
    #     def _fillAttr(panel):
    #         panel.fillMatch(attrs)
    #         return panel
        
    #     fillNumpy = np.vectorize(_fillAttr)

    #     panelsSubset = copy.deepcopy(self.panels[slc])
    #     self.panels[slc] = fillNumpy(panelsSubset)

    #     return  
