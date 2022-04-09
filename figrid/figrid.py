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
    
    def arrange(self, rowAttr = [], colAttr = [], panel_length = 3, 
            panel_bt = 0.11, xborder = 0.33, yborder = 0.33, 
            height_ratios = None, width_ratios = None, 
            dpi = 100):
        
        # panel_length gives the values in inches
        # the other values should scale with panel_length
        # so they are given as fractions of panel_length
        
        # if rowAttr colAttr are strings, convert them to lists
        rowAttr = list(rowAttr)
        colAttr = list(colAttr)

        panel_bt = panel_bt * panel_length
        xborder = xborder * panel_length
        yborder = yborder * panel_length
        
        rowValues = []
        colValues = []
        valToAttr = {}

        for ra in rowAttr:
            vals = self.dl.getAttrVals(ra)
            rowValues.extend(vals)
            for v in vals:
                valToAttr[v] = ra
        

        for ca in colAttr:
            vals = self.dl.getAttrVals(ca)
            colValues.extend(vals)
            for v in vals:
                valToAttr[v] = ca


        rowValues = self.rowOrderFunc(rowValues)
        colValues = self.colOrderFunc(colValues)

        print('The row values for %s: %s'%(rowAttr, str(rowValues)))
        print('The column values for %s: %s'%(colAttr, str(colValues)))
        
        nrows = max(1, len(rowValues))
        ncols = max(1, len(colValues))
        
        self.rowValues = rowValues
        self.colValues = colValues

        self._makeFig(nrows, ncols, panel_length, panel_bt,
            xborder, yborder, height_ratios, width_ratios, dpi)
        
        self.panels = np.empty((nrows, ncols), dtype = object)
        for i in range(nrows):
            for j in range(ncols):
                attr_for_row = valToAttr[rowValues[i]]
                attr_for_col = valToAttr[colValues[j]]
                rowColAttr = {}
                rowColAttr[attr_for_row] = rowValues[i]
                rowColAttr[attr_for_col] = colValues[j]

                dlPanel = self.dl.getMatching(rowColAttr)
                
                self.panels[i, j] = DataList(copy.deepcopy(dlPanel))
        
        return


    def _makeFig(self, nrows, ncols, panel_length, wspace,
            hspace, xborder, yborder, height_ratios = None, 
            width_ratios = None, dpi = 100):
        """
        Make a gridspec and corresponding figure according to the
        given specifications.

        Args:
            nrows (int): number of rows
            ncols (int): number of columns
            panel_length (float): size of panel in inches.
                Used to scale the other inputs.
            wspace (float, np.array): the width of the padding
                between panels, as a fraction of panel_length.
                If a float, applies that value to all padding.
                An array must have dimensions (ncols - 1).
            hspace (float, np.array): the height of the padding
                between panels, as a fraction of panel_length.
                If a float, applies that value to all padding.
                An array must have dimensions (nrows - 1).
            xborder (float, np.array): the size of the padding
                on the left and right borders of the figure.
                As a fraction of panel_length. If given a float,
                applies that value to both borders.
            yborder (float, np.array): the size of the padding
                on the top and bottom borders of the figure.
                As a fraction of panel_length. If given a float,
                applies that value to both borders.
            height_ratios (list, np.array, optional): the 
                respective heights of each the panels in the 
                corresponding rows. Must have dimensions (nrows). 
                As a fraction of panel_length. 
            width_ratios (list, np.array, optional): the respective heights
                of each the panels in the corresponding rows. Must
                have dimensions (nrows). As a fraction of 
                panel_length. 
            dpi (float, optional): dots per inch, the resolution of the
                figure.
        """

        # default behavior for borders
        if isinstance(xborder, float) or isinstance(xborder, int):
            xborder = np.array([xborder, xborder])
        if isinstance(yborder, float) or isinstance(yborder, int):
            yborder = np.array([yborder, yborder])

        # default behavior for padding
        paddim = [max(1, ncols - 1), max(1, nrows - 1)]
        if isinstance(wspace, float) or isinstance(wspace, int):
            wspace = np.ones(paddim[0]) * wspace
        if isinstance(hspace, float) or isinstance(hspace, int):
            hspace = np.ones(paddim[1]) * hspace

        # default behavior for ratios
        if height_ratios is None:
            height_ratios = np.ones(nrows)
        else:
            # renormalize
            maxval = np.max(height_ratios)
            height_ratios /= maxval

        if width_ratios is None:
            width_ratios = np.ones(ncols)
        else:
            #renormalize
            maxval = np.max(width_ratios)
            width_ratios /= maxval

        #TODO handle input errors with panelbt, height/width ratios

        # creating Figure object

        # convert everything into units of inches
        width_ratios *= panel_length; height_ratios *= panel_length
        xborder *= panel_length; yborder *= panel_length
        wspace *= panel_length; hspace *= panel_length
        
        total_widths = np.sum(width_ratios)
        total_wspace = np.sum(wspace)
        wborder_space = np.sum(xborder)

        total_heights = np.sum(height_ratios)
        total_hspace = np.sum(hspace)
        hborder_space = np.sum(yborder)
        
        # get figwidth and figheight in inches
        figwidth = total_widths + total_wspace + wborder_space
        figheight = total_heights + total_hspace + hborder_space 

        fig = plt.figure(figsize=(figwidth, figheight), dpi = dpi)
                
        axes = np.empty((nrows, ncols), dtype = object)
        for i in range(nrows):
            for j in range(ncols):
                # a label makes each axis unique - otherwise mpl will
                # return a previously made axis
                
                ax = fig.add_subplot(label = str((i, j)))
                
                height = height_ratios[i]
                width = width_ratios[j]
                
                total_hspace = np.sum(hspace[:i])
                total_heights = np.sum(height_ratios[:i+1])
                total_widths = np.sum(width_ratios[:j])
                total_wspace = np.sum(wspace[:j])
                
                bot = figheight - yborder[0] - total_hspace - total_heights
                left = xborder[0] + total_widths + total_wspace
                

                axdim = [left / figwidth, bot / figheight, 
                        width / figwidth, height / figheight]
                ax.set_position(axdim)
                axes[i, j] = ax
                
        self.fig = fig
        self.axes = axes
        self.xborder = xborder / figwidth
        self.yborder = yborder / figheight
        self.wspace = wspace / figwidth
        self.hspace = hspace / figheight
        self.figsize = [figwidth, figheight]
        self.panel_length = panel_length
        return fig
    
    ##### INTERFACING WITH DATA CONTAINERS ##########################

    def setPlotArgs(self, attrs, plotArgs, slc = []):
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

    # def adjustPanelSpacing(self, xory, inbt, new_panel_bt):
    #     xborder = self.xborder; yborder = self.yborder
    #     figwidth = self.figsize[0]; figheight = self.figsize[1]

    #     if xory == 'x' or xory == 'both':
    #         gs = gspec.GridSpec(inbt[0], self.dim[1], left= xborder[0]/figwidth, right=0,
    #                 top=1-yborder[1]/figheight, bottom=yborder[0]/figheight,
    #                 wspace=panel_bt[0]*ncols/figwidth, hspace=panel_bt[1]*nrows/figheight,
    #                 height_ratios = height_ratios, width_ratios = width_ratios)
    #         gsright = gspec.GridSpec(self.dim[0] - inbt[0], self.dim[1], right = )
            
    #     return
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
        
        txtargs['ha'] = 'left'
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
        heights = np.zeros(nrows)
        widths = np.zeros(ncols)
        heights[selfslc[0]] = self.panel_heights
        widths[selfslc[1]] = self.panel_widths
        if loc == 'bottom' or loc == 'top':

            heights[newslc[0]] = figrid.panel_heights
        elif loc == 'left' or loc == 'right':
            widths[newslc[1]] = figrid.panel_widths
        print(heights)
        print(widths)
        pl = max(np.max(heights), np.max(widths))
        self._makeFig(nrows, ncols, pl, self.panel_bt,
            self.xborder, self.yborder, heights, widths,
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
