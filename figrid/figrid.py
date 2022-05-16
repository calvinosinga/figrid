#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from figrid.data_container import DataContainer


class Figrid():
   
    def __init__(self, panels, panel_attr, row_values, col_values):
        self.row_values = row_values
        self.col_values = col_values
        self.panel_attr = panel_attr
        self.panels = panels
        self.tick_params = {}
        self.axis_params = {}
        self.legend_params = {}
        self.attr_args = {}
        self.fig_params = {}
        self.display_names = {}
        self.axis_labels = {}
        self.row_label_args = ()
        self.col_label_args = ()
        self.row_labels = []
        self.col_labels = []
        self.legend_slc = None
        return

    ########## MAKING FIGURES #####################################

    def makeFig(self, nrows, ncols, panel_length = 3, 
            wspace = 0.25, hspace = 0.25, 
            xborder = [0.33, 0], yborder = [0, 0.33], 
            height_ratios = None, 
            width_ratios = None, 
            figkw = {}):
        """
        Make a figure according to the
        given specifications for the subpanels.

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
        if isinstance(xborder, list):
            xborder = np.array(xborder)
        if isinstance(yborder, list):
            yborder = np.array(yborder)

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

        figkw['figsize'] = (figwidth, figheight)
        fig = plt.figure(**figkw)
                
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
        self.xborder = xborder
        self.yborder = yborder
        self.wspace = wspace 
        self.hspace = hspace
        self.figsize = [figwidth, figheight]
        self.panel_length = panel_length
        self.panel_heights = height_ratios
        self.panel_widths = width_ratios
        self.dim = [nrows, ncols]
        return fig
    
    ##### INTERFACING WITH DATA CONTAINERS ##########################

    def setPlotArgs(self, attrs, plotArgs, slc = None):
        if slc is None:
            slc = (slice(None), slice(None))

        def _panelArgs(panel):
            panel.setArgs(attrs, plotArgs)
            return
        
        argnp = np.vectorize(_panelArgs)
        argnp(self.panels[slc])
        return
        
    ########## INTERFACING WITH PANELS ##############################

    def tickArgs(self, tickParams, xory = 'both', which = 'both', 
            slc = None):
        if slc is None:
            slc = (slice(None), slice(None))

        def _panelTicks(axis):
            axis.tick_params(axis = xory, which = which, **tickParams)
            return
        
        ticknp = np.vectorize(_panelTicks, cache = True)
        ticknp(self.axes[slc])
        return

    def axisArgs(self, axisParams, slc = None):
        if slc is None:
            slc = (slice(None), slice(None))

        def _panelAxis(axis):
            axis.set(**axisParams)
            return

        axisnp = np.vectorize(_panelAxis, cache = True)
        axisnp(self.axes[slc])
        return

    def legendArgs(self, legendParams, slc = None):
        self.legend_params.update(legendParams)
        self.legend_slc = slc
        return

    def _makeLegend(self):
        legendParams = self.legend_params
        slc = self.legend_slc
        if slc is None:
            slc = (slice(None), slice(None))

        def _panelLegend(axis):
            axis.legend(**legendParams)
            return

        legnp = np.vectorize(_panelLegend, cache = True)
        legnp(self.axes[slc])
        return
    
    def matchLimits(self, xory = 'both', slc = None):
        
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
    
    def _makeRowLabels(self):
        rowlabels = self.row_labels
        
        if rowlabels:
            pos, textKwargs, colidx = self.row_label_args
            for i in range(self.dim[0]):
                p = self.axes[i, colidx]
                p.text(pos[0], pos[1], rowlabels[i],
                        transform = p.transAxes, **textKwargs)
        return

    def rowLabelArgs(self, rowlabels, pos = [], textKwargs = {},
            colidx = 0):
        if not pos:
            pos = [0.05, 0.05]

        self.row_labels = rowlabels
        self.row_label_args = (pos, textKwargs, colidx)
        return
    
    def _makeColLabels(self):
        
        collabels = self.col_labels
        
        if collabels:
            pos, textKwargs, rowidx = self.col_label_args
            for i in range(self.dim[1]):
                p = self.axes[rowidx, i]
                p.text(pos[0], pos[1], collabels[i],
                        transform = p.transAxes, **textKwargs)
        return
    
    def colLabelArgs(self, collabels, pos = [], textKwargs = {},
            rowidx = 0):
        if not pos:
            pos = [0.5, 0.9]

        self.col_labels = collabels
        self.col_label_args = (pos, textKwargs, rowidx)
        return
    
    ##### INTERFACE WITH THE FIGURE #################################

    def annotateFig(self, text, pos, textKwargs = {}):
        self.fig.text(pos[0], pos[1], text, **textKwargs)
        return

    ############ PLOTTING ROUTINES ##################################
    
    def plotPanel(self, rowidx, colidx):
        idx = (rowidx, colidx)
        panel = self.panels[idx]
        for dc in panel:
            dc.plot(self.axes[idx])
        return

    def plot(self):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.plotPanel(i, j)
        
        self._makeLegend()
        self._makeColLabels()
        self._makeRowLabels()
        return

    def setFunc(self, attrs, func, slc = None):
        if slc is None:
            slc = (slice(None), slice(None))
        
        def _panelFunc(panel):
            panel.setFunc(attrs, func)
            return
        
        funcnp = np.vectorize(_panelFunc, cache = True)
        funcnp(self.panels[slc])
        return
    
    def makeFills(self, attrs, fillKwargs = {}, slc = None):
        
        if slc is None:
            slc = (slice(None), slice(None))
        
        def _panelFill(panel):
            data = None
            for dc in panel:
                if dc.isMatch(attrs):
                    if data is None:

                        data = dc.getData()
                        x = data[0]; y = data[1]
                        ymins = np.ones_like(y) * y
                        ymaxs = np.ones_like(y) * y
                    else:
                        data = dc.getData()
                        x = data[0]; y = data[1]
                        ymins = np.minimum(y, ymins)
                        ymaxs = np.maximum(y, ymaxs)
                    
                    args = {'visible':False, 'zorder':-1,
                        'label':'_nolegend_'}
                    dc.setArgs(args)
            filldc = DataContainer([x, ymins, ymaxs])
            attrs['figrid_process'] = 'fill'

            def _plotFill(ax, data, kwargs):
                ax.fill_between(data[0], data[1], data[2], **kwargs)
                return
            
            filldc.setFunc(_plotFill)
            filldc.setArgs(fillKwargs)
            self.append(filldc)
            return

        fillnp = np.vectorize(_panelFill, cache = True)
        fillnp(self.panels[slc])
        return

    ##### CONVENIENCE METHODS #######################################

    def setXLabel(self, text, pos = [], txtargs = {}):
        if not pos:
            fl = self.figsize[0]
            xb = self.xborder
            pos = [(0.5 * (fl - np.sum(xb)) + xb[0]) / fl, 0]
        
        txtargs['ha'] = 'center'
        txtargs['va'] = 'bottom'

        self.annotateFig(text, pos, txtargs)
        return
    
    def setYLabel(self, text, pos = [], txtargs = {}):
        if not pos:
            fh = self.figsize[1]
            yb = self.yborder
            pos = [0, (0.5 * (fh - np.sum(yb)) + yb[1])/fh]
        
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
            self.tickArgs(params, 'x', slc = topslc)
        
        if self.dim[1] > 1:
            params = {'labelleft':False}
            self.tickArgs(params, 'y', slc = rightslc)
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

    def clf(self):
        plt.clf()
        plt.close()
        return
    
    def save(self, path):
        self.fig.savefig(path)
        return
    
