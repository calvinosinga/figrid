#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from figrid.data_container import DataContainer
import copy

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
        self.axis_label_args = {'x':{}, 'y':{}, 'both':{}}
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

        if isinstance(wspace, list):
            wspace = np.array(wspace)
        if isinstance(hspace, list):
            hspace = np.array(hspace)
        # default behavior for padding
        paddim = [max(1, ncols - 1), max(1, nrows - 1)]
        if isinstance(wspace, float) or isinstance(wspace, int):
            if ncols == 1:
                wspace = np.zeros(0)
            else:
                wspace = np.ones(paddim[0]) * wspace
        if isinstance(hspace, float) or isinstance(hspace, int):
            if nrows == 1:
                hspace = np.zeros(0)
            else:
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

    def plotArgs(self, attrs, plotArgs, slc = None):
        slc = self._getSlice(slc)

        # so you can specify with just string
        is_num = isinstance(attrs, int) or isinstance(attrs, float)
        if isinstance(attrs, str) or is_num:
            attrs = {self.panel_attr:attrs}

        # TODO if fill is made don't update args for constituent lines
        def _panelArgs(panel):
            for dc in panel:
                if dc.isMatch(attrs):
                    dc.setArgs(plotArgs)
            return
        
        argnp = np.vectorize(_panelArgs)
        argnp(self.panels[slc])
        return
        
    ########## INTERFACING WITH PANELS ##############################

    def _getSlice(self, slc):
        if slc is None:
            return (slice(None), slice(None))
        
        elif isinstance(slc, list):
            if len(slc) == 0:
                return (slice(None), slice(None))
            
            elif isinstance(slc[0], str):
                mask = np.zeros(self.dim, dtype = bool)
                for s in slc:
                    if s in self.row_values:
                        for rl in range(len(self.row_values)):
                            if s == self.row_values[rl]:
                                mask[rl, :] = True
                    if s in self.col_values:
                        for cl in range(len(self.col_values)):
                            if s == self.col_values[cl]:
                                mask[:, cl] = True
                return mask
            
            elif isinstance(slc[0], tuple):
                mask = np.zeros(self.dim, dtype = bool)
                for tup in slc:
                    mask[tup] = True
                return mask
        
        elif isinstance(slc, tuple):
            return slc
        
        return slc # assume user knows what they're doing
                
                
    def tickArgs(self, tickParams, xory = 'both', which = 'both', 
            slc = None):

        slc = self._getSlice(slc)

        def _panelTicks(axis):
            axis.tick_params(axis = xory, which = which, **tickParams)
            return
        
        ticknp = np.vectorize(_panelTicks, cache = True, otypes = [object])
        ticknp(self.axes[slc])
        return

    def spineArgs(self, spine_args, which = 'all', slc = None):
        slc = self._getSlice(slc)

        def _setSpine(axis):
            if which == 'all':
                spines = ['bottom', 'top', 'right', 'left']
                for s in spines:
                    axis.spines[s].set(**spine_args)
            else:
                axis.spines[which].set(**spine_args)

        spinenp = np.vectorize(_setSpine, cache=True, otypes=[object])
        spinenp(self.axes[slc])
        return

    def axisArgs(self, axisParams, slc = None):
        slc = self._getSlice(slc)

        def _panelAxis(axis):
            axis.set(**axisParams)
            return

        axisnp = np.vectorize(_panelAxis, cache = True, otypes = [object])
        axisnp(self.axes[slc])
        return

    def figArgs(self, figArgs):
        self.fig.set(**figArgs)
        return
    
    def legendArgs(self, legendParams, slc = None):
        self.legend_params.update(legendParams)
        self.legend_slc = slc
        return

    def _makeLegend(self):
        legendParams = self.legend_params
        slc = self.legend_slc
        slc = self._getSlice(slc)

        def _panelLegend(axis):
            axis.legend(**legendParams)
            return

        legnp = np.vectorize(_panelLegend, cache = True, otypes = [object])
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
        
        npset = np.vectorize(_setlim, cache = True, otypes = [object])

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

    def rowLabelArgs(self, rowlabels = [], pos = [], textKwargs = {},
            colidx = 0):
        if rowlabels:
            self.row_labels = rowlabels
        
        if self.row_label_args:
            if not pos:
                pos = self.row_label_args[0]
            
            temp = self.row_label_args[1]
            temp.update(textKwargs)
            textKwargs = temp
            if colidx is None:
                colidx = self.row_label_args[2]
        
        if not pos:
            pos = [0.5, 0.9]
        
        if colidx is None:
            colidx = 0
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
    
    def colLabelArgs(self, collabels = [], pos = [], textKwargs = {},
            rowidx = None):
        
        if collabels:
            self.col_labels = collabels
        
        if self.col_label_args:
            if not pos:
                pos = self.col_label_args[0]
            
            temp = self.col_label_args[1]
            temp.update(textKwargs)
            textKwargs = temp
            
            if rowidx is None:
                rowidx = self.col_label_args[2]
        
        if not pos:
            pos = [0.5, 0.9]
        
        if rowidx is None:
            rowidx = 0
        self.col_label_args = (pos, textKwargs, rowidx)
        return
    
    def annotateAxis(self, text, pos, idx, text_kwargs = {}):
        ax = self.axes[idx]
        ax.text(pos[0], pos[1], text, transform = ax.transAxes, **text_kwargs)
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
        slc = self._getSlice(slc)
        
        def _panelFunc(panel):
            for dc in panel:
                if dc.isMatch(attrs):
                    dc.setFunc(func)
            return
        
        funcnp = np.vectorize(_panelFunc, cache = True, otypes = [object])
        funcnp(self.panels[slc])
        return
    
    def norm(self, num_attrs, denom_attrs, ones_args = {}, idx = 1):
        #TODO check to make sure that this doesn't change original
        # stored data values
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                panel = self.panels[i, j]
                norm = None
                num_list = []
                # find the denominator to norm by
                for dc in panel:
                    if dc.isMatch(denom_attrs):
                        ddata = dc.getData()
                        norm = copy.deepcopy(ddata[idx])
                        ddata[idx] = np.ones_like(ddata[idx])
                        dc.setData(ddata)
                        dc.setArgs(ones_args)
                    elif dc.isMatch(num_attrs):
                        num_list.append(dc)
                if norm is not None:
                    for num in num_list:
                        ndata = num.getData()
                        ndata[idx] /= norm
                        num.setData(ndata)
        
        return

                        
    def fill(self, attrs, fillKwargs = {}, slc = None):
        
        slc = self._getSlice(slc)

        def _panelFill(panel):
            data = None
            x = 0; ymins = 0; ymaxs = 0
            labels = []; colors = []
            match_found = False
            for dc in panel:
                if dc.isMatch(attrs):
                    match_found = True
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
                    # TODO need to specify which ones can translate
                    # to fill_between
                    dcargs = dc.getArgs()
                    if 'label' in dcargs:
                        labels.append(dcargs['label'])
                    if 'color' in dcargs:
                        colors.append(dcargs['color'])
                    args = {'visible':False, 'zorder':-1,
                        'label':'_nolegend_'}
                    dc.setArgs(args)

            if match_found:
                if len(labels) > 0:
                    if labels.count(labels[0]) == len(labels):
                        fillKwargs['label'] = labels[0]
                if len(colors) > 0:
                    if colors.count(colors[0]) == len(colors):
                        fillKwargs['color'] = colors[0]
                filldc = DataContainer([x, ymins, ymaxs])
                filldc.update(attrs)
                filldc.add('figrid_process', 'fill')
                def _plotFill(ax, data, kwargs):
                    ax.fill_between(data[0], data[1], data[2], **kwargs)
                    return
                
                filldc.setFunc(_plotFill)
                filldc.setArgs(fillKwargs)
                panel.append(filldc)
            return

        fillnp = np.vectorize(_panelFill, cache = True, otypes=[object])
        fillnp(self.panels[slc])
        return

    ##### CONVENIENCE METHODS #######################################
    def axisLabelArgs(self, txtargs):
        self.axis_label_args.update(txtargs)
        return
    
    def setXLabel(self, text, pos = [], txtargs = {}):
        if not pos:
            fl = self.figsize[0]
            xb = self.xborder
            pos = [(0.5 * (fl - np.sum(xb)) + xb[0]) / fl, 0]
        default_args = {'ha':'center', 'va':'bottom'}
        default_args.update(self.axis_label_args['both'])
        default_args.update(self.axis_label_args['x'])
        default_args.update(txtargs)
        self.annotateFig(text, pos, default_args)
        return
    
    def setYLabel(self, text, pos = [], txtargs = {}):
        if not pos:
            fh = self.figsize[1]
            yb = self.yborder
            pos = [0, (0.5 * (fh - np.sum(yb)) + yb[1])/fh]
        
        default_args = {'ha':'left', 'va':'center', 
                'rotation':'vertical'}
        default_args.update(self.axis_label_args['both'])
        default_args.update(self.axis_label_args['y'])
        default_args.update(txtargs)
        self.annotateFig(text, pos, default_args)
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

    def autoFill(self, fill_kwargs = {}):
        pa = self.panel_attr
        panelvals = []
        counts = {}
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                panel = self.panels[i, j]
                for dc in panel:
                    val = dc.get(pa)
                    if val not in panelvals:
                        panelvals.append(val)
                        counts[val] = np.zeros(self.dim)
                        counts[val][i, j] = 1
                    
                    else:
                        counts[val][i, j] += 1
        

        for pv in panelvals:
            mask = counts[pv] > 1
            self.fill({pa:pv}, fill_kwargs, mask)
        return
    
    def autoNorm(self, denominator_attr, must_match = [], idx = 1):
        denom_arr = np.empty(self.dim, dtype = object)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                panel = self.panels[i, j]
                denom_arr[i, j] = []
                for dc in panel:
                    if denominator_attr == dc.get(self.panel_attr):
                        denom_arr[i, j].append(dc)


        # print(denom_arr[0, 0])

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                # print(i, j)
                panel = self.panels[i, j]
                denoms = denom_arr[i, j]
                for d in denoms:
                    norm = copy.copy(d.getData()[idx])
                    match_attrs = {}
                    for mm in must_match:
                        match_attrs[mm] = d.get(mm)
                    # print(match_attrs)
                    for dc in panel:
                        # for mm in must_match:
                        #     print(dc.get(mm))
                        if dc.isMatch(match_attrs):
                            # print('found %s'%dc.get('HI_res'))
                            dcdata = dc.getData()
                            # print(dcdata[idx][0])
                            dcdata[idx][:] = dcdata[idx][:] / norm[:]
                            # print(dcdata[idx][0])

                            # print(dcdata)
                            dc.setData(dcdata)
    
        return

    def plotOnes(self, args = {'color':'gray', 'linestyle':'--'}):
        # add ability to make slices
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                ax = self.axes[i, j]
                ax.plot(ax.get_xlim(), [1,1], **args)
        
        return

    def clf(self):
        plt.clf()
        plt.close()
        return
    
    def save(self, path, 
            save_kw = {'bbox_inches':'tight', 'facecolor':'auto'}):
        self.fig.savefig(path, **save_kw)
        return
    
