#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from figrid.data_container import DataContainer
import copy
import seaborn as sns

class Figrid():
   
    def __init__(self, panels, panel_attr, row_values, col_values):
        self.row_values = row_values
        self.col_values = col_values
        self.panel_attr = panel_attr
        self.panels = panels
        self.panelsize = [3, 3]
        self.dim = panels.shape
        self.rm_legend = False

        empty_types = {'both':{}, 'major':{}, 'minor':{}}
        empty_ticks = {}; axes_list = ['x', 'y', 'both']
        for a in axes_list:
            empty_ticks[a] = copy.deepcopy(empty_types)
        self.tick_args = self._emptyDicts(panels.shape, empty_ticks)

        self.axis_args = self._emptyDicts(panels.shape, {})

        self.leg_args = {}
        self.legend_slc = None

        self.gspec_args = {}
        gspec_args = {
            'height_ratios' : np.ones(self.dim[0]),
            'width_ratios' : np.ones(self.dim[1]),
            'hspace' : 0.25,
            'wspace' : 0.25,
            'xborder' : np.array([0.33, 0]),
            'yborder' : np.array([0, 0.33])
        }
        self.gspecArgs(gspec_args)
        
        self.fig_args = {}
        self.calculateFigsize()

        empty_spines = {'bottom':{}, 'left':{}, 'right':{}, 'top':{}}
        self.spine_args = self._emptyDicts(panels.shape, empty_spines)

        self.axis_label_args = {'x':{}, 'y':{}, 'both':{}}
        self.axis_label_pos = {'x':[], 'y':[]}
        self.axis_label_text = {'x':'', 'y':''}
        self.axis_label_panels = {'x':[slice(0, self.dim[1])], 'y':[slice(0, self.dim[0])]}

        self.row_label_args = {'ha':'left', 'va':'bottom'}
        self.col_label_args = {'ha':'center', 'va':'top'}
        self.row_label_pos = [0.05, 0.05]
        self.col_label_pos = [0.5, 0.9]
        self.row_label_text = []
        self.col_label_text = []
        self.row_label_col = 0
        self.col_label_row = 0

        self.plot_order = []
        self.fig_annotations = []
        self.panel_annotations = []
        self.other_plots = []
        self.fig = None
        self.is_plotted = False
        return

    def _emptyDicts(self, shape, dict_element):
        empty = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                empty[i, j] = copy.deepcopy(dict_element)
        return empty

    ########## MAKING FIGURES #####################################
    def _makeAxes(self):
        fig = self.fig
        
        ga = self.gspec_args
        nrows = self.dim[0]; ncols = self.dim[1]
        hrs = ga['height_ratios']
        wrs = ga['width_ratios']
        yb = ga['yborder']; xb = ga['xborder']
        hs = ga['hspace']; ws = ga['wspace']
        figsize = self.calculateFigsize()
        figwidth = figsize[0]; figheight = figsize[1]

        panel_width = self.panelsize[0]
        panel_height = self.panelsize[1]
        width_ratios = wrs * panel_width; height_ratios = hrs * panel_height
        xborder = xb * panel_width; yborder = yb * panel_height
        wspace = ws * panel_width; hspace = hs * panel_height
        
        
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
        
        self.axes = axes
        return
    
    def gspecArgs(self, gspec_kwargs = {}, **other_kwargs):
            
        gspec_kwargs.update(other_kwargs)
        self.gspec_args.update(gspec_kwargs)
        
        ga = self.gspec_args
        nrows, ncols = self.dim
        height_ratios = ga['height_ratios']
        width_ratios = ga['width_ratios']
        yborder = ga['yborder']; xborder = ga['xborder']
        hspace = ga['hspace']; wspace = ga['wspace']
        
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
        
        self.gspec_args.update({
            'wspace':wspace,
            'hspace':hspace,
            'xborder':xborder,
            'yborder':yborder,
            'height_ratios':height_ratios,
            'width_ratios':width_ratios
        })
        return
    
    def setPanelsize(self, panel_width, panel_height):
        self.panelsize = [panel_width, panel_height]
        return
    
    def calculateFigsize(self):
        ga = self.gspec_args
        hrs = ga['height_ratios']
        wrs = ga['width_ratios']
        yb = ga['yborder']; xb = ga['xborder']
        hs = ga['hspace']; ws = ga['wspace']

        panel_width = self.panelsize[0]
        panel_height = self.panelsize[1]
        width_ratios = wrs * panel_width; height_ratios = hrs * panel_height
        xborder = xb * panel_width; yborder = yb * panel_height
        wspace = ws * panel_width; hspace = hs * panel_height
        
        total_widths = np.sum(width_ratios)
        total_wspace = np.sum(wspace)
        wborder_space = np.sum(xborder)

        total_heights = np.sum(height_ratios)
        total_hspace = np.sum(hspace)
        hborder_space = np.sum(yborder)
        
        # get figwidth and figheight in inches
        figwidth = total_widths + total_wspace + wborder_space
        figheight = total_heights + total_hspace + hborder_space

        return [figwidth, figheight]
        

    def figArgs(self, fig_kwargs = {}, **other_kwargs):
        fig_kwargs.update(other_kwargs)
        self.fig_args.update(fig_kwargs)
        return
    
    def setFig(self, fig):
        self.fig = fig
        return

    def _applyFigArgs(self):
        
        self.fig.set(**self.fig_args)
        return
    
    def _makeFig(self):
        figsize = self.calculateFigsize()
        self.fig = plt.figure(figsize = figsize)
        return
    
    ##### INTERFACING WITH DATA CONTAINERS ##########################

    def plotArgs(self, attrs, plot_kwargs = {}, slc = None, 
            **other_kwargs):
        plot_kwargs = copy.deepcopy(plot_kwargs)
        slc = self._getSlice(slc)
        plot_kwargs.update(other_kwargs)
        # so you can specify with just string
        is_num = isinstance(attrs, int) or isinstance(attrs, float)
        if isinstance(attrs, str) or is_num:
            attrs = {self.panel_attr:attrs}

        # TODO if fill is made don't update args for constituent lines
        def _panelArgs(panel):
            for dc in panel:
                if dc.isMatch(attrs):
                    dc.setArgs(plot_kwargs)
            return
        
        argnp = np.vectorize(_panelArgs)
        argnp(self.panels[slc])
        return
    
    def setCmap(self, cmap, vals, attrs = {}):
        colors = sns.color_palette(cmap, len(vals))
        for i in range(len(vals)):
            attrs.update({self.panel_attr:vals[i]})
            self.plotArgs(attrs, color = colors[i])
        return
    ########## INTERFACING WITH PANELS ##############################

    def _getSlice(self, slc):
        if slc is None:
            return (slice(0, self.dim[0]), slice(0, self.dim[1]))
        
        elif isinstance(slc, list):
            if len(slc) == 0:
                return (slice(0, self.dim[0]), slice(0, self.dim[1]))
            
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
        
        elif isinstance(slc, str):
            mask = np.zeros(self.dim, dtype = bool)
            s = slc
            if s in self.row_values:
                for rl in range(len(self.row_values)):
                    if s == self.row_values[rl]:
                        mask[rl, :] = True
            if s in self.col_values:
                for cl in range(len(self.col_values)):
                    if s == self.col_values[cl]:
                        mask[:, cl] = True
            return mask
        
        return slc # assume user knows what they're doing
                
                
    def tickArgs(self, xory = 'both', which = 'both', tick_kwargs = {},
            slc = None, **other_kwargs):
        tick_kwargs = copy.deepcopy(tick_kwargs)
        slc = self._getSlice(slc)
        tick_kwargs.update(other_kwargs)

        def _panelTicks(tick_args_panel):
            tick_args_panel[xory][which].update(tick_kwargs)
            return
        
        ticknp = np.vectorize(_panelTicks, cache = True, otypes = [object])
        ticknp(self.tick_args[slc])
        return

    def _applyTickArgs(self):
        nr, nc = self.dim
        for i in range(nr):
            for j in range(nc):
                ax = self.axes[i, j]
                targs = self.tick_args[i, j]
                for xory in targs:
                    for which in targs[xory]:
                        ax.tick_params(axis = xory, which = which,
                                **targs[xory][which])
        return

    def spineArgs(self, which = 'all', spine_kwargs = {}, slc = None,
            **other_kwargs):
        spine_kwargs = copy.deepcopy(spine_kwargs)
        spine_kwargs.update(other_kwargs)
        slc = self._getSlice(slc)

        def _setSpine(spine_args_panel):
            if which == 'all':
                spines = ['bottom', 'top', 'right', 'left']
                for s in spines:
                    spine_args_panel[s].update(spine_kwargs)
            else:
                spine_args_panel[which].update(spine_kwargs)
            return

        spinenp = np.vectorize(_setSpine, cache=True, otypes=[object])
        spinenp(self.spine_args[slc])
        return

    def _applySpineArgs(self):
        nr, nc = self.dim
        for i in range(nr):
            for j in range(nc):
                ax = self.axes[i, j]
                for side in self.spine_args[i, j]:
                    
                    sargs = self.spine_args[i, j][side]
                    ax.spines[side].set(**sargs)
        return
    
    def plotOrder(self, order = []):
        self.plot_order = order
        return
    
    def axisArgs(self, axis_kwargs = {}, slc = None, **other_kwargs):
        axis_kwargs = copy.deepcopy(axis_kwargs)
        slc = self._getSlice(slc)
        axis_kwargs.update(other_kwargs)
        def _panelAxis(axis_args_panel):
            axis_args_panel.update(axis_kwargs)
            return

        axisnp = np.vectorize(_panelAxis, cache = True, otypes = [object])
        axisnp(self.axis_args[slc])
        return

    def _applyAxisArgs(self):
        nr, nc = self.dim
        for i in range(nr):
            for j in range(nc):
                self.axes[i, j].set(**self.axis_args[i, j])
        return
    
    def legendArgs(self, leg_kwargs = {}, slc = -np.inf, 
            rm_legend = False, **other_kwargs):
        leg_kwargs = copy.deepcopy(leg_kwargs)
        leg_kwargs.update(other_kwargs)
        self.leg_args.update(leg_kwargs)
        self.rm_legend = rm_legend
        if not slc == -np.inf:
            self.legend_slc = slc
        return

    def setLegend(self, kwargs = {}, **other_kwargs):
        if not self.is_plotted:
            msg = "Not plotted, so no legend to give arguments to."
            raise ValueError(msg)
        
        kwargs = copy.deepcopy(kwargs)
        kwargs.update(other_kwargs)

        def _panelLegend(axis):
            leg = axis.get_legend()
            leg.set(**kwargs)
            return
        
        legnp = np.vectorize(_panelLegend, cache = True, otypes = [object])
        legnp(self.axes[self.legend_slc])
        return
    
    def _makeLegend(self):
        legendParams = self.leg_args
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
        rowlabels = self.row_label_text
        
        if rowlabels:
            pos = self.row_label_pos
            colidx = self.row_label_col
            text_kwargs = self.row_label_args
            for i in range(self.dim[0]):
                p = self.axes[i, colidx]
                p.text(pos[0], pos[1], rowlabels[i],
                        transform = p.transAxes, **text_kwargs)
        return

    def rowLabels(self, rowlabels = [], pos = [], colidx = None):
        if rowlabels:
            self.row_label_text = rowlabels
        
        if pos:
            self.row_label_pos = pos
        
        if not colidx is None:
            self.row_label_col = colidx

        return

    def rowLabelArgs(self, text_kwargs = {}, **other_kwargs):
        text_kwargs = copy.deepcopy(text_kwargs)
        text_kwargs.update(other_kwargs)
        self.row_label_args.update(text_kwargs)
        return
    
    def _makeColLabels(self):
        
        collabels = self.col_label_text
        
        if collabels:
            text_kwargs = self.col_label_args
            rowidx = self.col_label_row
            pos = self.col_label_pos
            for i in range(self.dim[1]):
                p = self.axes[rowidx, i]
                p.text(pos[0], pos[1], collabels[i],
                        transform = p.transAxes, **text_kwargs)
        return
    
    def colLabels(self, collabels = [], pos = [], rowidx = None):
        if collabels:
            self.col_label_text = collabels
        
        if pos:
            self.col_label_pos = pos
        
        if not rowidx is None:
            self.col_label_row = rowidx
        
        return

    def colLabelArgs(self, text_kwargs = {}, **other_kwargs):
        text_kwargs = copy.deepcopy(text_kwargs)
        text_kwargs.update(other_kwargs)
        self.col_label_args.update(text_kwargs)
        return
    
    def annotatePanel(self, text, pos, idx, text_kwargs = {}, 
            **other_kwargs):
        text_kwargs = copy.deepcopy(text_kwargs)
        text_kwargs.update(other_kwargs)

        self.panel_annotations.append((text, pos, idx, text_kwargs))
        return
    

    ##### INTERFACE WITH THE FIGURE #################################

    def annotateFig(self, text, pos, text_kwargs = {}, 
            **other_kwargs):
        text_kwargs = copy.deepcopy(text_kwargs)
        text_kwargs.update(other_kwargs)
        self.fig_annotations.append((text, pos, text_kwargs))
        return

    def _makeAnnotations(self):
        for pa in self.panel_annotations:
            text = pa[0]; pos = pa[1]
            idx = pa[2]; text_kwargs = pa[3]
            ax = self.axes[idx]
            ax.text(pos[0], pos[1], text, 
                    transform = ax.transAxes, **text_kwargs)
        
        for fa in self.fig_annotations:
            text = fa[0]; pos = fa[1]
            text_kwargs = fa[2]
            self.fig.text(pos[0], pos[1], text, **text_kwargs)
        return
    ############ PLOTTING ROUTINES ##################################
    
    def plotPanel(self, rowidx, colidx):
        idx = (rowidx, colidx)
        panel = self.panels[idx]

        if self.plot_order:
            plotted_idx = []
            for po in self.plot_order:
                attr_dict = {self.panel_attr:po}
                for dc in range(len(panel)):
                    if panel[dc].isMatch(attr_dict):
                        panel[dc].plot(self.axes[idx])
                        plotted_idx.append(dc)

            for dc in range(len(panel)):
                if dc not in plotted_idx:
                    panel[dc].plot(self.axes[idx])
        else:
            for dc in panel:
                dc.plot(self.axes[idx])
        return

    def plot(self, subfig = None, axes = None):
        
        if subfig is None:
            if self.fig is None:
                self._makeFig()
            self._applyFigArgs()
        else:
            self.setFig(subfig)
        
        if axes is None:
            self._makeAxes()
        else:
            self.axes = axes
        self._applyAxisArgs()
        self._applySpineArgs()
        self._applyTickArgs()

        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.plotPanel(i, j)
        
        if not self.rm_legend:
            self._makeLegend()
        
        self._makeColLabels()
        self._makeRowLabels()
        self._makeXLabel()
        self._makeYLabel()
        self._makeAnnotations()
        self._makeAltPlots()
        self.is_plotted = True
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
    
    def norm(self, num_attrs, denom_attrs, plot_kwargs = {}, idx = 1,
            **other_kwargs):
        #TODO check to make sure that this doesn't change original
        # stored data values

        plot_kwargs.update(other_kwargs)
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
                        dc.setArgs(plot_kwargs)
                    elif dc.isMatch(num_attrs):
                        num_list.append(dc)
                if norm is not None:
                    for num in num_list:
                        ndata = num.getData()
                        ndata[idx] /= norm
                        num.setData(ndata)
        
        return

                        
    def fill(self, attrs, fill_kwargs = {}, slc = None, 
            **other_kwargs):
        
        fill_kwargs.update(other_kwargs)
        slc = self._getSlice(slc)

        def _panelFill(panel):
            data = None
            x = 0; ymins = 0; ymaxs = 0
            labels = []; colors = []
            match_found = False
            for dc in panel:
                if dc.isMatch(attrs):
                    if 'no_fill' not in dc.storage:
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
                        fill_kwargs['label'] = labels[0]
                if len(colors) > 0:
                    if colors.count(colors[0]) == len(colors):
                        fill_kwargs['color'] = colors[0]
                filldc = DataContainer([x, ymins, ymaxs])
                filldc.update(attrs)
                filldc.add('figrid_process', 'fill')
                def _plotFill(ax, data, kwargs):
                    ax.fill_between(data[0], data[1], data[2], **kwargs)
                    return
                
                filldc.setFunc(_plotFill)
                filldc.setArgs(fill_kwargs)
                panel.append(filldc)
            return

        fillnp = np.vectorize(_panelFill, cache = True, otypes=[object])
        fillnp(self.panels[slc])
        return

    ##### CONVENIENCE METHODS #######################################
    def axisLabelArgs(self, xory, text_kwargs = {}, **other_kwargs):
        text_kwargs = copy.deepcopy(text_kwargs)
        text_kwargs.update(other_kwargs)
        self.axis_label_args[xory].update(text_kwargs)
        return
    
    def setXLabel(self, text = '', pos = [], panels = [], 
            text_kwargs = {}, **other_kwargs):
        text_kwargs.update(other_kwargs)
        if panels:
            if isinstance(text, list) and not len(text) == len(panels):
                raise ValueError("length of text list must match panel list")
            self.axis_label_panels['y'] = panels
        
        self.axis_label_pos['x'] = pos
        self.axis_label_text['x'] = text
        self.axis_label_args['x'].update(text_kwargs)
        return
    
    def _defaultAxLabelPos(self, xory, slc):
        if isinstance(slc, int) or isinstance(slc, float):
            slc = slice(slc, slc + 1)
        if xory == 'x':
            fl = self.calculateFigsize()[0]
            pw = self.gspec_args['width_ratios'] * self.panelsize[0]
            xb = self.gspec_args['xborder'] * self.panelsize[0]
            bt = self.gspec_args['wspace'] * self.panelsize[0]
            # pos = [(0.5 * (fl - np.sum(xb)) + xb[0]) / fl, 0]
            xstart = np.sum(bt[:slc.start]) + \
                np.sum(pw[:slc.start]) + xb[0]
            pos = [(0.5 * (np.sum(pw[slc]) + np.sum(bt[slc])) + xstart) / fl, 0]
        elif xory == 'y':
            fh = self.calculateFigsize()[1]
            yb = self.gspec_args['yborder'] * self.panelsize[1]
            ph = self.gspec_args['height_ratios'] * self.panelsize[1]
            bt = self.gspec_args['hspace'] * self.panelsize[1]
            # pos = [0, (0.5 * (fh - np.sum(yb)) + yb[1])/fh]
            ystart = yb[0] + np.sum(bt[:slc.start]) + \
                np.sum(ph[:slc.start])
            pos = [0, 1 - (0.5 * (np.sum(ph[slc]) + np.sum(bt[slc])) + ystart) / fh]
        return pos

    def _makeXLabel(self):

        default_args = {'ha':'center', 'va':'bottom'}
        default_args.update(self.axis_label_args['both'])
        default_args.update(self.axis_label_args['x'])

        for i, slc in enumerate(self.axis_label_panels['x']):
            pos = self.axis_label_pos['x']
            text = self.axis_label_text['x']
            if not pos:
                pos = self._defaultAxLabelPos('x', slc)
            else:
                pos = pos[i]
            if isinstance(text, list):
                text = text[i]
            
            self.annotateFig(text, pos, default_args)
        return
    
    def setYLabel(self, text = '', pos = [], panels = [], 
            text_kwargs = {}, **other_kwargs):
        
        text_kwargs.update(other_kwargs)
        # if panels is given, then only apply label to those panels 
        
        if panels:
            if isinstance(text, list) and not len(text) == len(panels):
                raise ValueError("length of text list must match panel list")

            self.axis_label_panels['y'] = panels
        
        self.axis_label_text['y'] = text
        self.axis_label_pos['y'] = pos
        self.axis_label_args['y'].update(text_kwargs)
        return

    def _makeYLabel(self):

        default_args = {'ha':'left', 'va':'center', 
                'rotation':'vertical'}
        default_args.update(self.axis_label_args['both'])
        default_args.update(self.axis_label_args['y'])

        for i, slc in enumerate(self.axis_label_panels['y']):
            pos = self.axis_label_pos['y']
            text = self.axis_label_text['y']
            if not pos:
                pos = self._defaultAxLabelPos('y', slc)
            else:
                pos = pos[i]
            if isinstance(text, list):
                text = text[i]
            
            self.annotateFig(text, pos, default_args)
        return
    
    def setDefaultTicksParams(self):
        # slice of everything but bottom row
        topslc = (slice(0, -1), slice(None))

        # slice of everything but leftmost column
        rightslc = (slice(None), slice(1, None))
        
        if self.dim[0] > 1:
            params = {'labelbottom':False}
            self.tickArgs('x', tick_kwargs = params, slc = topslc)
        
        if self.dim[1] > 1:
            params = {'labelleft':False}
            self.tickArgs('y', tick_kwargs = params, slc = rightslc)
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

    def autoFill(self, fill_kwargs = {}, **other_kwargs):
        fill_kwargs.update(other_kwargs)
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
    
    def autoNorm(self, denom_attr_dict, must_match = [], idx = 1):
        denom_arr = np.empty(self.dim, dtype = object)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                panel = self.panels[i, j]
                denom_arr[i, j] = []
                for dc in panel:
                    if dc.isMatch(denom_attr_dict):
                        denom_arr[i, j].append(dc)
                        dc.setArgs({'visible':False, 'label':'_nolegend_'})
                        dc.store('no_fill', True)


        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                panel = self.panels[i, j]
                denoms = denom_arr[i, j]
                for d in denoms:
                    norm = copy.copy(d.getData()[idx])
                    match_attrs = {}
                    for mm in must_match:
                        match_attrs[mm] = d.get(mm)
                    for dc in panel:

                        if dc.isMatch(match_attrs):
                            dcdata = dc.getData()
                            zmask = (norm > 0) | (norm < 0)
                            dcdata[idx][zmask] = dcdata[idx][zmask] / norm[zmask]
                            dcdata[idx][~zmask] = 0

                            dc.setData(dcdata)
    
        return

    def _makeAltPlots(self):

        for op in self.other_plots:
            if op[0] == 'plot_ones':
                slc = op[1]
                plot_kwargs = op[2]
            
                def _plotOnes(ax):
                    ax.plot(ax.get_xlim(), [1, 1], **plot_kwargs)
                    return
                
                np_ones = np.vectorize(_plotOnes, cache = True, 
                            otypes = [object])
                np_ones(self.axes)
        
        return
    def plotOnes(self, plot_kwargs = {'color':'gray', 'linestyle':'--'},
            slc = None, **other_kwargs):
        
        plot_kwargs = copy.deepcopy(plot_kwargs)
        plot_kwargs.update(other_kwargs)
        slc = self._getSlice(slc)

        self.other_plots.append(('plot_ones', slc, plot_kwargs))
        return

    def clf(self):
        plt.clf()
        plt.close()
        return
    
    def save(self, path, 
            save_kwargs = {'bbox_inches':'tight', 'facecolor':'auto'},
            **other_kwargs):
        
        save_kwargs.update(other_kwargs)
        self.fig.savefig(path, **save_kwargs)
        return

    # STATIC METHODS

    @staticmethod
    def combine(figrid_arr, wspace = 0, hspace = 0):
        figrid_arr = np.array(figrid_arr)
        if len(figrid_arr.shape) == 1:
            figrid_arr = np.reshape(figrid_arr, (1, -1))
        
        
        nrows = figrid_arr.shape[0]
        ncols = figrid_arr.shape[1]
        # total figsize
        figwidths = np.zeros((nrows, ncols))
        figheights = np.zeros((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                figwidths[i, j], figheights[i, j] = \
                    figrid_arr[i, j].calculateFigsize()

        figsize = np.zeros(2)
        figsize[0] = np.max(np.sum(figwidths, axis = 1))
        figsize[1] = np.max(np.sum(figheights, axis = 0))


        figsize[0] += wspace * (ncols - 1)
        figsize[1] += hspace * (nrows - 1)
        

        # get width/height ratios
        width_ratios = figwidths[0, :] / np.max(figwidths[0, :])
        height_ratios = figheights[:, 0] / np.max(figheights[:, 0])
        # TODO make more general way of obtaining the ratios
        # TODO wspace not working, is just changing figsize and nothing else
        fig = plt.figure(figsize = figsize)
        subfigs = fig.subfigures(nrows, ncols, 
                wspace = wspace * (ncols - 1),
                hspace = hspace * (nrows - 1),
                width_ratios = width_ratios,
                height_ratios = height_ratios)
        subfigs = np.reshape(subfigs, (nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                sf = subfigs[i, j]
                fg = figrid_arr[i, j]
                fg.plot(sf)
        
        return fig
    
    def addCbar(self, loc = 'row', norm = 'lin', cmap = 'viridis',
            cbar_kwargs = {}, **other_kwargs):
        
        cbar_kwargs.update(other_kwargs)

        def _calcCbarSize(type):
            if type == 'horizontal':
                h = self.panelsize[1]
                cbar_width = self.panelsize[0]
                cbar_height = h * np.sum(self.gspec_args['height_ratios'])
                cbar_height += h * np.sum(self.gspec_args['yborder'])
                cbar_height += h * np.sum(self.gspec_args['hspace'])
                figsize = self.calculateFigsize()
                figsize[0] += cbar_width

            
            return cbar_width, cbar_height, figsize

        def _getClims(axslc):
            axlist = np.ravel(self.axes[axslc])
            clim = np.zeros(2)
            clim[0] = np.inf; clim[1] = -np.inf
            for col in range(len(axlist)):
                ax = axlist[col]
                if len(ax.images) > 0:
                    im = ax.images[0]
                    temp_clim = im.get_clim()
                    if clim[0] > temp_clim[0]:
                        clim[0] = temp_clim[0]
                    if clim[1] < temp_clim[1]:
                        clim[1] = temp_clim[1]
            
            return clim

        def _setCbar(axslc, smap):
            axlist = np.ravel(self.axes[axslc])

            for col in range(len(axlist)):
                ax = axlist[col]
                if len(ax.images) > 0:
                    im = ax.images[0]
                    im.set(
                        clim = smap.get_clim(),
                        norm = smap.norm,
                        cmap = smap.get_cmap()
                    )
            
            return

        def _makeSmap(clim, norm):
            if isinstance(norm, str):
                if norm == 'lin':
                    norm = mpl.colors.Normalize(clim[0], clim[1])
                elif norm == 'log':
                    norm = mpl.colors.LogNorm(clim[0], clim[1])
                else:
                    raise NotImplementedError(
                        "%s not currently accepted keyword norm input..."%norm
                        )
            smap = mpl.cm.ScalarMappable(norm = norm, cmap = cmap)
            return smap
        
        def _makeSubplots(fig, gspec, cbar_size):
            ga = gspec
            nrows = len(ga['height_ratios'])
            ncols = len(ga['width_ratios'])
            hrs = ga['height_ratios']
            wrs = ga['width_ratios']
            yb = ga['yborder']; xb = ga['xborder']
            hs = ga['hspace']; ws = ga['wspace']
            figwidth = cbar_size[0]; figheight = cbar_size[1]

            panel_width = self.panelsize[0]
            panel_height = self.panelsize[1]
            width_ratios = wrs * panel_width; height_ratios = hrs * panel_height
            xborder = xb * panel_width; yborder = yb * panel_height
            wspace = ws * panel_width; hspace = hs * panel_height
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
            
            return axes
        
        def _applyTicksAxisArgs(ax, axis_kwargs, tick_kwargs):
            
            ax.set(**axis_kwargs)
            for xory in tick_kwargs:
                for which in tick_kwargs[xory]:
                    ax.tick_params(axis = xory,
                        which = which, **tick_kwargs[xory][which])
            
            return
        
        if loc == 'row':
            # calculate cbar subfig size
            cbar_width, cbar_height, figsize = _calcCbarSize('horizontal')

            figwidth_ratios = np.array([1, cbar_width/figsize[0]])
            figheight_ratios = np.array([1])


            fig = plt.figure(figsize = figsize)
            subfigs = fig.subfigures(1, 2, wspace = 0, hspace = 0,
                    width_ratios = figwidth_ratios,
                    height_ratios = figheight_ratios)
            
            subfigs = np.reshape(subfigs, (1, 2))

            # make usual plots
            self.gspec_args['xborder'][1] = self.gspec_args['wspace'][-1]
            self.plot(subfig = subfigs[0, 0])
            
            # place subplots in cbar figure
            cbar_gspec = copy.deepcopy(self.gspec_args)
            cbar_gspec['width_ratios'] = [1]
            cbar_gspec['xborder'][0] = 0
            cbar_gspec['wspace'] = []
            cbar_axes = _makeSubplots(subfigs[0, 1], cbar_gspec, 
                    [cbar_width, cbar_height])
            # get the color limits for the colorbar

            clims = np.empty((self.dim[0], 2))
            for row in range(self.dim[0]):
                axslc = (row, slice(None))
                clims[row, :] = _getClims(axslc)
            
            # now make the colorbar uniform across the row
            # then create the colorbar
            label = None
            for row in range(self.dim[0]):
                axslc = (row, slice(None))
                smap = _makeSmap(clims[row, :], norm)
                _setCbar(axslc, smap)

                cax = cbar_axes[row, 0]
               
                if 'label' in cbar_kwargs:
                    label = cbar_kwargs.pop('label')
                
                subfigs[0, 1].colorbar(cax = cax, mappable = smap, 
                        **cbar_kwargs)

                _applyTicksAxisArgs(cax, self.axis_args[0, 0], self.tick_args[0, 0])
                if 'aspect' in cbar_kwargs:
                    _applyTicksAxisArgs(cax, 
                        {'aspect':cbar_kwargs['aspect']}, {})
                
                if label is not None:
                    cax.set_ylabel(label, 
                        **self.axis_label_args['both'])
                
                
                
        
        elif loc == 'right':
            cbar_width, cbar_height, figsize = _calcCbarSize('horizontal')
            figwidth_ratios = np.array([1, cbar_width/figsize[0]])
            figheight_ratios = np.array([1])


            fig = plt.figure(figsize = figsize)
            subfigs = fig.subfigures(1, 2, wspace = 0, hspace = 0,
                    width_ratios = figwidth_ratios,
                    height_ratios = figheight_ratios)
            
            subfigs = np.reshape(subfigs, (1, 2))

            # make usual plots
            self.plot(subfig = subfigs[0, 0])

            # get colorbar limits
            axslc = (slice(None), slice(None))
            clims = _getClims(axslc)

            smap = _makeSmap(clims, norm)
            _setCbar(axslc, smap)
            ax = subfigs[0, 1].subplot()
            subfigs[0, 1].colorbar(cax = ax, mappable = smap, **cbar_kwargs)
            
        return fig








            

            






    
