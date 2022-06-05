from figrid.data_container import DataContainer
from figrid.figrid import Figrid
import copy
import h5py as hp
import numpy as np


class DataSort():
    def __init__(self, dclist = []):
        self.dclist = dclist

        self.tick_args = {}
        self.axis_args = {'panel':{}, 'row':{}, 'col':{}}
        self.gspec_args = {}
        self.attr_orders = {}
        self.legend_args = {}
        self.spine_args = {}
        self.legend_slice = None
        self.attr_args = {}
        self.fig_args = {}
        self.display_names = {}
        self.axis_label_args = {}
        self.row_label_args = {}
        self.col_label_args = {}
        return
    
    ##### I/O METHODS ###############################################

    def loadHdf5(self, path):
        f = hp.File(path, 'r')
        for k in list(f.keys()):
            data = f[k][:]
            dc = DataContainer(data)
            dc.update(f[k].attrs)
            self.dclist.append(dc)
        return
    
    def loadResults(self, results, new_props = {}):
        for r in results:
            data = [r.xvalues, r.yvalues, r.zvalues]
            dc = DataContainer(data)
            dc.store('count', r.count)
            dc.update(new_props)
            dc.update(r.props)
            self.dclist.append(dc)
        return

    ##### DATA ACCESS/MANAGEMENT ####################################

    def append(self, dataContainer):
        self.dclist.append(dataContainer)
        return
    
    def extend(self, dataContList):
        self.dclist.extend(dataContList)
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

    def getList(self):
        return self.dclist
        
    def _rmMatching(self, desired_attrs, dclist):
        rmidx = []
        for dc in range(len(dclist)):
            if dclist[dc].isMatch(desired_attrs):
                rmidx.append(dc)
        rmidx = np.array(rmidx)
        for rm in range(len(rmidx)):

            dclist.pop(rmidx[rm])
            rmidx = rmidx - 1
            
        return

    def getMatching(self, desired_attrs, rm_attrs = {}):
        matches = []
        for dc in self.dclist:
            if dc.isMatch(desired_attrs):
                matches.append(dc)
        
        # default behavior is to give True if given 
        # an empty dict, so only do remove if not
        # empty
        if rm_attrs:
            self._rmMatching(rm_attrs, matches)
        return matches
    
    def printMatching(self, desired_attrs, rm_attrs = {}):
        matches = self.getMatching(desired_attrs, rm_attrs)
        for m in matches:
            print(m.attrs)
            print()
        return
    ##### POST-PROCESS DATA #########################################

    def makeFill(self, attrs, fill_kwargs, **other_kwargs):

        fill_kwargs.update(other_kwargs)

        attrs = copy.deepcopy(attrs)
        attrs['figrid_process'] = 'no key found'
        matches = self.getMatching(attrs)
        for at in attrs:
            if isinstance(attrs[at], list):
                name = ''
                for l in attrs[at]:
                    name+=l + '_'
                attrs[at] = name[:-1]

        default_args = {}
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
                default_args.update(m.getArgs())
            
            filldc = DataContainer([x, ymins, ymaxs])
            attrs['figrid_process'] = 'fill'
            filldc.update(attrs)
            
            takedefault = ['color', 'label', 'alpha']
            for td in takedefault:
                if td in default_args and td not in fill_kwargs:
                    fill_kwargs[td] = default_args[td]
                
            def _plotFill(ax, data, kwargs):
                ax.fill_between(data[0], data[1], data[2], **kwargs)
                return
            
            filldc.setFunc(_plotFill)

            filldc.setArgs(fill_kwargs)
            self.append(filldc)
        return
    
    ##### INTERFACING WITH FIGRID ###################################

    def tickArgs(self, xory = 'both', 
            which = 'both', tick_kwargs = {}, **other_kwargs):
        """
        Specify the tick keyword arguments to use as a default
        for any figrids that are created. Keyword arguments must
        be compatible with tick_params(...) method in mpl.

        Saved to tick_args as a dictionary with data structure:
        xory -> which -> args

        The user can either give a dictionary for the keywords or 
        give them as keywords in the tickArgs(...) function itself.
        
        Args:
            tick_kwargs (dict, optional): Dictionary of keyword
                arguments. Defaults to {}.
            xory (str, optional): specifies which axis to apply each to. 
                Can be 'x', 'y' or 'both'. Defaults to 'both'.
            which (str, optional): specifies which types of ticks
                to apply the args to. Can be 'major', 'minor' or 'both'.
                Defaults to 'both'.
        """
        
        tick_kwargs.update(other_kwargs)

        tp = self.tick_args
        if not xory in tp:
            tp[xory] = {}
        
        if not which in tp[xory]:
            tp[xory][which] = {}
        
        tp[xory][which].update(tick_kwargs)
        return
    
    def axisArgs(self, panel_attr = '_default_',
            row_attr = '_default_', col_attr = '_default_',
            axis_kwargs = {}, **other_kwargs):
        """
        Specify the Axis keyword arguments (the mpl object) to use
        for any figrids that are created. Keyword arguments must
        be compatible with Axis.set(...) method in mpl.
        
        One can specify row/col attributes, so if a figrid is 
        created with the corresponding attribute along its rows
        or columns it'll use those arguments instead.

        Saved to axis_args as a dictionary with data structure:
        panel/row/col -> panel/row/col attribute -> axis_args

        The user can either give a dictionary for the keywords or 
        give them as keywords in the axisArgs(...) function itself.
        
        Args:
            axis_kwargs (dict, optional): axis keyword arguments. 
                Defaults to {}.
            panel_attr (str, optional): will only use the given 
                arguments if a figrid has this panel_attr. Will use
                '_default_' otherwise. Defaults to '_default_'.
            row_attr (str, optional): will only use the given 
                arguments if a figrid has this row_attr.
                Will use '_default_' arguments otherwise. 
                Defaults to '_default_'.
            col_attr (str, optional): will only use the given 
                arguments if a figrid has this col_attr.
                Will use '_default_' arguments otherwise. 
                Defaults to '_default_'.
        """
        axis_kwargs.update(other_kwargs)
        ### TEMPORARY ###
        if 'panel' not in self.axis_args:
            self.axis_args['panel'] = {}
        if 'row' not in self.axis_args:
            self.axis_args['row'] = {}
        if 'col' not in self.axis_args:
            self.axis_args['col'] = {}
        ############

        pa = self.axis_args['panel']
        ra = self.axis_args['row']
        ca = self.axis_args['col']
        if panel_attr not in pa:
            pa[panel_attr] = {}
        pa[panel_attr].update(axis_kwargs)

        if row_attr not in ra:
            ra[row_attr] = {}
        ra[row_attr].update(axis_kwargs)
        
        if col_attr not in ca:
            ca[col_attr] = {}
        ca[col_attr].update(axis_kwargs)

        return

    def figArgs(self, fig_kwargs = {}, **other_kwargs):
        fig_kwargs.update(other_kwargs)
        self.fig_args.update(fig_kwargs)
        return
    
    def gspecArgs(self, gspec_kwargs = {}, **other_kwargs):
        gspec_kwargs.update(other_kwargs)
        self.gspec_args.update(gspec_kwargs)
        return
    
    def spineArgs(self, which = 'all', spine_kwargs = {},
            **other_kwargs):
        spine_kwargs.update(other_kwargs)

        if which == 'all':
            axes = ['bottom', 'top', 'right', 'left']
            for a in axes:
                if a not in self.spine_args:
                    self.spine_args[a] = {}
                self.spine_args[a].update(spine_kwargs)
        elif which == 'y':
            axes = ['bottom', 'top']
            for a in axes:
                if a not in self.spine_args:
                    self.spine_args[a] = {}
                self.spine_args[a].update(spine_kwargs)
        elif which == 'x':
            axes = ['left', 'right']
            for a in axes:
                if a not in self.spine_args:
                    self.spine_args[a] = {}
                self.spine_args[a].update(spine_kwargs)
        else:
            if which not in self.spine_args:
                self.spine_args[which] = {}
            self.spine_args[which].update(spine_kwargs)
        return
    
    def legendArgs(self, slc = None, leg_kwargs = {}, **other_kwargs):
        leg_kwargs.update(other_kwargs)
        self.legend_args.update(leg_kwargs)
        self.legend_slice = slc
        return

    def setOrder(self, attr, order):
        self.attr_orders[attr] = order
        return
    
    def plotArgs(self, attr, val, plot_kwargs, **other_kwargs):
        plot_kwargs.update(other_kwargs)
        if attr not in self.attr_args:
            self.attr_args[attr] = {}
        if val not in self.attr_args[attr]:
            self.attr_args[attr][val] = {}
        self.attr_args[attr][val].update(plot_kwargs)
        return

    def axisLabelArgs(self, xory = 'both', text_kwargs = {},
            **other_kwargs):
        text_kwargs.update(other_kwargs)
        if xory not in self.axis_label_args:
            self.axis_label_args[xory] = {}
        self.axis_label_args[xory].update(text_kwargs)
        return
    
    def rowLabelArgs(self, row_attr = '_default_', 
            pos = [], text_kwargs = {}, colidx = 0,
            **other_kwargs):
        text_kwargs.update(other_kwargs)
        self.row_label_args[row_attr] = (pos, text_kwargs, colidx)
        return
    
    def colLabelArgs(self, col_attr = '_default_', 
            pos = [], text_kwargs = {}, rowidx = 0,
            **other_kwargs):
        text_kwargs.update(other_kwargs)
        self.col_label_args[col_attr] = (pos, text_kwargs, rowidx)
        return

    def displayAs(self, attr, vals, names):
        """
        Sets the display name for any data that has the attribute
        that matches the given value. This is used for names in
        legends and names in labeling rows and columns.

        Args:
            attr (str): name of the attribute.
            vals (str, list): values of that attribute to assign
                names to.
            names (str, list): the string that will be displayed to
                represent the data.

        Raises:
            ValueError: if vals and names are given as lists,
                they must have the same length.
        """
        if not isinstance(vals, list):
            vals = [vals]
        if not isinstance(names, list):
            names = [names]
        
        if not len(vals) == len(names):
            msg = "names and values should be the same length..."
            raise ValueError(msg)
        
        if not attr in self.display_names:
            self.display_names[attr] = {}

        for v in range(len(vals)):
            self.display_names[attr][vals[v]] = names[v]
        
        return

    def figrid(self, panel_attr, row_attr, col_attr,
            in_attrs = {}, rm_attrs = {}, gspec_kwargs = {},
            fig_kwargs = {}):
        #TODO make sure that the attrs used in row/col actually have 
        # stuff in them

        # figure out the values for the row attribute that
        # are desired from in_attrs, rm_attrs, and if the
        # user set an order for this attribute
        row_values = []
        if row_attr in in_attrs:
            # first check if in in_attrs - if so,
            # the values here are used in the order
            # they are given
            if isinstance(in_attrs[row_attr], list):
                row_values.extend(in_attrs[row_attr])
            else:
                row_values.append(in_attrs[row_attr])
        elif row_attr in self.attr_orders:
            # if user specified an order for this attr,
            # use that order! otherwise just get the
            # values in any order.
            row_values = self.attr_orders[row_attr]
        else:
            row_values = self.getAttrVals(row_attr)
        
        # finally check rm_attrs - if here, then remove equivalent
        # values from vals
        if row_attr in rm_attrs:

            rmvals = rm_attrs[row_attr]
            if not isinstance(rmvals, list):
                rmvals = [rmvals]
            
            for rm in rmvals:
                if rm in row_values:
                    row_values.remove(rm)
        
        # now figure out the values to include for the columns, 
        # in a similar process to the row attr
        col_values = []
        if col_attr in in_attrs:
            if isinstance(in_attrs[col_attr], list):
                col_values.extend(in_attrs[col_attr])
            else:
                col_values.append(in_attrs[col_attr])
        elif col_attr in self.attr_orders:
            col_values = self.attr_orders[col_attr]
        else:
            col_values = self.getAttrVals(col_attr)


        if col_attr in rm_attrs:

            rmvals = rm_attrs[col_attr]
            if not isinstance(rmvals, list):
                rmvals = [rmvals]
            
            for rm in rmvals:
                if rm in col_values:
                    col_values.remove(rm)
        
        
        # if user specified a particular way to display the
        # row/col attributes in the figure, save those here
        row_labels = []
        col_labels = []

        # the reasoning for this is that long, descriptive
        # labels may be desired but would be inconvenient
        # to access them with long strings.
        # So this setup allows for you to use shorthands
        # for the dictionary
        if row_attr in self.display_names:
            names_for_attr = self.display_names[row_attr]
            for v in row_values:
                if v in names_for_attr:
                    row_labels.append(names_for_attr[v])
                else: 
                    # if name not specified, the attr name is
                    # used by default
                    row_labels.append(v)
        else:
            for v in row_values:
                row_labels.append(v)
        

        if col_attr in self.display_names:
            names_for_attr = self.display_names[col_attr]
            for v in col_values:
                if v in names_for_attr:
                    col_labels.append(names_for_attr[v])
                else: 
                    # if name not specified, the value is
                    # used by default
                    col_labels.append(v)
        else:
            for v in col_values:
                col_labels.append(v)
        

        rowtup = (str(row_attr), str(row_values))
        coltup = (str(col_attr), str(col_values))
        print('The row values for %s: %s'%rowtup)
        print('The column values for %s: %s'%coltup)
        
        # the number of rows and columns for the figrid (must have at least one)
        nrows = max(1, len(row_values))
        ncols = max(1, len(col_values))
        
        # obtain the data that should belong to each panel
        panels = np.empty((nrows, ncols), dtype = object)
        for i in range(nrows):
            for j in range(ncols):
                
                attr_for_panel = copy.deepcopy(in_attrs)
                if j < len(col_values):
                    
                    attr_for_panel[col_attr] = col_values[j]

                if i < len(row_values):
                    attr_for_panel[row_attr] = row_values[i]

                containers_for_panel = \
                    self.getMatching(attr_for_panel, rm_attrs)

                containers_for_panel = \
                    copy.deepcopy(containers_for_panel)

                for dc in containers_for_panel:
                    pval = dc.get(panel_attr)
                    if panel_attr in self.display_names:
                        if pval in self.display_names[panel_attr]:
                            l = self.display_names[panel_attr][pval]
                            dc.setArgs({'label':l})
                    
                    if panel_attr in self.attr_args:
                        if pval in self.attr_args[panel_attr]:
                            plot_args = self.attr_args[panel_attr][pval]
                            dc.setArgs(plot_args)
                
                panels[i, j] = containers_for_panel
        
        figrid = Figrid(panels, panel_attr, row_values, col_values)
        fa = copy.deepcopy(self.fig_args)
        fa.update(fig_kwargs)
        fgargs = copy.deepcopy(self.gspec_args)
        fgargs.update(gspec_kwargs)
        fgargs['figkw'] = fa
        figrid.makeFig(nrows, ncols, **fgargs)
        for xory in self.tick_args:
            for which in self.tick_args[xory]:
                figrid.tickArgs(self.tick_args[xory][which], xory, which)

        pa = self.axis_args['panel']
        ra = self.axis_args['row']
        ca = self.axis_args['col']
        if col_attr in ca:
            figrid.axisArgs(ca[col_attr])
        elif '_default_' in ca:
            figrid.axisArgs(ca['_default_'])
        
        if row_attr in ra:
            figrid.axisArgs(ra[row_attr])
        elif '_default_' in ca:
            figrid.axisArgs(ra['_default_'])

        if panel_attr in pa:
            figrid.axisArgs(pa[panel_attr])
        elif '_default_' in pa:
            figrid.axisArgs(pa['_default_'])

        for which in self.spine_args:
            figrid.spineArgs(self.spine_args[which], which)
        
        figrid.legendArgs(self.legend_args, self.legend_slice)

        if panel_attr in self.attr_orders:
            figrid.plotOrder(self.attr_orders[panel_attr])
        
        figrid.axisLabelArgs(self.axis_label_args)
                    
        if row_attr in self.row_label_args:
            figrid.rowLabelArgs(row_labels, 
                    *self.row_label_args[row_attr])
        elif '_default_' in self.row_label_args:
            figrid.rowLabelArgs(row_labels, 
                    *self.row_label_args['_default_'])
        
        if col_attr in self.col_label_args:
            figrid.colLabelArgs(col_labels, *self.col_label_args[col_attr])
        elif '_default_' in self.col_label_args:
            figrid.colLabelArgs(col_labels, *self.col_label_args['_default_'])
        
        return figrid
    
     
    def combineFigrids(self, fg_init, fg_add, loc = 'bottom', spacing = None):
        nrows = fg_init.dim[0]
        ncols = fg_init.dim[1]
        hspace = fg_init.hspace / fg_init.panel_length
        wspace = fg_init.wspace / fg_init.panel_length

        if len(hspace) == 0 and len(fg_add.hspace) == 0 and spacing is None:
            raise ValueError("for two figrids of length 1 must" + \
                " define a spacing.")

        if loc == 'bottom':
            nrows += fg_add.dim[0]
            newslc = (slice(fg_init.dim[0], None), slice(None))
            initslc = (slice(0, fg_init.dim[0]), slice(None))

            fg_init.row_labels.extend(fg_add.row_labels)
            fg_init.row_values.extend(fg_add.row_values)
            
            new_hspace = np.zeros(nrows - 1)
            transidx = len(hspace)
            new_hspace[:transidx] = hspace[:]
            if not spacing is None:
                new_hspace[transidx] = spacing
            else:
                new_hspace[transidx] = hspace[-1]
            if len(new_hspace) > transidx + 1:
                new_hspace[transidx+1:] = fg_add.hspace[:]
            new_wspace = wspace
        elif loc == 'top':
            nrows += fg_add.dim[0]
            newslc = (slice(0, fg_add.dim[0]), slice(None))
            initslc = (slice(fg_add.dim[0], None), slice(None))

            fg_init.row_labels = fg_add.row_labels.extend(fg_init.row_labels)
            fg_init.row_values = fg_add.row_values.extend(fg_init.row_values)

            new_hspace = np.zeros(nrows - 1)
            transidx = len(fg_add.hspace)
            new_hspace[transidx:] = hspace[:]
            if not spacing is None:
                new_hspace[transidx] = spacing
            else:
                new_hspace[transidx] = hspace[0]
            if transidx - 1 >= 0:
                new_hspace[:transidx - 1] = fg_add.hspace[:]
            new_wspace = wspace
        elif loc == 'right':
            ncols += fg_add.dim[1]
            newslc = (slice(None), slice(fg_init.dim[1], None))
            initslc = (slice(None), slice(0, fg_init.dim[1]))

            fg_init.col_labels.extend(fg_add.col_labels)
            fg_init.col_values.extend(fg_add.col_values)
            
            new_wspace = np.zeros(ncols - 1)
            transidx = len(wspace)
            new_wspace[:transidx] = wspace[:]
            if not spacing is None:
                new_wspace[transidx] = spacing
            else:
                new_wspace[transidx] = wspace[-1]
            if len(new_wspace) > transidx + 1:
                new_wspace[transidx+1:] = fg_add.wspace[:]
            new_hspace = hspace
        elif loc == 'left':
            ncols += fg_add.dim[1]
            newslc = (slice(None), slice(0, fg_add.dim[1]))
            initslc = (slice(None), slice(fg_add.dim[1], None))

            fg_init.col_labels = fg_add.col_labels.extend(fg_init.col_labels)
            fg_init.col_values = fg_add.col_values.extend(fg_init.col_values)

            new_wspace = np.zeros(ncols - 1)
            transidx = len(fg_add.wspace)
            new_wspace[transidx:] = wspace[:]
            if not spacing is None:
                new_wspace[transidx] = spacing
            else:
                new_wspace[transidx] = wspace[0]
            if transidx - 1 >= 0:
                new_wspace[:transidx - 1] = fg_add.wspace[:]
            new_hspace = hspace
        else:
            raise ValueError('not accepted location')
        heights = np.zeros(nrows)
        widths = np.zeros(ncols)
        heights[initslc[0]] = fg_init.panel_heights
        widths[initslc[1]] = fg_init.panel_widths
        
        if loc == 'bottom' or loc == 'top':
            heights[newslc[0]] = fg_add.panel_heights
        elif loc == 'left' or loc == 'right':
            widths[newslc[1]] = fg_add.panel_widths
        
        # pl = max(np.max(heights), np.max(widths))
        fgargs = copy.deepcopy(self.gspec_args)
        fgargs['height_ratios'] = heights
        fgargs['width_ratios'] = widths
        fgargs['wspace'] = new_wspace
        fgargs['hspace'] = new_hspace
        fg_init.makeFig(nrows, ncols, **fgargs)
        for xory in self.tick_args:
            for which in self.tick_args[xory]:
                fg_init.tickArgs(self.tick_args[xory][which], xory, which)

        fg_init.axisArgs(self.axis_args)
        fg_init.legendArgs(self.legend_args, self.legend_slice)

        if 'x' in self.axis_label_args:
            fg_init.setXLabel(*self.axis_label_args['x'])
        if 'y' in self.axis_label_args:
            fg_init.setYLabel(*self.axis_label_args['y'])
                    
        fg_init.rowLabelArgs(fg_init.row_labels, *fg_init.row_label_args)
        fg_init.colLabelArgs(fg_init.col_labels, *fg_init.col_label_args)

        newpanels = np.empty((nrows, ncols), dtype = object)
        newpanels[initslc] = fg_init.panels.copy()
        newpanels[newslc] = fg_add.panels.copy()
        fg_init.panels = newpanels
        return fg_init
