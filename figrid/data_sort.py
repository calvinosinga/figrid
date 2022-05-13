from figrid.data_container import DataContainer
from figrid.figrid import Figrid
import copy
import numpy as np

"""
save orders of rows, columns, names for props
create figrid objects, spit them out
save standard figrid properties - ticks, axis, panel size, gspec_kw
"""

class DataSort():
    def __init__(self, dclist = []):
        self.dclist = dclist
        self.tick_args = {}
        self.axis_args = {}
        self.figrid_args = {}
        self.attr_orders = {}
        self.legend_args = {}
        self.legend_slice = None
        self.attr_args = {}
        self.fig_args = {}
        self.display_names = {}
        self.axis_labels = {}
        self.row_label_args = {}
        self.col_label_args = {}
        return
    
    ##### I/O METHODS ###############################################

    def loadHdf5(self):
        return
    
    def loadResults(self, results, new_props = {}):
        for r in results:
            data = [r.xvalues, r.yvalues, r.zvalues]
            dc = DataContainer(data)
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

    def getData(self):
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
    
    ##### POST-PROCESS DATA #########################################

    def makeFill(self, attrs, fillkwargs = {}):
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
                if td in default_args and td not in fillkwargs:
                    fillkwargs[td] = default_args[td]
                
            def _plotFill(ax, data, kwargs):
                ax.fill_between(data[0], data[1], data[2], **kwargs)
                return
            
            filldc.setFunc(_plotFill)

            filldc.setArgs(fillkwargs)
            self.append(filldc)
        return
    
    ##### INTERFACING WITH FIGRID ###################################

    def tickArgs(self, tick_args, xory = 'both', which = 'both'):
        tp = self.tick_args
        if not xory in tp:
            tp[xory] = {}
        
        if not which in tp[xory]:
            tp[xory][which] = {}
        
        tp[xory][which].update(tick_args)
        return
    
    def axisArgs(self, axis_args):
        self.axis_args.update(axis_args)
        return

    def figArgs(self, fig_args):
        self.fig_args.update(fig_args)
        return
    
    def figridArgs(self, fgrid_args):
        self.figrid_args.update(fgrid_args)
        return
    
    def legendArgs(self, leg_args, slc = None):
        self.legend_args.update(leg_args)
        self.legend_slice = slc
        return

    def setOrder(self, attr, order):
        self.attr_orders[attr] = order
        return
    
    def plotArgs(self, attr, val, args):
        if attr not in self.attr_args:
            self.attr_args[attr] = {}
        if val not in self.attr_args[attr]:
            self.attr_args[attr][val] = {}
        self.attr_args[attr][val].update(args)
        return

    def setAxisLabel(self, xory, label, pos = [], 
            txtkw = {}):
        self.axis_labels[xory] = (label, pos, txtkw)
        return
    
    def rowLabelArgs(self, row_attr = '_default_', 
            pos = [], txtkw = {}, colidx = 0):
        self.row_label_args[row_attr] = (pos, txtkw, colidx)
        return
    
    def colLabelArgs(self, col_attr = '_default_', 
            pos = [], txtkw = {}, rowidx = 0):
        self.col_label_args[col_attr] = (pos, txtkw, rowidx)
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
            in_attrs = {}, rm_attrs = {},
            figkw = {}):

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
        fa.update(figkw)
        self.figrid_args['figkw'] = fa
        figrid.makeFig(nrows, ncols, **self.figrid_args)
        for xory in self.tick_args:
            for which in self.tick_args[xory]:
                figrid.setTicks(self.tick_args[xory][which], xory, which)

        figrid.setAxisParams(self.axis_args)
        figrid.setLegend(self.legend_args, self.legend_slice)

        if 'x' in self.axis_labels:
            figrid.setXLabel(*self.axis_labels['x'])
        if 'y' in self.axis_labels:
            figrid.setYLabel(*self.axis_labels['y'])
        
        if row_attr in self.row_label_args:
            figrid.setRowLabels(row_labels, 
                    *self.row_label_args[row_attr])
        elif '_default_' in self.row_label_args:
            figrid.setRowLabels(row_labels, 
                    *self.row_label_args['_default_'])
        
        if col_attr in self.col_label_args:
            figrid.setColLabels(col_labels, *self.col_label_args[col_attr])
        elif '_default_' in self.col_label_args:
            figrid.setColLabels(col_labels, *self.col_label_args['_default_'])
        
        return figrid
    
     
