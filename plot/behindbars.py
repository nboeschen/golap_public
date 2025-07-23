from matplotlib import pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict


class BehindBars():
    """ Helper Class """
    @dataclass
    class PlotObject:
        xs: list
        ys: list
        label: str
        args: dict = field(default_factory=dict)

    """BehindBars"""
    def __init__(self,total_width=0.6):
        self.plot_objs = defaultdict(list)
        self.total_width = total_width
        self.cats = {}
        self.next_cat = 1

    """
        Sort the gathered plot objects by label lexicograpically.
        Optionally add invisible separators after specified prefixes.
    """
    def sort(self, prefixes=None):
        if prefixes:
            for prefix in prefixes:
                self.add([],[],label=f"{prefix}zzzzz")
        plot_objs_tmp = self.plot_objs.copy()
        self.plot_objs = {}

        for part_of in sorted(plot_objs_tmp.keys()):
            self.plot_objs[part_of] = plot_objs_tmp[part_of][:]
        if not prefixes:
            return
        # delete all delimiters, plus the last unnecessary bar
        for part_of in self.plot_objs.keys():
            if "zzzzz" in part_of:
                self.plot_objs[part_of][0].label = "_"
        # self.plot_objs.pop(-1)

    def cat_to_x(self, x):
        if x not in self.cats:
            self.cats[x] = self.next_cat
            self.next_cat += 1
        return self.cats[x]

    def order(self, given_order):
        tmp = self.plot_objs.copy()
        del self.plot_objs
        self.plot_objs = {}

        for part_of in given_order:
            # find plotobj in old array and keep insertion order = given order
            for plot_obj_name,po in tmp.items():
                if plot_obj_name == part_of:
                    self.plot_objs[plot_obj_name] = po

        if len(given_order) != len(self.plot_objs):
            print("Missing some plot_objs names in given order, this shouldnt happen")

    def sort_cats(self,asint=True):
        t = lambda x: int(x) if asint else x
        order = np.argsort([t(x) for x in self.cats.keys()])
        print(order)
        print(np.array(list(self.cats.keys()),dtype=object))

        self.next_cat = 0
        for cat in list(np.array(list(self.cats.keys()),dtype=object)[order]):
            print(cat)
            self.cats[cat] = self.next_cat
            self.next_cat += 1


    def add(self,xs,ys,label,part_of=None,args={}):
        if not part_of:
            part_of = label
        self.plot_objs[part_of].append(self.PlotObject(xs,ys,label,args))
    def add_cat(self,catxs,ys,label,part_of=None,args={}):
        if not part_of:
            part_of = label
        self.plot_objs[part_of].append(self.PlotObject([self.cat_to_x(catx) for catx in catxs],ys,label,args))

    def do_plot(self,shape="bar",axes=None,**args):
        if not axes:
            axes = plt.figure(figsize=(20,10), dpi=100).add_subplot(1,1,1)
        num_cmp = len(self.plot_objs)
        bar_width=self.total_width/num_cmp
        plotted_xs,plotted_ys = [],[]
        # for cmp_idx,xs,ys,label in zip(range(num_cmp),self.xss,self.yss,self.labels):
        for cmp_idx,(part_of,plot_objs) in enumerate(self.plot_objs.items()):
            for plot_obj in plot_objs:
                plotted_x = np.array(plot_obj.xs) - (num_cmp-1)*0.5*bar_width + cmp_idx*bar_width
                plotted_xs.append(plotted_x)
                if shape=="violin":
                    if len(plot_obj.ys) == 0:
                        continue
                    parts = axes.violinplot(plot_obj.ys,positions=np.unique(plotted_x),widths=bar_width*.9,
                                            showmedians=True)
                    axes.scatter([],[],s=500,alpha=.75,color=parts["bodies"][0].get_facecolor().flatten(),label=plot_obj.label)
                    for pc in parts['bodies']:
                        pc.set_alpha(.75)
                elif shape=="box":
                    if len(plot_obj.ys) == 0:
                        continue
                    color = next(axes._get_lines.prop_cycler)['color']
                    plotted_ys.append(plot_obj.ys)
                    parts = axes.boxplot(plot_obj.ys,positions=np.unique(plotted_x),widths=bar_width*.9,patch_artist=True)
                    # for y,x in zip(parts["medians"],plotted_x):
                    #     y = y.get_ydata()[0]
                    #     plt.text(x-0.01,1,f"{y:.2f}",fontsize=15)
                    plt.setp(parts['boxes'], color=color)
                    plt.setp(parts['medians'], color="black")
                    axes.scatter([],[],s=500,color=color,label=plot_obj.label)
                elif shape=="bar":
                    plotted_ys.append(plot_obj.ys)
                    axes.bar(plotted_x,plot_obj.ys,width=bar_width,label=plot_obj.label,**plot_obj.args,**args)
        if self.cats:
            axes.set_xticks(list(range(1,len(self.cats)+1)))
            axes.set_xticklabels(self.cats.keys())
        return plotted_xs,plotted_ys

if __name__ == '__main__':
    bars = BehindBars()
    bars.add_cat([1,2,0], [15,16,17], "Ducks")
    bars.add_cat([1,2,0], [17,16,15], "Geese")
    bars.add_cat([1,2,3], [20,20,20], "Donkeys")
    bars.add_cat([0,1,2,3], [2,5,5,10], "Alpacas Total", part_of="Alpacas")
    bars.add_cat([0,1,2,3], [1,4,4,9], "Alpacas Part", part_of="Alpacas")

    # bars.sort_cats()

    bars.do_plot(edgecolor="black", linewidth=2,)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=3, mode="expand", borderaxespad=0.)
    plt.show()