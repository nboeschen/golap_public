from matplotlib import pyplot as plt
from seaborn import color_palette
from behindbars import BehindBars

plt.rcParams.update({'font.size': 30,"pdf.fonttype": 42,"ps.fonttype": 42})
fig = plt.figure(figsize=(20,10), dpi=100)
axes = fig.add_subplot(1, 1, 1)
axes.set_prop_cycle('color', color_palette("colorblind"))

bars = BehindBars()
bars.add_cat(["SSD Read BW"], [19], "Geese")
bars.add_cat(["CPU Memory BW"], [80], "Geese")
bars.add_cat(["GPU Memory BW"], [500], "Geese")
# bars.add_cat([0,1,2,3], [2,5,5,10], "Alpacas Total", part_of="Alpacas")
# bars.add_cat([0,1,2,3], [1,4,4,9], "Alpacas Part", part_of="Alpacas")

# bars.sort_cats()

bars.do_plot(axes=axes,edgecolor="black", linewidth=2,)
# plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=3, mode="expand", borderaxespad=0.)
plt.grid()
plt.tight_layout()
plt.show()