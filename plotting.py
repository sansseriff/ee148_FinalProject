import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter



#ax = df['myvar'].plot(kind='bar')
#ax.yaxis.set_major_formatter(mtick.PercentFormatter())

matplotlib.rcParams['axes.linewidth'] = 2


EPEs = [.064,.084, .106,.121,.154]
sizes = [100, 30, 10, 3, 1]
fig1, ax1 = plt.subplots()
#plt.loglog(sizes,EPEs, 'o-')
plt.plot(sizes,EPEs, 'o-')

ax1.set_xscale('log')
ax1.set_yscale('log')

y = []
start = .1
for tick in range(-4,7):
    y.append(round(start + tick*.01,3))

print(y)

#ax1.minorticks_off()
#ax1.set_yticks([.06,.066,.073,.080,.088, .097, .106, .117, .129, .141,.156])
#ax1.set_yticks([])
print(y)
ax1.set_yticks(y)
#ax1.yaxis.set_major_formatter(NullFormatter())
#ax1.yaxis.set_minor_formatter(NullFormatter())
ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.xaxis.set_major_formatter(mtick.PercentFormatter())


#plt.xscale('log', subsx=[2, 3, 4, 5, 6, 7, 8, 9])

plt.xlabel('Train Dataset Size', fontsize=14)
plt.ylabel('End Point Error', fontsize=14)
plt.title('End point Error vs Training Set Size', fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.xaxis.set_tick_params(width=2)
ax1.yaxis.set_tick_params(width=2)
plt.grid(True, which="both", axis = "x", ls="-")
plt.grid(True, which="major", axis = "y", ls="-")
ax1.tick_params(which='both', width=1.5, length = 6)
ax1.yaxis.set_major_formatter(ScalarFormatter())
#ax1.ticklabel_format(style='plain')
#ax1.tick_params(which='major', length=7)
#ax1.tick_params(which='minor', length=4, color='r')
#ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
