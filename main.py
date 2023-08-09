from load import df
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

###########################################################################
###                       SET REGIONS OF INTEREST                       ###
###########################################################################
roi = [
    [600, 1000],
    [1200, 2800],
    [4000, 7500],
    [7800, 8200],
    [8900, 9800],
    [10150, 10450]
]
###########################################################################
###                         PLOTTING DATA & ROI                         ###
###########################################################################
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)

t = df.loc[:, "GNSS: PPS Timestamp [s]"]

temp_in = df.loc[:, "Temperature: Board [degC]"].rolling(10).mean()

temp_out = df.loc[:, ("Temperature: Ext LM75 1 [degC]", "Temperature: Ext MS8607 1 [degC]")].mean(axis=1).rolling(10).mean()

plt_temp_out = axs.plot(t, temp_out, label="Außentemperatur")
plt_temp_in = axs.plot(t, temp_in, label="Innentemperatur")

axs.set_xlabel('Time [s]')
axs.set_ylabel('Innen- und Außentemperatur [degC]')
axs.grid(True)


plt_regions = [axs.axvspan(x1, x2, alpha=0.2, color='green') for x1, x2 in roi]

plt_regions[0].set_label("Region of interest")

###########################################################################
#### LINEAR REGRESSION ON THE OUTDOOR TEMPTERATURE'S REGION OF INTEREST ###
###########################################################################
for x1, x2 in roi:

    region_data = df.query(f"{x1} <= `GNSS: PPS Timestamp [s]` <= {x2}")

    region_temp_out = region_data.loc[:, ("Temperature: Ext LM75 1 [degC]", "Temperature: Ext MS8607 1 [degC]")].mean(axis=1)
    region_t = region_data.loc[:, "GNSS: PPS Timestamp [s]"]

    mean_T = region_temp_out.sum() / len(region_temp_out)
    mean_t = (x1+x2)/2

    region_gradient = None

    t0 = region_t.iloc[0]
    T0 = region_temp_out.iloc[0]

    for rt, rT in zip(region_t[1:], region_temp_out[1:]):
        g = (rT-T0)/(rt-t0)
        if region_gradient is not None:
            region_gradient += g
            region_gradient /= 2
        else:
            region_gradient = g

    axs.plot(t, [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` < {x1}").index) + [region_gradient*(x-mean_t)+mean_T for x in region_t] + [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` > {x2}").index), c="red")
###########################################################################
###    DETERMINE SADDLE POINTS OF INDOOR TEMPERATURE [T_out'(t) = 0]    ###
###########################################################################
window_width = 20


df[]


for i in df.index[:-window_width]:
    ...





###SHOW PLOT WINDOW###
plt.legend()       ###
plt.show()         ###
######################
