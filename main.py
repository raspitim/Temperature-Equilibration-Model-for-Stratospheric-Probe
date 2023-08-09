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
fig, ax1 = plt.subplots(1, 1, sharex=True)

t = df.loc[:, "GNSS: PPS Timestamp [s]"]

temp_in = df.loc[:, "Temperature: Board [degC]"].rolling(10).mean()

temp_out = df.loc[:, ("Temperature: Ext LM75 1 [degC]", "Temperature: Ext MS8607 1 [degC]")].mean(axis=1).rolling(10).mean()

plt_temp_out = ax1.plot(t, temp_out, label="Außentemperatur")
plt_temp_in = ax1.plot(t, temp_in, label="Innentemperatur")

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Innen- und Außentemperatur [degC]')
ax1.grid(True)


plt_regions = [ax1.axvspan(x1, x2, alpha=0.2, color='green') for x1, x2 in roi]

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

    ax1.plot(t, [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` < {x1}").index) + [region_gradient*(x-mean_t)+mean_T for x in region_t] + [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` > {x2}").index), c="red")



###########################################################################
###          CALCULARE T_in'(t) = 0 USING DIFFERENCE QUOTIENT           ###
###########################################################################
window_width = 100

def gradient_window(x):
    in_roi = False
    for i, j in roi:
        if i <= sum(x.index)/window_width <= j:
            in_roi = True
    if not in_roi: return np.NaN

    gradient = None

    t0, T0 = x.index[0], x.values[0]

    for rt, rT in zip(x.index[1:], x.values[1:]):
        g = (rT-T0)/(rt-t0)
        if g == float("inf") or g == -float("inf") or g is None or np.isnan(g): continue

        if gradient is not None:
            gradient += g
            gradient /= 2
        else:
            gradient = g

    return gradient

temp_in_t = temp_in.copy()
temp_in_t.index = t.values

temp_in_grad = temp_in_t.rolling(window=window_width, center=True).apply(gradient_window)

df["T_in'"] = list(temp_in_grad)

ax2 = ax1.twinx()
ax2.plot(t, temp_in_grad, color="purple", label="Differenzenquotient von T_in(t) ≈ T_in'(t)")
ax2.set_ylabel("Temperaturänderung in K/dt")


###########################################################################
####LINEAR REG. ON THE INDOOR TEMPTERATURE-GRADIENT'S REGION OF INTEREST###
###########################################################################
for x1, x2 in roi:

    region_data = df.query(f"{x1} <= `GNSS: PPS Timestamp [s]` <= {x2}")

    region_temp_in_grad = region_data.loc[:, "T_in'"]
    region_t = region_data.loc[:, "GNSS: PPS Timestamp [s]"]

    mean_T = region_temp_in_grad.sum() / len(region_temp_in_grad)
    mean_t = (x1+x2)/2

    region_gradient = None

    for i in range(len(region_t)):
        t0 = region_t.iloc[i]
        T0 = region_temp_in_grad.iloc[i]

        if not np.isnan(t0) and not np.isnan(T0):
            start_idx = i +1
            break
        

    print(t0, T0)

    for rt, rT in zip(region_t[start_idx:], region_temp_in_grad[start_idx:]):
        g = (rT-T0)/(rt-t0)
        if g == float("inf") or g == -float("inf") or g is None: continue
        if region_gradient is not None:
            region_gradient += g
            region_gradient /= 2
        else:
            region_gradient = g
    
    ax2.plot(t, [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` < {x1}").index) + [region_gradient*(x-mean_t)+mean_T for x in region_t] + [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` > {x2}").index), c="blue")


###########################################################################
###                     CALCULATE T_out(t) = T_in(t)                    ###
###########################################################################


for it, iTi, iTo in zip(t, temp_in, temp_out):
    if iTi - iTo == 0 and it > 0:
        ax2.axvline(it, color="green")
        print(it)




### SHOW PLOT WINDOW ###
handles1, labels1 = ax1.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1, labels1, loc='upper right')
fig.canvas.manager.set_window_title('Temperature Equilibration Model for Stratospheric Probe')
plt.show()
########################
