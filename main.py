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
    [4000, 7700],
    [7800, 8200],
    [8900, 9800],
    [10150, 10450]
]
###########################################################################
###                         PLOTTING DATA & ROI                         ###
###########################################################################
fig, ax1 = plt.subplots(1, 1, sharex=True)
ax2 = ax1.twinx()
ax2.set_ylabel("Temperaturänderung in K/dt")

t = df.loc[:, "GNSS: PPS Timestamp [s]"]

temp_in = df.loc[:, "Temperature: Board [degC]"].rolling(10).mean()

temp_out = df.loc[:, ("Temperature: Ext LM75 1 [degC]", "Temperature: Ext MS8607 1 [degC]")].mean(axis=1).rolling(10).mean()

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Innen- und Außentemperatur [degC]')
ax1.grid(True)


plt_regions = [ax1.axvspan(x1, x2, alpha=0.2, color='green') for x1, x2 in roi]

plt_regions[0].set_label("Region of interest")

###########################################################################
#### LINEAR REGRESSION ON THE OUTDOOR TEMPTERATURE'S REGION OF INTEREST ###
###########################################################################

linear_T_out = ([None]*len(t.index))

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

    
    final_lst = [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` < {x1}").index) + [region_gradient*(x-mean_t)+mean_T for x in region_t] + [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` > {x2}").index)

    linear_T_out = [i+j if (j is not None and i is not None) else i if i is not None else j for i, j in zip(linear_T_out, final_lst)]


###########################################################################
###            CALCULATE T_in'(t) USING DIFFERENCE QUOTIENT             ###
###########################################################################
window_width = 100

def gradient_window(x):
    in_roi = False
    for i, j in roi:
        if i <= sum(x.index)/window_width <= j:
            in_roi = True
    #if not in_roi: return np.NaN

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
    return gradient if gradient is not None else np.nan

temp_in_t = temp_in.copy()
temp_in_t.index = t.values

temp_in_grad = temp_in_t.rolling(window=window_width, center=True).apply(gradient_window)

df["T_in'"] = list(temp_in_grad)




###########################################################################
####LINEAR REG. ON THE INDOOR TEMPTERATURE-GRADIENT'S REGION OF INTEREST###
###########################################################################
linear_T_in_grad = ([None]*len(t.index))
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
        


    for rt, rT in zip(region_t[start_idx:], region_temp_in_grad[start_idx:]):
        g = (rT-T0)/(rt-t0)
        if g == float("inf") or g == -float("inf") or g is None: continue
        if region_gradient is not None:
            region_gradient += g
            region_gradient /= 2
        else:
            region_gradient = g
    
    final_lst = [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` < {x1}").index) + [region_gradient*(x-mean_t)+mean_T for x in region_t] + [None]*len(df.query(f"`GNSS: PPS Timestamp [s]` > {x2}").index)
    

    linear_T_in_grad = [i+j if (j is not None and i is not None) else i if i is not None else j for i, j in zip(linear_T_in_grad, final_lst)]


###########################################################################
###                   CALCULATE β (T_out(t) = T_in(t))                  ###
###########################################################################

betas = []
for it, iTi, iTo, iTi_grad, iTi_grad_linear, iTo_linear in zip(t, temp_in, temp_out, temp_in_grad, linear_T_in_grad, linear_T_out):
    if  True in [x1 <= it <= x2 for x1, x2 in roi] and abs(iTi - iTo) < 0.01 and it > 0:
        ax2.axvline(it, color="green")
        betas.append(iTi_grad)


beta = sum(betas)/len(betas)
print(f"β = {beta} K/s")


###########################################################################
###           CALCULATING α USING DIFFERENTIAL EQUATION ON ROI          ###
###########################################################################

alphas = []

for (it, iTi, iTo, iTi_grad, iTo_linear, iTi_grad_linear) in zip(t, temp_in, temp_out, temp_in_grad, linear_T_out, linear_T_in_grad):
    if not True in [x1 <= it <= x2 for x1, x2 in roi] or abs(iTo - iTi) < 0.01: 
        alphas.append(np.NaN)
        continue
    alphas.append((iTi_grad - beta) / (iTo - iTi))



alphas_t = alphas
alphas = [i for i in alphas if not np.isnan(i)]
print(min(alphas), max(alphas))


alpha = sum(alphas)/len(alphas)

###########################################################################
###             PREDICT T_in'(t) USING DIFFERENTIAL EQUATION            ###
###########################################################################

pred_T_in_grad = []

for (iTo, iTi) in zip(temp_out, temp_in):
    pred_T_in_grad.append(alpha*(iTo - iTi) + beta)

L = {
    "To": {"label": "Außentemperatur"},
    "Ti": {"label": "Innentemperatur"},
    "lTo": {"c": "red"},
    "lTig": {"c": "blue"},
    "alphas": {"c": "yellow"},
    "Tig": {"label": "Differenzenquotient von T_in(t) ≈ T_in'(t)", "c": "purple"},
    "pTig": {"c": "orange"}
}


# COMMENT/UNCOMMENT FOLLOWING LINES TO SELECT PLOTTING DATA:

ax1.plot(t, temp_out, **L["To"])            # Measured outdoor temperature of the probe                                                      [Default: ENABLED]
ax1.plot(t, temp_in, **L["Ti"])             # Measured indoor (circuit board) temperature of the probe                                       [Default: ENABLED]

# ax1.plot(t, linear_T_out, **L["lTo"])       # Linear Regression on the ROI of the outdoor temperature                                        [Default: ENABLED]
# ax2.plot(t, linear_T_in_grad, **L["lTig"])  # Linear Regression on the ROI of the indoor temperatures gradient                               [Default: ENABLED]

# ax2.plot(t, alphas_t, L["alphas"])          # Calculated α values (differential equation coefficient) for each timestamp                   [Default: DISABLED]

ax2.plot(t, temp_in_grad, **L["Tig"])       # Indoor temperatures gradient (derivative) caclulated over rolling window
ax2.plot(t, pred_T_in_grad, **L["pTig"])    # Indoor temperatires derivative predicted over differental equation using calculated α and β    [Default: ENABLED]


text = f"Mean value: α = {((sum(alphas)/len(alphas))*60*60).__round__(4)} h⁻¹\nMean value: β = {((beta)*60*60).__round__(4)} K/h"

plt.text(0.01, 0.01, text, fontsize=10, transform=plt.gcf().transFigure)

### SHOW PLOT WINDOW ###
handles1, labels1 = ax1.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()
fig.legend(handles1, labels1, loc='upper right')
fig.canvas.manager.set_window_title('Temperature Equilibration Model for Stratospheric Probe')
plt.show()
########################
