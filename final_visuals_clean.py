import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime, timedelta


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FIG 2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

# time series graphs
utility_water_ts = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_utility_water_use_ts.xlsx", sheet_name="all_s", skiprows=[47])
utility_rev_ts_nos = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_utility_revenue_ts.xlsx", sheet_name="all_no_s", skiprows=[47])
utility_rev_ts = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_utility_revenue_ts.xlsx", sheet_name="all_s", skiprows=[47])

HH_water_ts = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_HH_water_use_ts.xlsx", sheet_name="all_s", skiprows=[1])
HH_bills_ts_nos = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_HH_bills_ts.xlsx", sheet_name="all_no_s", skiprows=[1,47])
HH_bills_ts = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_HH_bills_ts.xlsx", sheet_name="all_s", skiprows=[1])

dates = pd.date_range('2012-02-01','2015-11-01', freq='MS')

fig, axs = plt.subplots(3,2)

axs[0,0].plot(dates, utility_water_ts['sum']/10000, color='#9f86c0', linewidth=3)
axs[0,0].plot(dates, (list(utility_water_ts['sum'][:11]/10000)+(list(utility_water_ts['sum'][:11]/10000)+[utility_water_ts['sum'][45]/10000])*2)+list(utility_water_ts['sum'][35:]/10000), '--', color='#e4c1f9', linewidth=3)
axs[0,0].set_ylim(0, 35)
axs[0,0].set_ylabel('(MMcf/mon)', fontsize=12)
axs[0,0].set_xticks(pd.to_datetime(dates[4::12]).strftime("%Y-%m"))
axs[0,0].set_xticklabels([])
axs[0,0].set_title("Utility Scale\nWater Demand", fontsize=12)

values = (list(utility_rev_ts_nos['sum'][:11]/1000000)+(list(utility_rev_ts_nos['sum'][:11]/1000000)+[utility_rev_ts_nos['sum'][45]/1000000])*2)+list(utility_rev_ts_nos['sum'][35:]/1000000)
axs[1,0].plot(dates, utility_rev_ts_nos['sum']/1000000, color='#9f86c0', linewidth=3)
axs[1,0].plot(dates, values, '--', color='#e4c1f9', linewidth=3)
axs[1,0].set_ylabel('($M/mon)', fontsize=12)
axs[1,0].set_ylim(0, 4)
axs[1,0].set_yticks([0,1, 2.0, 3.0, 4])
axs[1,0].set_xticks(pd.to_datetime(dates[4::12]).strftime("%Y-%m"))
axs[1,0].set_xticklabels([])
axs[1,0].fill_between(dates, utility_rev_ts_nos['sum']/1000000, values, where=values >= utility_rev_ts_nos['sum']/1000000, color='#EDE3FF')
axs[1,0].set_title('Revenue, No Surcharge')

axs[2,0].plot(dates, utility_rev_ts['sum']/1000000, label='Drought', color='#9f86c0', linewidth=3)
axs[2,0].plot(dates, (list(utility_rev_ts['sum'][:11]/1000000)+(list(utility_rev_ts['sum'][:11]/1000000)+[utility_rev_ts['sum'][45]/1000000])*2)+list(utility_rev_ts['sum'][35:]/1000000), '--', label='Non-Drought', color='#e4c1f9', linewidth=3)
axs[2,0].set_ylabel('($M/mon)', fontsize=12)
axs[2,0].set_ylim(0, 4)
axs[2,0].set_yticks([0,1, 2.0, 3.0, 4])
axs[2,0].set_xticks(pd.to_datetime(dates[4::12]).strftime("%Y-%m"))
myFmt = mdates.DateFormatter('%Y')
axs[2,0].xaxis.set_major_formatter(myFmt)
patch = mpatches.Patch(color='#EDE3FF', label='Revenue\nReduction')
axs[1,0].legend(handles=[patch], loc='lower center', bbox_to_anchor=(0.45, -2), fancybox=True, fontsize=10, frameon=False)
axs[2,0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=1, fontsize=10, frameon=False)
axs[2,0].set_title('Revenue, Surcharge')

axs[0,1].plot(dates, HH_water_ts[15], color= '#00b4d8', linewidth=3)
axs[0,1].plot(dates, (list(HH_water_ts[15][:11])+(list(HH_water_ts[15][:11])+[HH_water_ts[15][45]])*2)+list(HH_water_ts[15][35:]), '--', color='#90e0ef', linewidth=3)
axs[0,1].plot(dates, HH_water_ts[0], color='#d90429', linewidth=3)
axs[0,1].plot(dates, (list(HH_water_ts[0][:11])+(list(HH_water_ts[0][:11])+[HH_water_ts[0][45]])*2)+list(HH_water_ts[0][35:]), '--', color='#ffb5a7', linewidth=3)
axs[0,1].set_ylabel('(Ccf/mon)', fontsize=12)
axs[0,1].set_ylim(0, 17.5)
axs[0,1].yaxis.set_label_position("right")
axs[0,1].yaxis.tick_right()
axs[0,1].set_xticks(pd.to_datetime(dates[4::12]).strftime("%Y-%m"))
axs[0,1].set_xticklabels([])
axs[0,1].set_title("Household Scale\nWater Use", fontsize=12)

axs[1,1].plot(dates[:45], HH_bills_ts_nos[15][:45], color= '#00b4d8', linewidth=3)
axs[1,1].plot(dates[:45], (list(HH_bills_ts_nos[15][:11])+(list(HH_bills_ts_nos[15][:11])+[HH_bills_ts_nos[15][44]])*2)+list(HH_bills_ts_nos[15][35:45]), '--', color='#90e0ef', linewidth=3)
axs[1,1].plot(dates[:45], HH_bills_ts_nos[0][:45], color='#d90429', linewidth=3)
axs[1,1].plot(dates[:45], (list(HH_bills_ts_nos[0][:11])+(list(HH_bills_ts_nos[0][:11])+[HH_bills_ts_nos[0][44]])*2)+list(HH_bills_ts_nos[0][35:45]), '--', color='#ffb5a7', linewidth=3)
axs[1,1].set_ylabel('($/mon)', fontsize=12)
axs[1,1].set_ylim(0, 175)
axs[1,1].yaxis.set_label_position("right")
axs[1,1].yaxis.tick_right()
axs[1,1].set_xticks(pd.to_datetime(dates[4::12]).strftime("%Y-%m"))
axs[1,1].set_xticklabels([])
axs[1,1].set_title("Bills, No Surcharge", fontsize=12)

axs[2,1].plot(dates, HH_bills_ts[15], label="High-Income:\nDrought", color= '#00b4d8', linewidth=3)
axs[2,1].plot(dates, (list(HH_bills_ts[15][:11])+(list(HH_bills_ts[15][:11])+[HH_bills_ts[15][45]])*2)+list(HH_bills_ts[15][35:]), '--', label="High-Income:\nNon-Drought", color='#90e0ef', linewidth=3)
axs[2,1].plot(dates, HH_bills_ts[0], label="Low-Income:\nDrought", color='#d90429', linewidth=3)
axs[2,1].plot(dates, (list(HH_bills_ts[0][:11])+(list(HH_bills_ts[0][:11])+[HH_bills_ts[0][45]])*2)+list(HH_bills_ts[0][35:]), '--', label="Low-Income:\nNon-Drought", color='#ffb5a7', linewidth=3)
axs[2,1].set_ylabel('($/mon)', fontsize=12)
axs[2,1].yaxis.set_label_position("right")
axs[2,1].set_ylim(0, 175)
axs[2,1].yaxis.tick_right()
axs[2,1].set_xticks(pd.to_datetime(dates[4::12]).strftime("%Y-%m"))
myFmt = mdates.DateFormatter('%Y')
axs[2,1].xaxis.set_major_formatter(myFmt)
axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2, fontsize=10, frameon=False)
axs[2,1].set_title("Bills, Surcharge", fontsize=12)

axs[0,0].text(0.01,0.9, "(a)", transform=axs[0,0].transAxes, fontsize=12)
axs[0,1].text(0.01,0.9, "(b)", transform=axs[0,1].transAxes, fontsize=12)
axs[1,0].text(0.01,0.9, "(c)", transform=axs[1,0].transAxes, fontsize=12)
axs[1,1].text(0.01,0.9, "(d)", transform=axs[1,1].transAxes, fontsize=12)
axs[2,0].text(0.01,0.9, "(e)", transform=axs[2,0].transAxes, fontsize=12)
axs[2,1].text(0.01,0.9, "(f)", transform=axs[2,1].transAxes, fontsize=12)

fig.set_size_inches(5.9, 8)
fig.tight_layout(pad = 0.2)

plt.subplots_adjust(hspace=.3)
plt.savefig('time_series.jpg', bbox_inches='tight', dpi=500)
plt.show()



'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FIG 5~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# equity vs equality graph
LH_bills_min = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Minor_Drought/IBR/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_hist = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_sev = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Severe_Drought/IBR/final_summary.xlsx", sheet_name="drought_annual_bills_LH")

# final Bills Graph
#revenue by tier, low and high income
labels = ['Warning', 'Historical', 'Critical', 'Warning ', 'Historical ', 'Critical ']
fixed_crev = [LH_bills_min[0][0], LH_bills_hist[0][0], LH_bills_sev[0][0],
              LH_bills_min[0][1], LH_bills_hist[0][1], LH_bills_sev[0][1]]
tier_1_crev =[LH_bills_min[1][0], LH_bills_hist[1][0], LH_bills_sev[1][0],
              LH_bills_min[1][1], LH_bills_hist[1][1], LH_bills_sev[1][1]]
tier_2_crev = [LH_bills_min[2][0], LH_bills_hist[2][0], LH_bills_sev[2][0],
              LH_bills_min[2][1], LH_bills_hist[2][1], LH_bills_sev[2][1]]
tier_3_crev = [LH_bills_min[3][0], LH_bills_hist[3][0], LH_bills_sev[3][0],
              LH_bills_min[3][1], LH_bills_hist[3][1], LH_bills_sev[3][1]]
tier_4_crev = [LH_bills_min[4][0], LH_bills_hist[4][0], LH_bills_sev[4][0],
              LH_bills_min[4][1], LH_bills_hist[4][1], LH_bills_sev[4][1]]
fixed_lh = [LH_bills_min[5][0], LH_bills_hist[5][0], LH_bills_sev[5][0],
              LH_bills_min[5][1], LH_bills_hist[5][1], LH_bills_sev[5][1]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

fig, axs = plt.subplots(2)

axs[0].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703')
axs[0].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6')
axs[0].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC')
axs[0].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6')
axs[0].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047')
axs[0].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500')

axs[0].set_title('Water Bills', fontsize=12)
axs[0].set_ylabel('($/yr)', fontsize=12)
axs[0].set_xlabel('Low Income                            High Income', fontsize=12)
axs[0].set_ylim([0, 1400])


# final AR Graph
#revenue by tier, low and high income affordability ratio
fixed_crev = [i/20000*100 for i in fixed_crev[:3]]+[i/137500*100 for i in fixed_crev[3:]]
tier_1_crev = [i/20000*100 for i in tier_1_crev[:3]]+[i/137500*100 for i in tier_1_crev[3:]]
tier_2_crev = [i/20000*100 for i in tier_2_crev[:3]]+[i/137500*100 for i in tier_2_crev[3:]]
tier_3_crev = [i/20000*100 for i in tier_3_crev[:3]]+[i/137500*100 for i in tier_3_crev[3:]]
tier_4_crev = [i/20000*100 for i in tier_4_crev[:3]]+[i/137500*100 for i in tier_4_crev[3:]]
fixed_lh = [i/20000*100 for i in fixed_lh[:3]]+[i/137500*100 for i in fixed_lh[3:]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

axs[1].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703')
axs[1].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6')
axs[1].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC')
axs[1].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6')
axs[1].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047')
axs[1].bar(labels, fixed_lh, width, bottom=t_f1234, label='Surcharge', color='#FB8500')

axs[1].set_title('Affordability Ratio', fontsize=12)
axs[1].set_ylabel('(% Annual Income)', fontsize=12)
axs[1].set_xlabel('Low Income                            High Income', fontsize=12)
axs[1].set_ylim([0, 5.0])
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, fontsize=10, frameon=False, ncol=3)

axs[0].text(0.02,0.9, "(a)", transform=axs[0].transAxes, fontsize=12)
axs[1].text(0.02,0.9, "(b)", transform=axs[1].transAxes, fontsize=12)

fig.set_size_inches(5.9, 7.5)
fig.tight_layout(pad=0.5)

plt.subplots_adjust(hspace=.35)
plt.savefig('equity_equality3.jpg', bbox_inches='tight', dpi=500)
plt.show()



'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FIG 3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

# barplot overview graph
water_pre_drought_utility = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="pre_utility_water_use")
water_drought_utility = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_utility_water_use")
rev_pre_drought_utility = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="pre_utility_rev")
rev_drought_utility = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_utility_rev")
LH_water_pre = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="pre_annual_water_use_LH")
LH_water_fixed = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="drought_annual_water_use_LH")
LH_bills_pre = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="pre_annual_bills_LH")
LH_bills_vol = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_fixed = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")


# before after drought water use by tiers graph
labels = ['Base', 'Drought']
tier_1 = [water_pre_drought_utility[0][1]/10000, water_drought_utility[1][0]/10000]
tier_2 = [water_pre_drought_utility[0][2]/10000, water_drought_utility[2][0]/10000]
tier_3 = [water_pre_drought_utility[0][3]/10000, water_drought_utility[3][0]/10000]
tier_4 = [water_pre_drought_utility[0][4]/10000, water_drought_utility[4][0]/10000]
width = 0.8  # the width of the bars: can also be len(x) sequence

t_12 = [(tier_1[i]+tier_2[i]) for i in range(len(tier_1))]
t_123 = [(tier_1[i]+tier_2[i]+tier_3[i]) for i in range(len(tier_1))]

fig, axs = plt.subplots(2,2, gridspec_kw={'width_ratios': [2, 4]})

axs[0,0].bar(labels, tier_1, width, label='Tier 1', color = '#8ECAE6', align='center')
axs[0,0].bar(labels, tier_2, width, bottom=tier_1, label='Tier 2', color='#219EBC', align='center')
axs[0,0].bar(labels, tier_3, width, bottom=t_12, label='Tier 3', color='#0077B6', align='center')
axs[0,0].bar(labels, tier_4, width, bottom=t_123, label='Tier 4', color='#023047', align='center')

axs[0,0].set_ylim(0,350)
axs[0,0].set_ylabel('Water Consumption\n(MMcf/yr)', fontsize=12)
axs[0,0].set_title('Utility Scale\n', fontsize=12)
axs[0,0].text(0.01,0.92, "(a)", transform=axs[0,0].transAxes, fontsize=12)

# before after drought revenue by tiers graph
labels = ['Base', 'Drought']
fixed = [rev_pre_drought_utility[0][0]/1000000, rev_drought_utility[0][0]/1000000]
tier_1 = [rev_pre_drought_utility[0][1]/1000000, rev_drought_utility[1][0]/1000000]
tier_2 = [rev_pre_drought_utility[0][2]/1000000, rev_drought_utility[2][0]/1000000]
tier_3 = [rev_pre_drought_utility[0][3]/1000000, rev_drought_utility[3][0]/1000000]
tier_4 = [rev_pre_drought_utility[0][4]/1000000, rev_drought_utility[4][0]/1000000]
surcharge = [0, rev_drought_utility[5][0]/1000000]
width = 0.8  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1[i]+fixed[i]) for i in range(len(tier_1))]
t_f12 = [(tier_1[i]+tier_2[i]+fixed[i]) for i in range(len(tier_1))]
t_f123 = [(tier_1[i]+tier_2[i]+tier_3[i]+fixed[i]) for i in range(len(tier_1))]
t_f1234 = [(tier_1[i]+tier_2[i]+tier_3[i]+tier_4[i]+fixed[i]) for i in range(len(tier_1))]

axs[1,0].bar(labels, fixed, width, label='Fixed', color='#FFB703', align='center')
axs[1,0].bar(labels, tier_1, width, bottom=fixed, label='Tier 1', color = '#8ECAE6', align='center')
axs[1,0].bar(labels, tier_2, width, bottom=t_fi, label='Tier 2', color='#219EBC', align='center')
axs[1,0].bar(labels, tier_3, width, bottom=t_f12, label='Tier 3', color='#0077B6', align='center')
axs[1,0].bar(labels, tier_4, width, bottom=t_f123, label='Tier 4', color='#023047', align='center')
axs[1,0].bar(labels, surcharge, width, bottom=t_f1234, label='Surcharge', color='#FB8500', align='center')

axs[1,0].set_ylabel('Revenue and Costs\n($M/yr)', fontsize=12)
axs[1,0].set_ylim([0,43])
axs[1,0].text(0.01,0.92, "(c)", transform=axs[1,0].transAxes, fontsize=12)

# water use during drought by tier, low and high income
labels = ['LI', 'HI ', 'LI ', 'HI']
tier_1_lh =[LH_water_pre[1][0], LH_water_pre[1][1],
              LH_water_fixed[1][0], LH_water_fixed[1][1]]
tier_2_lh = [LH_water_pre[2][0], LH_water_pre[2][1],
              LH_water_fixed[2][0], LH_water_fixed[2][1]]
tier_3_lh = [LH_water_pre[3][0], LH_water_pre[3][1],
              LH_water_fixed[3][0], LH_water_fixed[3][1]]
tier_4_lh = [LH_water_pre[4][0], LH_water_pre[4][1],
              LH_water_fixed[4][0], LH_water_fixed[4][1]]
width = 0.8  # the width of the bars: can also be len(x) sequence

t_12 = [tier_1_lh[i]+tier_2_lh[i] for i in range(len(tier_1_lh))]
t_123 = [tier_1_lh[i]+tier_2_lh[i]+tier_3_lh[i] for i in range(len(tier_1_lh))]

axs[0,1].bar(labels, tier_1_lh, width, label='Tier 1', color = '#8ECAE6', align='center')#BBBBBB
axs[0,1].bar(labels, tier_2_lh, width, bottom=tier_1_lh, label='Tier 2', color='#219EBC', align='center') #7B7B7B
axs[0,1].bar(labels, tier_3_lh, width, bottom=t_12, label='Tier 3', color='#0077B6', align='center') #5A5A5A
axs[0,1].bar(labels, tier_4_lh, width, bottom=t_123, label='Tier 4', color='#023047', align='center') #242424

axs[0,1].set_title('Household Scale\n')
axs[0,1].set_ylabel('(Ccf/yr)', fontsize=12)
axs[0,1].yaxis.set_label_position("right")
axs[0,1].yaxis.tick_right()
axs[0,1].set_ylim(([0, 120]))
axs[0,1].text(0.01,0.92, "(b)", transform=axs[0,1].transAxes, fontsize=12)


# revenue by tier, low and high income
labels = ['LI', 'HI ', 'LI ', 'HI']
fixed_lh = [LH_bills_pre[0][0], LH_bills_pre[0][1],
              LH_bills_fixed[0][0], LH_bills_fixed[0][1]]
tier_1_lh =[LH_bills_pre[1][0], LH_bills_pre[1][1],
              LH_bills_fixed[1][0], LH_bills_fixed[1][1]]
tier_2_lh = [LH_bills_pre[2][0], LH_bills_pre[2][1],
              LH_bills_fixed[2][0], LH_bills_fixed[2][1]]
tier_3_lh = [LH_bills_pre[3][0], LH_bills_pre[3][1],
              LH_bills_fixed[3][0], LH_bills_fixed[3][1]]
tier_4_lh = [LH_bills_pre[4][0], LH_bills_pre[4][1],
              LH_bills_fixed[4][0], LH_bills_fixed[4][1]]
surcharge_lh = [0, 0,
              LH_bills_fixed[5][0], LH_bills_fixed[5][1]]
width = 0.8  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_lh[i]+fixed_lh[i]) for i in range(len(tier_1_lh))]
t_f12 = [(tier_1_lh[i]+tier_2_lh[i]+fixed_lh[i]) for i in range(len(tier_1_lh))]
t_f123 = [(tier_1_lh[i]+tier_2_lh[i]+tier_3_lh[i]+fixed_lh[i]) for i in range(len(tier_1_lh))]
t_f1234 = [(tier_1_lh[i]+tier_2_lh[i]+tier_3_lh[i]+tier_4_lh[i]+fixed_lh[i]) for i in range(len(tier_1_lh))]

axs[1,1].bar(labels, fixed_lh, width, label='Fixed', color='#FFB703')
axs[1,1].bar(labels, tier_1_lh, width, bottom=fixed_lh, label='Tier 1', color = '#8ECAE6')
axs[1,1].bar(labels, tier_2_lh, width, bottom=t_fi, label='Tier 2', color='#219EBC')
axs[1,1].bar(labels, tier_3_lh, width, bottom=t_f12, label='Tier 3', color='#0077B6')
axs[1,1].bar(labels, tier_4_lh, width, bottom=t_f123, label='Tier 4', color='#023047')
axs[1,1].bar(labels, surcharge_lh, width, bottom=t_f1234, label='Surcharge', color='#FB8500')

axs[1,1].set_ylabel('($/yr)', fontsize=12)
axs[1,1].yaxis.set_label_position("right")
axs[1,1].yaxis.tick_right()
axs[1,1].set_xlabel('Base             Drought', fontsize=12)
axs[1,1].legend(loc='upper center', frameon=False, bbox_to_anchor=(0.2, -0.25), fancybox=True, ncol=3, fontsize=10)
axs[1,1].set_ylim(([0, 1400]))
axs[1,1].text(0.01,0.92, "(d)", transform=axs[1,1].transAxes, fontsize=12)

fig.set_size_inches(5.9, 7.6)

plt.subplots_adjust(hspace=.2)
plt.savefig('barplot_overview.jpg', bbox_inches='tight', dpi=500)
plt.show()

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FIG 4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# load data
LH_bills_pre_ibr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="pre_annual_bills_LH")
LH_bills_pre_dbr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/DBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="pre_annual_bills_LH")
LH_bills_fixed_ibr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_fixed_dbr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/DBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_pre = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="pre_annual_bills_LH")
LH_bills_vol = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_fixed = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")

#Base Rate
#Low Income
labels = ['Base ', 'Drought ', ' Base ', ' Drought ']
fixed_crev = [LH_bills_pre_ibr[0][0], LH_bills_fixed_ibr[0][0], LH_bills_pre_dbr[0][0], LH_bills_fixed_dbr[0][0]]
tier_1_crev = [LH_bills_pre_ibr[1][0], LH_bills_fixed_ibr[1][0], LH_bills_pre_dbr[1][0], LH_bills_fixed_dbr[1][0]]
tier_2_crev = [LH_bills_pre_ibr[2][0], LH_bills_fixed_ibr[2][0], LH_bills_pre_dbr[2][0], LH_bills_fixed_dbr[2][0]]
tier_3_crev = [LH_bills_pre_ibr[3][0], LH_bills_fixed_ibr[3][0], LH_bills_pre_dbr[3][0], LH_bills_fixed_dbr[3][0]]
tier_4_crev = [LH_bills_pre_ibr[4][0], LH_bills_fixed_ibr[4][0], LH_bills_pre_dbr[4][0], LH_bills_fixed_dbr[4][0]]
fixed_lh = [0, LH_bills_fixed_ibr[5][0], 0, LH_bills_fixed_dbr[5][0]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

fig, axs = plt.subplots(2,2, gridspec_kw={'width_ratios': [4, 3]})

axs[0,0].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["//", "//",  "", ""])

axs[0,0].set_title('Base Rate Comparison\n', fontsize=12)
#axs[0,0].set_xlabel('IBR           DBR', fontsize=12)
axs[0,0].set_xticklabels([])
axs[0,0].set_ylim([0, 1550])
axs[0,0].text(0.01,0.92, "(a)", transform=axs[0,0].transAxes, fontsize=12)
axs[0,0].set_ylabel("Low-Income Bills\n($/yr)", fontsize=12)

# Surcharge
# Low Income
labels = ['Base', 'Fixed', 'Volumetric']
fixed_crev = [LH_bills_pre[0][0], LH_bills_fixed[0][0], LH_bills_vol[0][0]]
tier_1_crev =[LH_bills_pre[1][0], LH_bills_fixed[1][0], LH_bills_vol[1][0]]
tier_2_crev = [LH_bills_pre[2][0], LH_bills_fixed[2][0], LH_bills_vol[2][0]]
tier_3_crev = [LH_bills_pre[3][0], LH_bills_fixed[3][0], LH_bills_vol[3][0]]
tier_4_crev = [LH_bills_pre[4][0], LH_bills_fixed[4][0], LH_bills_vol[4][0]]
fixed_lh = [0, LH_bills_fixed[5][0], LH_bills_vol[5][0]]
tier_1_lh = [0, 0, LH_bills_vol[6][0]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12345 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_lh[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

axs[0,1].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["", "", "//"])
axs[0,1].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_1_lh, width, bottom=t_f12345, label='Volumetric Surcharge', color='#E34234', hatch = ["", "", "//"])

axs[0,1].set_title('Surcharge Comparison\n', fontsize=12)
axs[0,1].set_ylim([0, 1550])
axs[0,1].set_yticklabels([])
axs[0,1].set_xticklabels([])
axs[0,1].text(0.01,0.92, "(b)", transform=axs[0,1].transAxes, fontsize=12)

#Base Rate
#High Income
labels = ['Base', 'Drought ', ' Base', ' Drought']
fixed_crev = [LH_bills_pre_ibr[0][1], LH_bills_fixed_ibr[0][1], LH_bills_pre_dbr[0][1], LH_bills_fixed_dbr[0][1]]
tier_1_crev = [LH_bills_pre_ibr[1][1], LH_bills_fixed_ibr[1][1], LH_bills_pre_dbr[1][1], LH_bills_fixed_dbr[1][1]]
tier_2_crev = [LH_bills_pre_ibr[2][1], LH_bills_fixed_ibr[2][1], LH_bills_pre_dbr[2][1], LH_bills_fixed_dbr[2][1]]
tier_3_crev = [LH_bills_pre_ibr[3][1], LH_bills_fixed_ibr[3][1], LH_bills_pre_dbr[3][1], LH_bills_fixed_dbr[3][1]]
tier_4_crev = [LH_bills_pre_ibr[4][1], LH_bills_fixed_ibr[4][1], LH_bills_pre_dbr[4][1], LH_bills_fixed_dbr[4][1]]
fixed_lh = [0, LH_bills_fixed_ibr[5][1], 0, LH_bills_fixed_dbr[5][1]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

axs[1,0].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["//", "//",  "", ""])

axs[1,0].set_xlabel('IBR                    DBR', fontsize=11)
axs[1,0].set_ylim([0, 1550])
axs[1,0].text(0.01,0.92, "(c)", transform=axs[1,0].transAxes, fontsize=12)
axs[1,0].set_ylabel("High-Income Bills\n($/yr)", fontsize=12)


# Surcharge
# High Income
labels = ['Base', 'Fixed', 'Volumetric']
fixed_crev = [LH_bills_pre[0][1], LH_bills_fixed[0][1], LH_bills_vol[0][1]]
tier_1_crev =[LH_bills_pre[1][1], LH_bills_fixed[1][1], LH_bills_vol[1][1]]
tier_2_crev = [LH_bills_pre[2][1], LH_bills_fixed[2][1], LH_bills_vol[2][1]]
tier_3_crev = [LH_bills_pre[3][1], LH_bills_fixed[3][1], LH_bills_vol[3][1]]
tier_4_crev = [LH_bills_pre[4][1], LH_bills_fixed[4][1], LH_bills_vol[4][1]]
fixed_lh = [0, LH_bills_fixed[5][1], LH_bills_vol[5][1]]
tier_1_lh = [0, 0, LH_bills_vol[6][1]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12345 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_lh[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

axs[1,1].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["", "", "//"], alpha=0.99)
axs[1,1].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["", "", "//"])
axs[1,1].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_1_lh, width, bottom=t_f12345, label='Volumetric Surcharge', color='#E34234', hatch = ["", "", "//"])

axs[1,1].set_ylim([0, 1550])
axs[1,1].set_yticklabels([])
axs[1,1].text(0.01,0.92, "(d)", transform=axs[1,1].transAxes, fontsize=12)
handles, labels = axs[1,1].get_legend_handles_labels()
circ1 = mpatches.Patch(hatch='/////', label='Not Commonly Adopted\nUnder Prop 218', alpha=0.0)
circ2 = mpatches.Patch(alpha=0, hatch='', label='Commonly Adopted\nUnder Prop 218')
circ3 = mpatches.Patch(alpha=0, hatch='', label='')
handles.extend([circ3, circ1, circ2, circ3])
labels.extend(['', 'Not Commonly Adopted\nUnder Prop 218', 'Commonly Adopted\nUnder Prop 218',''])
fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0.07), fancybox=True, fontsize=10, frameon=False, ncol=3)

fig.set_size_inches(5.9, 5.5)

plt.subplots_adjust(hspace=.1)
plt.savefig('general_rate_compare.jpg', bbox_inches = 'tight', dpi=500)
plt.show()


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FIG S2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

LH_bills_pre_ibr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="pre_annual_bills_LH")
LH_bills_pre_dbr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/DEFICIT_NEW/final_summary.xlsx", sheet_name="pre_annual_bills_LH")
LH_bills_fixed_ibr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_fixed_dbr = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/DEFICIT_NEW/final_summary.xlsx", sheet_name="drought_annual_bills_LH")

# load data
LH_bills_pre = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="pre_annual_bills_LH")
LH_bills_vol = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Volumetric_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")
LH_bills_fixed = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/FINAL_RESULTS/Historic_Drought/IBR/Fixed_Surcharge/final_summary.xlsx", sheet_name="drought_annual_bills_LH")

#Base Rate
#Low Income
labels = ['Base ', 'Drought ', ' Base ', ' Drought ']
fixed_crev = [LH_bills_pre_ibr[0][0], LH_bills_fixed_ibr[0][0], LH_bills_pre_dbr[0][0], LH_bills_fixed_dbr[0][0]]
tier_1_crev = [LH_bills_pre_ibr[1][0], LH_bills_fixed_ibr[1][0], LH_bills_pre_dbr[1][0], LH_bills_fixed_dbr[1][0]]
tier_2_crev = [LH_bills_pre_ibr[2][0], LH_bills_fixed_ibr[2][0], LH_bills_pre_dbr[2][0], LH_bills_fixed_dbr[2][0]]
tier_3_crev = [LH_bills_pre_ibr[3][0], LH_bills_fixed_ibr[3][0], LH_bills_pre_dbr[3][0], LH_bills_fixed_dbr[3][0]]
tier_4_crev = [LH_bills_pre_ibr[4][0], LH_bills_fixed_ibr[4][0], LH_bills_pre_dbr[4][0], LH_bills_fixed_dbr[4][0]]
fixed_lh = [0, LH_bills_fixed_ibr[5][0], 0, LH_bills_fixed_dbr[5][0]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

fig, axs = plt.subplots(2,2, gridspec_kw={'width_ratios': [4, 3]})

axs[0,0].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["//", "//",  "", ""])
axs[0,0].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["//", "//",  "", ""])

axs[0,0].set_title('Base Rate Comparison\n', fontsize=12)
axs[0,0].set_xticklabels([])
axs[0,0].set_ylim([0, 1550])
axs[0,0].text(0.01,0.92, "(a)", transform=axs[0,0].transAxes, fontsize=12)
axs[0,0].set_ylabel("Low-Income Bills\n($/yr)", fontsize=12)

# Surcharge
# Low Income
labels = ['Base', 'Fixed', 'Volumetric']
fixed_crev = [LH_bills_pre[0][0], LH_bills_fixed[0][0], LH_bills_vol[0][0]]
tier_1_crev =[LH_bills_pre[1][0], LH_bills_fixed[1][0], LH_bills_vol[1][0]]
tier_2_crev = [LH_bills_pre[2][0], LH_bills_fixed[2][0], LH_bills_vol[2][0]]
tier_3_crev = [LH_bills_pre[3][0], LH_bills_fixed[3][0], LH_bills_vol[3][0]]
tier_4_crev = [LH_bills_pre[4][0], LH_bills_fixed[4][0], LH_bills_vol[4][0]]
fixed_lh = [0, LH_bills_fixed[5][0], LH_bills_vol[5][0]]
tier_1_lh = [0, 0, LH_bills_vol[6][0]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12345 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_lh[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

axs[0,1].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["", "", "//"])
axs[0,1].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["", "", "//"])
axs[0,1].bar(labels, tier_1_lh, width, bottom=t_f12345, label='Volumetric Surcharge', color='#E34234', hatch = ["", "", "//"])

axs[0,1].set_title('Surcharge Comparison\n', fontsize=12)
axs[0,1].set_ylim([0, 1550])
axs[0,1].set_yticklabels([])
axs[0,1].set_xticklabels([])
axs[0,1].text(0.01,0.92, "(b)", transform=axs[0,1].transAxes, fontsize=12)

#Base Rate
#High Income
labels = ['Base', 'Drought ', ' Base', ' Drought']
fixed_crev = [LH_bills_pre_ibr[0][1], LH_bills_fixed_ibr[0][1], LH_bills_pre_dbr[0][1], LH_bills_fixed_dbr[0][1]]
tier_1_crev = [LH_bills_pre_ibr[1][1], LH_bills_fixed_ibr[1][1], LH_bills_pre_dbr[1][1], LH_bills_fixed_dbr[1][1]]
tier_2_crev = [LH_bills_pre_ibr[2][1], LH_bills_fixed_ibr[2][1], LH_bills_pre_dbr[2][1], LH_bills_fixed_dbr[2][1]]
tier_3_crev = [LH_bills_pre_ibr[3][1], LH_bills_fixed_ibr[3][1], LH_bills_pre_dbr[3][1], LH_bills_fixed_dbr[3][1]]
tier_4_crev = [LH_bills_pre_ibr[4][1], LH_bills_fixed_ibr[4][1], LH_bills_pre_dbr[4][1], LH_bills_fixed_dbr[4][1]]
fixed_lh = [0, LH_bills_fixed_ibr[5][1], 0, LH_bills_fixed_dbr[5][1]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

axs[1,0].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["//", "//",  "", ""])
axs[1,0].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["//", "//",  "", ""])

axs[1,0].set_xlabel('IBR                    DBR', fontsize=11)
axs[1,0].set_ylim([0, 1550])
axs[1,0].text(0.01,0.92, "(c)", transform=axs[1,0].transAxes, fontsize=12)
axs[1,0].set_ylabel("High-Income Bills\n($/yr)", fontsize=12)


# Surcharge
# High Income
labels = ['Base', 'Fixed', 'Volumetric']
fixed_crev = [LH_bills_pre[0][1], LH_bills_fixed[0][1], LH_bills_vol[0][1]]
tier_1_crev =[LH_bills_pre[1][1], LH_bills_fixed[1][1], LH_bills_vol[1][1]]
tier_2_crev = [LH_bills_pre[2][1], LH_bills_fixed[2][1], LH_bills_vol[2][1]]
tier_3_crev = [LH_bills_pre[3][1], LH_bills_fixed[3][1], LH_bills_vol[3][1]]
tier_4_crev = [LH_bills_pre[4][1], LH_bills_fixed[4][1], LH_bills_vol[4][1]]
fixed_lh = [0, LH_bills_fixed[5][1], LH_bills_vol[5][1]]
tier_1_lh = [0, 0, LH_bills_vol[6][1]]
width = 0.35  # the width of the bars: can also be len(x) sequence

t_fi = [(tier_1_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12 = [(tier_1_crev[i]+tier_2_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f123 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f1234 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]
t_f12345 = [(tier_1_crev[i]+tier_2_crev[i]+tier_3_crev[i]+tier_4_crev[i]+fixed_lh[i]+fixed_crev[i]) for i in range(len(tier_1_crev))]

axs[1,1].bar(labels, fixed_crev, width, label='Fixed', color='#FFB703', hatch = ["", "", "//"], alpha=0.99)
axs[1,1].bar(labels, tier_1_crev, width, bottom=fixed_crev, label='Tier 1', color = '#8ECAE6', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_2_crev, width, bottom=t_fi, label='Tier 2', color='#219EBC', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_3_crev, width, bottom=t_f12, label='Tier 3', color='#0077B6', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_4_crev, width, bottom=t_f123, label='Tier 4', color='#023047', hatch = ["", "", "//"])
axs[1,1].bar(labels, fixed_lh, width, bottom=t_f1234, label='Fixed Surcharge', color='#FB8500', hatch = ["", "", "//"])
axs[1,1].bar(labels, tier_1_lh, width, bottom=t_f12345, label='Volumetric Surcharge', color='#E34234', hatch = ["", "", "//"])

axs[1,1].set_ylim([0, 1550])
axs[1,1].set_yticklabels([])
axs[1,1].text(0.01,0.92, "(d)", transform=axs[1,1].transAxes, fontsize=12)
handles, labels = axs[1,1].get_legend_handles_labels()
circ1 = mpatches.Patch(hatch='/////', label='Not Commonly Adopted\nUnder Prop 218', alpha=0.0)
circ2 = mpatches.Patch(alpha=0, hatch='', label='Commonly Adopted\nUnder Prop 218')
circ3 = mpatches.Patch(alpha=0, hatch='', label='')
handles.extend([circ3, circ1, circ2, circ3])
labels.extend(['', 'Not Commonly Adopted\nUnder Prop 218', 'Commonly Adopted\nUnder Prop 218',''])
fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(0.5, 0.07), fancybox=True, fontsize=10, frameon=False, ncol=3)

fig.set_size_inches(5.9, 5.5)

plt.subplots_adjust(hspace=.1)

plt.savefig('constant_deficit_compare.jpg', bbox_inches = 'tight', dpi=500)
plt.show()


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FIG S4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

sensitivity = pd.read_excel("C:/Users/kcuw9/Documents/Thesis_3_30_23/Summer Results/Sensitivity_PED/Summary_Sensitivity.xlsx", sheet_name="Change in Bills")
sensitivity_t = sensitivity.T
sensitivity_new = pd.DataFrame(sensitivity_t.values[1:], columns=sensitivity_t.iloc[0])
PED = np.linspace(0,0.9,10)

fig, ax = plt.subplots()

ax.plot(PED,sensitivity_new['HI'], label="High Income")
ax.plot(PED,sensitivity_new['LI'], label="Low Income")
ax.scatter(PED,sensitivity_new['HI'])
ax.scatter(PED,sensitivity_new['LI'])
ax.set_ylabel("Change in Annual Bills During Drought ($)")
ax.set_xlabel("PED")
ax.set_title("Price Elasticity of Demand Sensitivity\nLow and High Income Bills")
ax.legend(loc='center right', frameon=False)

fig.set_size_inches(5.9, 4)
plt.savefig('sensitivity_PED.jpg', dpi=500)
plt.show()