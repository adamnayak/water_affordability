import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# function ratify: create bill from water use (num), rate scheme (r), tiered with widths (t) or untiered
def ratify(num, r, t):
    bill = r[0]
    bill_by_tier = [r[0], 0, 0, 0, 0]
    thres = num - t[0]
    if len(r) > 1:
        if len(r) == 2:
            bill += num*r[1]
            bill_by_tier[1] += num*r[1]
        else:
            if thres >= 0:
                bill += t[0]*r[1]
                bill_by_tier[1] += t[0]*r[1]
                num = thres
                thres = num - (t[1] - t[0])
                if len(r) > 2:
                    if thres >= 0:
                        bill += (t[1]-t[0])*r[2]
                        bill_by_tier[2] += (t[1]-t[0])*r[2]
                        num = thres
                        thres = num - (t[2] - t[1])
                        if len(r) > 3:
                            if thres >= 0:
                                bill += (t[2] - t[1]) * r[3]
                                bill_by_tier[3] += (t[2] - t[1]) * r[3]
                                num = thres
                                if num >= 0:
                                    bill += num * r[4]
                                    bill_by_tier[4] += num * r[4]
                            else:
                                if num > 0:
                                    bill += num * r[3]
                                    bill_by_tier[3] += num * r[3]
                    else:
                        if num > 0:
                            bill += num * r[2]
                            bill_by_tier[2] += num * r[2]
            else:
                if num > 0:
                    bill += num * r[1]
                    bill_by_tier[1] += num * r[1]
    return bill, bill_by_tier


# function split_wateruse_by_tier: split water use into four lists by tier from list of water demand
# where water is the water use list and t is the widths assuming 4 buckets
def split_wateruse_by_tier(water, t, HH_count):
    buckets = {}
    buckets[1] = [0.0] * len(water)
    buckets[2] = [0.0] * len(water)
    buckets[3] = [0.0] * len(water)
    buckets[4] = [0.0] * len(water)
    for i in range(len(water)):
        if water[i] > t[0]:
            buckets[1][i] = t[0]
            if water[i] > t[1]:
                buckets[2][i] = t[1]-t[0]
                if water[i] > t[2]:
                    buckets[3][i] = t[2]-t[1]
                    buckets[4][i] = water[i]-t[2]
                else:
                    buckets[3][i] = water[i]-t[1]
            else:
                buckets[2][i] = water[i]-t[0]
        else:
            buckets[1][i] = water[i]

    wateruse_by_bucket = [len(water)] + [np.sum(buckets[i]) for i in range(1, 5)]

    wateruse_by_bucket_allHH = [sum(HH_count)*12] + [np.sum([buckets[i][j]*HH_count[int(j/12)] for j in range(len(water))]) for i in range(1,5)]

    return wateruse_by_bucket_allHH, wateruse_by_bucket, buckets


# function rev_gen: create bills by tier dict from water use by tier dict and rate scheme
def bills_by_tier(dict, r):
    rev_dict = {}
    for j in dict.keys():
        rev_dict[j] = [dict[j][i] * r[1] for i in range(len(dict[j]))]
    rev_dict[0] = [r[0]] * len(dict[1])

    return rev_dict


# function annual_by_tier: condense monthly by tier dict to be annual aggregates across all income classes as well
# as simply low and high income averages rather than monthly
def annual_by_tier(dict, time_length):
    rev_dict = {}
    lh_dict = {}
    for j in dict.keys():
        rev_dict[j] = [sum(dict[j][i:i+12]) for i in range(0, len(dict[j]), time_length)]
        lh_dict[j] = [sum(rev_dict[j][:3])/3, sum(rev_dict[j][13:])/3]

    return rev_dict, lh_dict


# function rev_gen: create revenue from list by tier and rate scheme
def rev_gen(ls, r):
    return [ls[i]*r[i] for i in range(len(ls))]


# function flatten: create one list from a list of lists
def flatten(l):
    return [item for sublist in l for item in sublist]


# function adjust_water_use: calculate adjusted water use from current water use (list) and adjustment fraction (assumed above 1, eg. 0.15)
def adjust_water_use(water_use, adjustment):
    water_use_adjust = [i * (1 + adjustment) for i in water_use]  # create list of adjusted water use
    return water_use_adjust


# function new_demand: with two numerical bill inputs, a price elasticity value, and a current demand, new demand is outputted
# bill2 is a bill from two periods ago, bill1 is a bill from one period ago, PED is price elasticity, demand is current ancitipated demand for adjustment
def new_demand(bill1, bill2, PED, demand):
    val = (1+(PED*(bill1-bill2)/bill2))*demand
    return val


# function elastic_adjust: return a list of adjusted demand values from PED
def elastic_adjust(b1, b2, PED, w0):
    return [new_demand(b1[i],b2[i],PED, w0[i]) for i in range(len(w0))]


# function subtract_ls: subtract each pair of values in two lists (useful to calculate revenue deficit by tier)
def subtract_ls(ls_a, ls_b):
    return [a_i - b_i for a_i, b_i in zip(ls_a, ls_b)]


# function index_gen: create list of desired indexes from the demand/bills list based on a drought start month in a 12-month list
def index_gen(drought_start, drought_duration, drought_end, ls_length):
    pre_drought_year = flatten([[(i-1+j*12) for i in range(1,13) if (i >= drought_start)]+[(i-1+j*12) for i in range(1,13) if (i < drought_start)] for j in range(int(ls_length/12))])
    during_drought_indexes = flatten([[i%12+j*12 for i in range(drought_start, drought_start+drought_duration)]for j in range(int(ls_length/12))])
    post_drought_year = flatten([[(i-1+j*12) for i in range(1,13) if (i >= drought_end)]+[(i-1+j*12) for i in range(1,13) if (i < drought_end)] for j in range(int(ls_length/12))])

    return pre_drought_year, during_drought_indexes, post_drought_year


# function sub_list: create list from a list of indexes as a subset of a longer list
def sub_list(big_ls, index_ls):
    return [big_ls[i] for i in index_ls]


# function dict_mod: replace a dict of corresponding vals at certain indexes in a dictionary list
def dict_mod(dict, vals, index_ls):
    for i in dict.keys():
        for j in range(len(index_ls)):
            if (i != 5):
                dict[i][index_ls[j]] = vals[i][j]
    return dict


# function dict_transform: create list from a list of indexes as a subset of a longer list within a dictionary
def dict_transform(dict, index_ls):
    new_dict = {}
    for i in dict.keys():
        new_dict[i] = sub_list(dict[i], index_ls)

    return new_dict


# function gen_index: generate list of indexes to pull from condensed monthly list
def gen_index(init, num_buckets):
    return[init+j*12 for j in range(num_buckets)]


# function sum_2lists: add two lists together, lists assumed to be the same length
def sum_2lists(ls1, ls2):
    return [x + y for x, y in zip(ls1, ls2)]


# function sum_dict_lists: add multiple all lists together in a given dictionary, assumed all keys map to a list, lists all same length
def sum_dict_lists(dict):
    ls = [0]*len(dict[1])
    for i in dict.keys():
        ls = sum_2lists(ls, dict[i])

    return ls


# function combine_ts_dict: combine three dictionaries (pre, during, post drought) that represent by tier time series
def combine_ts_dict(pre, during, post, income_buckets, drought_duration):
    all_df = {}

    pre['all_s'] = [sum([pre[list(pre.keys())[i]][j] for i in range(len(pre.keys()))]) for j in range(len(pre[1]))]
    during['all_s'] = [sum([during[list(during.keys())[i]][j] for i in range(len(during.keys()))]) for j in range(len(during[1]))]
    post['all_s'] = [sum([post[list(post.keys())[i]][j] for i in range(len(post.keys()))]) for j in range(len(post[1]))]

    for i in post.keys():
        df1 = pd.DataFrame(np.array(pre[i]).reshape(16, 12), columns=range(1,13))
        df1.insert(0, 'Income', income_buckets)
        df1 = df1.transpose()
        df1.drop(df1.tail(1).index, inplace=True)
        df2 = pd.DataFrame(np.array(during[i]).reshape(16, drought_duration), columns=range(1, drought_duration+1))
        df2 = df2.transpose()
        df3 = pd.DataFrame(np.array(post[i]).reshape(16, 12), columns=range(1,13))
        df3 = df3.transpose()
        df3.drop(df3.head(1).index, inplace=True)
        df4 = df1.append(df2.append(df3, ignore_index=True), ignore_index=True)
        all_df[i] = df4

    if len(during.keys()) > 6:
        d1 = pd.DataFrame(np.zeros((12, 16)))
        fixed_surcharge = pd.DataFrame(np.array(during[5]).reshape(16, drought_duration), columns=range(1, drought_duration+1))
        fixed_surcharge = fixed_surcharge.transpose()
        d_f = d1.append(fixed_surcharge.append(d1, ignore_index=True), ignore_index=True)
        all_df['fixed_sur'] = d_f
        all_df['all_no_s'] = all_df['all_s'].subtract(d_f, fill_value=0)
        if len(during.keys()) > 7:
            d2 = pd.DataFrame(np.zeros((12, 16)))
            vol_surcharge = pd.DataFrame(np.array(during[6]).reshape(16, drought_duration), columns=range(1, drought_duration+1))
            vol_surcharge = vol_surcharge.transpose()
            d_v = d2.append(vol_surcharge.append(d2, ignore_index=True), ignore_index=True)
            all_df['vol_sur'] = d_v
            all_df['all_no_s'] = all_df['all_s'].subtract(d_v, fill_value=0)
    return all_df


# function utility_scale: transforms datasets on individual household level to utility scale use/revenue
def utility_scale(timeseries, HH_count):
    new_ts = {}
    for i in timeseries.keys():
        utility_ts = timeseries[i].tail(-1)
        for (index,colname) in enumerate(utility_ts):
            utility_ts[colname] = HH_count[index]*utility_ts[colname]
        new_ts[i] = utility_ts

    return new_ts


# function utility_summary: provide by tier summary of utility water use and revenue given dictionaries of annual use and bills with HH count
def utility_summary(HH_use, HH_bills, HH_count):
    use_summary = {}
    rev_summary = {}
    for j in HH_use.keys():
        use_summary[j] = sum([HH_count[i] * HH_use[j][i] for i in range(len(HH_count))])

    for k in HH_bills.keys():
        rev_summary[k] = sum([HH_count[i] * HH_bills[k][i] for i in range(len(HH_count))])

    return use_summary, rev_summary

'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~constants and inputs to our model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
#r = [11.26, 9.62, 9.62, 9.62, 9.62]  # rate structure for use, single volumetric rate
#r = [11.26, 11.11, 10.13, 8.07, 4.88]  # rate structure for use, decreasing block rate
r = [11.26, 10.6, 11.58, 13.64, 16.83]  # rate structure for use, increasing block rate
t = [6.0, 8.0, 10.0]  # tier widths
curtailment = -0.25  # percent curtailment (negative, because loss) -0.25 for 2014 statewide curtailment
adjustment = 0  # fraction to adjust water demand depending on rate scheme, (0 for ibr, 0.15 for svr, 0.05 for dbr)
PED = 0.35  # price elasticity of demand, should be 0.35
drought_start = 2  # start month of drought (3 is march)
drought_duration = 12  # number of months in simulated drought
income_buckets = [7500, 12500, 17500, 22500, 27500, 32500, 37500, 42500, 47500, 55000, 67500, 87500, 112500, 137500, 175000, 250000]  # income buckets
HH_count = [2365, 1648, 1456, 1285, 1424, 1027, 1018, 1077, 874, 1931, 2352, 4064, 3110, 2091, 3263, 4895]  # households per bucket
water_use_df = pd.read_excel("estimated_household_demand_DBR_Optimize_final.xlsx", sheet_name="IBR-Model Values", usecols=range(1, 13))  # load water demand modeled for an increasing block rate, before drought
water_use_ls = flatten(water_use_df.values.tolist())  # create list of water use from loaded dataframe
volumetric_sc = False  # calc fixed or volumetric surcharge, True for volumetric, False for fixed only
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

# pre-drought case scenario development
adjusted_use_ls = adjust_water_use(water_use_ls, adjustment)  # adjust water use based on rate scheme
utility_pre_drought_by_tier_tot, pre_drought_by_tier_tot, pre_drought_by_tier = split_wateruse_by_tier(adjusted_use_ls, t, HH_count) # split water use by tier of use
ann_HH_use_pre_drought_by_tier, LH_HH_use_pre_drought_by_tier = annual_by_tier(pre_drought_by_tier, 12) # annual water use

mon_HH_bills_pre_drought_by_tier = bills_by_tier(pre_drought_by_tier, r) # monthly water bills
ann_HH_bills_pre_drought_by_tier, LH_HH_bills_pre_drought_by_tier = annual_by_tier(mon_HH_bills_pre_drought_by_tier, 12) # annual water bills
rev_utility_pre_drought_by_tier_tot = rev_gen(utility_pre_drought_by_tier_tot, r) # annual water bills, low and high income only


# initial curtailment estimate
curtail_use_ls = adjust_water_use(adjusted_use_ls, curtailment)  # curtail water use
utility_init_drought_by_tier_tot, init_drought_by_tier_tot, init_drought_by_tier = split_wateruse_by_tier(curtail_use_ls, t, HH_count) # split curtailed water use by tier of use

mon_HH_bills_init_drought_by_tier = bills_by_tier(init_drought_by_tier, r) # monthly water bills
ann_HH_bills_init_drought_by_tier, LH_HH_bills_init_drought_by_tier = annual_by_tier(mon_HH_bills_init_drought_by_tier, 12) # annual water bills
rev_utility_init_drought_by_tier_tot = rev_gen(utility_init_drought_by_tier_tot, r) # annual water bills, low and high income only

# initial deficit estimations
water_use_init_deficit_by_tier_tot = subtract_ls(utility_pre_drought_by_tier_tot, utility_init_drought_by_tier_tot)  # water use decrease from curtail
rev_utility_init_deficit_by_tier_tot = subtract_ls(rev_utility_pre_drought_by_tier_tot, rev_utility_init_drought_by_tier_tot)  # revenue deficit from curtail


# initial surcharge calculations
tot_rev_deficit = sum(rev_utility_init_deficit_by_tier_tot)  # total revenue loss through curtailment
rev_utility_init_drought_by_tier_tot.append(tot_rev_deficit)  # add revenue loss as surcharge amount to end of init drought utility revenue metrics
tot_use_init = sum(utility_init_drought_by_tier_tot)  # total water use after curtailment
s_f = [tot_rev_deficit/sum(HH_count)/12, 0, 0, 0, 0]  # calc flat surcharge

perc_fixed = rev_utility_pre_drought_by_tier_tot[0]/sum(rev_utility_pre_drought_by_tier_tot)  # percent of pre-drought utility revenue coming from a fixed cost
s_s = [tot_rev_deficit*(1-perc_fixed)/tot_use_init]*5  # calc fixed surcharge for single volumetric surcharge
s_s[0] = tot_rev_deficit*perc_fixed/sum(HH_count)/12  # calc volumetric surcharge for single columetric surcharge

# adjusted bills with flat surcharge
mon_HH_bills_init_drought_by_tier_fixedsc = mon_HH_bills_init_drought_by_tier.copy()  # create new dictionary for the fixed surcharge
mon_HH_bills_init_drought_by_tier_fixedsc[5] = [s_f[0]]*len(mon_HH_bills_init_drought_by_tier[4])  # add the fixed surcharge as the final (6th) key
ann_HH_bills_init_drought_by_tier_fixedsc, LH_HH_bills_init_drought_by_tier_fixedsc = annual_by_tier(mon_HH_bills_init_drought_by_tier_fixedsc, 12)  # create the processed dictionaries by annual bills and low/high income for data visualization


# adjusted bills with volumetric surcharge
mon_HH_bills_init_drought_by_tier_volsc = mon_HH_bills_init_drought_by_tier.copy()  # create new dictionary for the volumetric surcharge
mon_HH_bills_init_drought_by_tier_volsc[5] = [s_s[0]]*len(mon_HH_bills_init_drought_by_tier[4])  # add the fixed surcharge as the 6th key
mon_HH_bills_init_drought_by_tier_volsc[6] = [sum([init_drought_by_tier[i+1][j]*s_s[1] for i in range(len(init_drought_by_tier.keys()))]) for j in range(len(init_drought_by_tier[1]))] # add the volumetric surcharge as the 7th key
ann_HH_bills_init_drought_by_tier_volsc, LH_HH_bills_init_drought_by_tier_volsc = annual_by_tier(mon_HH_bills_init_drought_by_tier_volsc, 12)  # create the processed dictionaries by annual bills and low/high income for data visualization

# find indexes of months based on time of the drought
drought_end = (drought_start+drought_duration)%12
pre_drought_indexes, during_drought_indexes, post_drought_indexes = index_gen(drought_start, drought_duration, drought_end, len(mon_HH_bills_pre_drought_by_tier[0]))

# create water use time series by drought phase
pre_drought_water_use_ts = dict_transform(pre_drought_by_tier, pre_drought_indexes) # create time series by tier for pre-drought
during_drought_water_use_ts = dict_transform(init_drought_by_tier, during_drought_indexes) # create time series by tier for during drought
post_drought_water_use_ts = dict_transform(pre_drought_by_tier, post_drought_indexes) # create time series by tier for post-drought

# create bills time series by drought phase
pre_drought_bills_ts = dict_transform(mon_HH_bills_pre_drought_by_tier, pre_drought_indexes) # create time series by tier for pre-drought

if (volumetric_sc):
    during_drought_bills_ts = dict_transform(mon_HH_bills_init_drought_by_tier_volsc, during_drought_indexes) # create time series by tier for during drought
else:
    during_drought_bills_ts = dict_transform(mon_HH_bills_init_drought_by_tier_fixedsc, during_drought_indexes)  # create time series by tier for during drought

post_drought_bills_ts = dict_transform(mon_HH_bills_pre_drought_by_tier, post_drought_indexes) # create time series by tier for post-drought

# compile by tier bills into totals
compiled_pre_drought_water_use = sum_dict_lists(pre_drought_water_use_ts)  # compile monthly bills pre-drought
compiled_drought_water_use = sum_dict_lists(during_drought_water_use_ts)
compiled_pre_drought_bills = sum_dict_lists(pre_drought_bills_ts)  # compile monthly bills pre-drought
compiled_drought_bills = sum_dict_lists(during_drought_bills_ts)  # compile monthly bills during drought

# PED interactive demand adjustment
for i in range(drought_duration-1):
    rel_indexes = gen_index(i, len(income_buckets)) # indexes of interest for water demand adjustment
    w = sub_list(compiled_drought_water_use, rel_indexes) # extract current month water use
    if i > 1:
        p_2 = sub_list(compiled_drought_bills, gen_index(i - 2, len(income_buckets))) # extract second to last month bills
        p_1 = sub_list(compiled_drought_bills, gen_index(i - 1, len(income_buckets)))  # extract last month bills
    if i == 1:
        p_2 = sub_list(compiled_pre_drought_bills, gen_index(11, len(income_buckets))) # extract last month in pre-drought bills ts if first one
        p_1 = sub_list(compiled_drought_bills, gen_index(0, len(income_buckets)))  # extract last month bills
    else:
        p_2 = sub_list(compiled_pre_drought_bills, gen_index(10, len(income_buckets)))  # extract second to last month in pre-drought bills ts if first one
        p_1 = sub_list(compiled_pre_drought_bills, gen_index(11, len(income_buckets)))  # extract last month bills

    # create new bills and use based on PED
    new_w = elastic_adjust(p_1, p_2, PED, w)
    new_split = split_wateruse_by_tier(new_w, t, HH_count)[2]
    new_bills = bills_by_tier(new_split, r)
    if (volumetric_sc):
        new_bills[6] = [sum([new_split[i + 1][j] * s_s[1] for i in range(len(new_split.keys()))]) for j in range(len(new_split[1]))]  # add the volumetric surcharge as the 7th key

    # update use and bills
    during_drought_water_use_ts = dict_mod(during_drought_water_use_ts, new_split, rel_indexes)
    during_drought_bills_ts = dict_mod(during_drought_bills_ts, new_bills, rel_indexes)
    compiled_drought_water_use = sum_dict_lists(during_drought_water_use_ts)
    compiled_drought_bills = sum_dict_lists(during_drought_bills_ts)

# create updated annual dicts
ann_HH_use_adj_drought_by_tier, LH_HH_use_adj_drought_by_tier = annual_by_tier(during_drought_water_use_ts, drought_duration) # annual water use
ann_HH_bills_adj_drought_by_tier, LH_HH_bills_adj_drought_by_tier = annual_by_tier(during_drought_bills_ts, drought_duration) # annual water bills

# utility water use and revenue by tier
utility_use_adj_drought_by_tier, rev_utility_adj_drought_by_tier = utility_summary(ann_HH_use_adj_drought_by_tier, ann_HH_bills_adj_drought_by_tier, HH_count)

# stitch data for export
water_use_ts_final = combine_ts_dict(pre_drought_water_use_ts, during_drought_water_use_ts, post_drought_water_use_ts, income_buckets, drought_duration)
bills_ts_final = combine_ts_dict(pre_drought_bills_ts, during_drought_bills_ts, post_drought_bills_ts, income_buckets, drought_duration)

# utility level aggregate
utility_water_use_ts = utility_scale(water_use_ts_final, HH_count)
utility_water_use_ts['all_s']['sum'] = utility_water_use_ts['all_s'].sum(axis=1)
utility_rev_ts = utility_scale(bills_ts_final, HH_count)
utility_rev_ts['all_s']['sum'] = utility_rev_ts['all_s'].sum(axis=1)
utility_rev_ts['all_no_s']['sum'] = utility_rev_ts['all_no_s'].sum(axis=1)

# HH water use timeseries export
writer = pd.ExcelWriter('final_HH_water_use_ts.xlsx', engine='xlsxwriter')

for i in water_use_ts_final.keys():
    water_use_ts_final[i].to_excel(writer, sheet_name=str(i))

writer.save()

# HH bills timeseries export
writer = pd.ExcelWriter('final_HH_bills_ts.xlsx', engine='xlsxwriter')

for i in bills_ts_final.keys():
    bills_ts_final[i].to_excel(writer, sheet_name=str(i))

writer.save()

# utility water use timeseries export
writer = pd.ExcelWriter('final_utility_water_use_ts.xlsx', engine='xlsxwriter')

for i in utility_water_use_ts.keys():
    utility_water_use_ts[i].to_excel(writer, sheet_name=str(i))

writer.save()

# utility revenue timeseries export
writer = pd.ExcelWriter('final_utility_revenue_ts.xlsx', engine='xlsxwriter')

for i in utility_rev_ts.keys():
    utility_rev_ts[i].to_excel(writer, sheet_name=str(i))

writer.save()

# annual summary export
writer = pd.ExcelWriter('final_summary.xlsx', engine='xlsxwriter')

pd.DataFrame.from_dict(utility_pre_drought_by_tier_tot).to_excel(writer, sheet_name='pre_utility_water_use')
pd.DataFrame.from_dict(rev_utility_pre_drought_by_tier_tot ).to_excel(writer, sheet_name='pre_utility_rev')

pd.DataFrame(utility_use_adj_drought_by_tier, index=[0]).to_excel(writer, sheet_name='drought_utility_water_use')
pd.DataFrame(rev_utility_adj_drought_by_tier, index=[0]).to_excel(writer, sheet_name='drought_utility_rev')

pd.DataFrame.from_dict(ann_HH_use_pre_drought_by_tier).to_excel(writer, sheet_name='pre_annual_water_use_all')
pd.DataFrame.from_dict(LH_HH_use_pre_drought_by_tier).to_excel(writer, sheet_name='pre_annual_water_use_LH')

pd.DataFrame.from_dict(ann_HH_use_adj_drought_by_tier).to_excel(writer, sheet_name='drought_annual_water_use_all')
pd.DataFrame.from_dict(LH_HH_use_adj_drought_by_tier).to_excel(writer, sheet_name='drought_annual_water_use_LH')

pd.DataFrame.from_dict(ann_HH_bills_pre_drought_by_tier).to_excel(writer, sheet_name='pre_annual_bills_all')
pd.DataFrame.from_dict(LH_HH_bills_pre_drought_by_tier).to_excel(writer, sheet_name='pre_annual_bills_LH')

pd.DataFrame.from_dict(ann_HH_bills_adj_drought_by_tier).to_excel(writer, sheet_name='drought_annual_bills_all')
pd.DataFrame.from_dict(LH_HH_bills_adj_drought_by_tier).to_excel(writer, sheet_name='drought_annual_bills_LH')

writer.save()