# REGENERATING RESULTS OF THE BAYESN X DOVEKIE PUB: G26 MODEL
import numpy as np
import os, glob, re
from bayesn.bayesn_model import SEDmodel
import sncosmo
from astropy.table import Table
import matplotlib.pyplot as plt
import pandas as pd
import inspect
import numpyro
import jax
import unicodedata
import inspect
import csv

def align(ref, sub, rtol=1e-5):
    """indices into ref for each element of sub, assuming sub is an ordered subsequence of ref"""
    idx, j = [], 0
    for i, r in enumerate(ref):
        if j < len(sub) and abs(r - sub[j]) <= rtol * max(1.0, abs(r)):
            idx.append(i); j += 1
    return np.array(idx)


#fitting same data using salt3
#extracting all SNe from photo files individually using min/max markers from head files
head_test = Table.read("/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/testing_datasets/DES-SN5YR_DES_HEAD.FITS").to_pandas()
phot_test = Table.read("/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/testing_datasets/DES-SN5YR_DES_PHOT.FITS").to_pandas()

#applying P_ia cuts 
classif = pd.read_csv("/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/testing_datasets/DES_classification.csv")
head_test["SNID_int"] = head_test["SNID"].apply(lambda b: int(b.decode("utf-8")))
# merge classification probability onto the HEAD table
head_test = head_test.merge(classif, left_on="SNID_int", right_on="SNID",how="left", suffixes=("", "_classif"))

# now build one boolean mask combining every cut:
head_clean = head_test[((head_test["MJD_TRIGGER"] != -9) &(head_test["REDSHIFT_FINAL"] > 0) &(head_test["PEAKMJD"] != 0.01) &(head_test["PROB_SNNTRAINV19_z_TRAINDES_V19"] >= 0.5) &(head_test["REDSHIFT_FINAL"] < 0.7))].reset_index(drop=True)

source = sncosmo.get_source('salt3')
model = sncosmo.Model(source=source,effects=[sncosmo.CCM89Dust()],effect_names=['mw'], effect_frames=['obs'])

band_map = {'g':'desg','r':'desr','i':'desi','z':'desz'}
rows = []
for i in range(len(head_clean)):
    if i%10==0:
        print(i)
    row = head_clean.iloc[i]
    low = row["PTROBS_MIN"]
    high = row["PTROBS_MAX"]
    p = phot_test.iloc[low-1:high]
    #standardising strings
    band = np.array([b.decode().strip() if isinstance(b, bytes) else str(b).strip() for b in p["BAND"]])
    #ok = (p["FLUXCALERR"].to_numpy() > 0) & np.isin(band, list(band_map))
    data = sncosmo.photdata.PhotometricData if False else None  # (ignore; plain dict below)
    tab = {'time': p["MJD"].to_numpy(),
           'band': [band_map[b] for b in band],
           'flux': p["FLUXCAL"].to_numpy(),
           'fluxerr': p["FLUXCALERR"].to_numpy(),
           'zp': np.full(len(p["FLUXCAL"]), 27.5), 'zpsys': ['ab']*len(p["FLUXCAL"])}
    z = row["REDSHIFT_FINAL"]
    model.set(z=z, mwebv=row["MWEBV"], mwr_v=3.1, t0=row["PEAKMJD"], x1=0., c=0.)
    try:
        res, fit = sncosmo.fit_lc(tab, model, ['t0','x0','x1','c'],
                              bounds={'t0':(row["PEAKMJD"]-15, row["PEAKMJD"]+15),
                                      'x1':(-4,4), 'c':(-0.5,0.5)},
                              guess_t0=False)   # guess_amplitude stays True
    except Exception as e:
        print(row["SNID"], "failed:", e); continue
    mB = fit.source_peakmag('bessellb', 'ab')
    rows.append((row["SNID"],row["REDSHIFT_FINAL"], mB, res.parameters[res.param_names.index('x1')],
                 res.parameters[res.param_names.index('c')], res.chisq/res.ndof))

file_path= '/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/SALT3_results.csv'
with open(file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # header row 
    writer.writerow(['SNID','Z', 'mB', 'x1','c','chi^2'])
    #adding entries
    writer.writerows(rows)

print(f"Data successfully saved to {file_path}")
'''

file_path = '/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/g26model_results.csv'
df=pd.read_csv(file_path)
print(df.describe())
distance_mod= df["Calculated Distance Modulus"]
theo_distance_mod = df["Theoretical Distance Modulus"]
Z=df["Z"]
hubble_scatter=df["Hubble Scatter"]

file_path_salt = '/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/SALT3_results.csv'
df_salt=pd.read_csv(file_path_salt)
print(df_salt.describe())
mB= df_salt["mB"]
x1 = df_salt["x1"]
c = df_salt["c"]
z = df_salt["Z"]

i_g = align(head_clean["REDSHIFT_HELIO"].to_numpy(),  df["Z"].to_numpy())
i_s = align(head_clean["REDSHIFT_HELIO"].to_numpy(),  df_salt["Z"].to_numpy())
print(len(i_g), len(df), "|", len(i_s), len(df_salt))      # each pair must be equal

skipped_g = head_clean.drop(index=head_clean.index[i_g])
print("G26 skipped:", skipped_g[["SNID","REDSHIFT_HELIO"]].to_string())

df["SNID"]      = head_clean["SNID"].to_numpy()[i_g]
df_salt["SNID"] = head_clean["SNID"].to_numpy()[i_s]
m = df.merge(df_salt.drop(columns="Z"), on="SNID", how="inner")
print("matched SNe:", len(m))

model_g26 = SEDmodel(load_model="/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/bayesn/bayesn/model_files/G26_model/BAYESN.YAML") 
mu_theo = model_g26.cosmo.distmod(m["Z"]).value
# mB - mu_th = M + (-alpha)*x1 + beta*c (TRIPP)
# standard SALT-style cuts, applied before fitting alpha/beta
sel = (np.abs(m["x1"]) < 3) & (np.abs(m["c"]) < 0.3) & (np.abs(m["chi^2"]) < 10.0)
m=m[sel]
mu_theo_sel= model_g26.cosmo.distmod(m["Z"]).value
y = m["mB"] - mu_theo_sel
print(y.shape)
A = np.column_stack([np.ones_like(m["Z"]), m["x1"], m["c"]])
keep=sel.copy()
for j in range(10):
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ coef
    s = 1.48 * np.median(np.abs(r))
    keep = sel & (np.abs(r) < 3 * s)

hubble_salt = (y - A @ coef)     # score on all selected SNe, not just the clipped ones
m["Hubble Scatter SALT3"]=hubble_salt
mu_salt = m["mB"] - A @ coef
m["Calculated Distance Modulus SALT3"] = mu_salt
print(m.describe())

#calculating sigma_NMAD (or hubble residual scatter normalised median absolute deviation)
snmad_bsn = 1.48 * np.median(np.abs(m["Hubble Scatter"]-np.median(m["Hubble Scatter"])))
snmad_salt = 1.48 * np.median(np.abs(hubble_salt-np.median(hubble_salt)))
print('σNMAD of Hubble residual scatter for G26 BayeSN fit = ',snmad_bsn)
print('σNMAD of Hubble residual scatter for SALT3.Dovekie fit = ',snmad_salt)
print('Percentage reduction in scatter = ',((snmad_salt - snmad_bsn)/snmad_salt)*100)

a = (m["Calculated Distance Modulus"].to_numpy() - mu_theo_sel)
a -= np.median(a)   # G26
b = hubble_salt - np.median(hubble_salt)                                               # SALT
nm = lambda x: 1.48*np.median(np.abs(x - np.median(x)))
print("corr(G26, SALT):", np.corrcoef(a, b)[0,1])
print("NMAD G26, SALT, (G26-SALT):", nm(a), nm(b), nm(a-b))
for lo, hi in [(0,0.3),(0.3,0.5),(0.5,0.7)]:
    k = (m.loc[sel,"Z"].to_numpy() >= lo) & (m.loc[sel,"Z"].to_numpy() < hi)
    print(f"{lo}-{hi}  N={k.sum()}  G26={nm(a[k]):.3f}  SALT={nm(b[k]):.3f}")

#plotting hubble residual scatter
fig,axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax_mu = axs[0] #distance modulus plot
ax_res = axs[1] #scatter plot
ax_mu.scatter(m.loc[sel,"Z"], m.loc[sel,"Calculated Distance Modulus SALT3"],color='b',label='Distance modulus (BayeSN fit)')
ax_mu.plot(m.loc[sel,"Z"], m.loc[sel,"Theoretical Distance Modulus"],'wo',markeredgecolor='black')
ax_mu.set_ylabel('Distance Modulus')
ax_mu.legend()
ax_mu.set_ylim(38,44)
#ax_mu.set_xlim(min(np.log(Z)),max(np.log(Z)))
ax_mu.set_xscale("log")
ax_res.scatter(m.loc[sel,"Z"], m.loc[sel, "Hubble Scatter SALT3"],color='b',label='Hubble residuals (flat LCDM)')
ax_res.plot(m.loc[sel,"Z"], np.zeros_like(m.loc[sel,"Z"]),'wo',markeredgecolor='black') #marking the x axis indicating 0 scatter
ax_res.set_ylabel('Scatter')
ax_res.legend()
fig.suptitle('FIG 5: HUBBLE SCATTER OF DES-SN5YR PHOTOMETRIC DATA FIT USING SALT3')
fig.supxlabel('Redshift')
plt.show()

#plotting hubble residual scatter
fig,axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
ax_mu = axs[0] #distance modulus plot
ax_res = axs[1] #scatter plot
ax_mu.scatter(m.loc[sel,"Z"], m.loc[sel,"Calculated Distance Modulus"],color='b',label='Distance modulus (BayeSN fit)')
ax_mu.plot(m.loc[sel,"Z"], m.loc[sel,"Theoretical Distance Modulus"],'wo',markeredgecolor='black')
ax_mu.set_ylabel('Distance Modulus')
ax_mu.legend()
ax_mu.set_ylim(38,44)
#ax_mu.set_xlim(min(np.log(Z)),max(np.log(Z)))
ax_mu.set_xscale("log")
ax_res.scatter(m.loc[sel,"Z"], m.loc[sel, "Hubble Scatter"],color='b',label='Hubble residuals (flat LCDM)')
ax_res.plot(m.loc[sel,"Z"], np.zeros_like(m.loc[sel,"Z"]),'wo',markeredgecolor='black') #marking the x axis indicating 0 scatter
ax_res.set_ylabel('Scatter')
ax_res.legend()
fig.suptitle('FIG 5: HUBBLE SCATTER OF DES-SN5YR PHOTOMETRIC DATA FIT USING G26')
fig.supxlabel('Redshift')
plt.show()
'''

