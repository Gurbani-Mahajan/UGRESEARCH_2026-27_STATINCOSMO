#REGENERATING RESULTS OF THE BAYESN X DOVEKIE PUB: G26 MODEL
import numpy as np
import os,glob,re
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

'''
# Force JAX to use exactly 1 CPU device (prevents multiprocessing crashes)
# Disable aggressive JAX allocator optimizations which cause Mac Segfaults
os.environ["G_SLICE"] = "always-malloc"
os.environ["MALLOC_CHECK_"] = "3"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
jax.config.update("jax_platform_name", "cpu")
'''

numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)

#checking that JAX actually sees 4 logical CPU devices
print(f"Available devices: {jax.local_device_count()}")

'''
FIGURE 1: REDSHIFTS HISTOGRAM
'''
'''
#loading calibrated dataset
ds_salt3 = pd.read_csv("/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/salt3_redshifts.csv") 
ds_dovekie = pd.read_csv("/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/dovekie_data.txt", sep=r"\s+", comment=None, skiprows=0, header=0)
print('SALT3:')
print(ds_salt3.head())
print('Dovekie:')
print(ds_dovekie.head())

z=[] #redshift list

#using those redshifts of those SNe from ds_salt3 that Dovekie has calibrated
dov_snid=[]
for i in ds_dovekie["#"]:
    str_i=str(i)
    dov_snid.append(str_i)

snid_match=[]
#snid_match=np.zeros([1003,2]) #matching file,snid list of matched SNe
file_match=[]
folder_match=[]
cnt=0
for i in range(len(ds_salt3)):
    snid_i=str(ds_salt3["snid"][i])
    if snid_i in dov_snid:
        snid_match.append(snid_i)
        file_match.append(str(ds_salt3["file"][i]))
        folder_match.append(str(ds_salt3["survey"][i]))
        cnt+=1
        z.append(ds_salt3["z"][i])

print(len(z))

#plotting histogram of redshifts
#creating log-bins
log_bins=np.logspace(np.log10(min(z)),np.log10(max(z)),30)
plt.figure()
plt.hist(z,bins=log_bins,histtype='step',color='r')
plt.xscale('log') 
plt.xlim(left=10**(-2))
plt.xlabel('Redshifts')
plt.ylabel('Count')
plt.title('FIG 1: TRAINING REDSHIFTS HISTOGRAM')
plt.axvline(x=(0.08),linestyle=':')
plt.show()
'''

'''
FIGURE 2: G26 MODEL SED (at peak brightness)
'''
'''
#generating indexing method to extract data files of matched SNe
snid_match=np.array(snid_match)
file_match=np.array(file_match)
folder_match=np.array(folder_match)
match_arr=np.column_stack([snid_match,file_match,folder_match])
print(match_arr[1:5,:])

#collecting photometric data from files
folder = "/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/datasets" 
in_folders = ["SALT3TRAIN_K21_Foundation_DR1", "SALT3TRAIN_K21_Hicken2009","SALT3TRAIN_K21_CfA4p1","SALT3TRAIN_K21_CfA4p2","SALT3TRAIN_K21_CSPDR2","SALT3TRAIN_K21_SDSS","SALT3TRAIN_K21_PS1MD","SALT3TRAIN_K21_DESSN3YR","SALT3TRAIN_K21_SNLS3"]

records_photo = [] 
records_spect = []
peak_dat=[]
for label in range(len(in_folders)):
    d=in_folders[label]

    #checking if folder matches
    if d in match_arr[:,2]:
        full_d = os.path.join(folder, d)
        listfile = glob.glob(os.path.join(full_d, "*.LIST"))[0]
        with open(listfile) as f:
            files=[]
            for l in f:
                if l.strip():
                    files.append(l.strip())

        #print(files)

        #checking if file matches
        cnt=0
        files_all=[]
        for fn in files:
            cnt+=1
            if fn in match_arr[:,1]:
                files_all.append(fn)
                path = os.path.join(full_d, fn)
                if not os.path.exists(path):
                    continue
                photometric_time=[]
                photometric_flux=[]
                spectroscopic_lambda=[]
                spectroscopic_flux=[]
                file_name_photo=[]
                file_name_spect=[]
                photometric_mb=0
                peak_dt = np.nan
                with open(path, errors="ignore") as fh:
                    for line in fh:

                        if line.startswith("SNID:"):
                            file_name=line.split()[1]
                        #collecting photometric observations
                        # estimated phase of peak brightness
                        if line.startswith("SEARCH_PEAKMJD:"):
                            peak_phase = float(line.split(":", 1)[1].strip())

                        if line.startswith("OBS:"):
                            parts_photo = line.split()
                            obs_time = float(parts_photo[1])
                            photometric_time.append(float(parts_photo[1]))
                            photometric_flux.append(float(parts_photo[4]))
                            file_name_photo.append(str(file_name))
                            #peak uncorrected magnitude
                            dt = abs(obs_time - peak_phase) #finding closest date
                            if len(parts_photo)>6:
                                obs_mag=float(parts_photo[6])
                                if dt < peak_dt and np.isfinite(obs_mag) and obs_mag > 0:
                                    peak_dt = dt
                                    photometric_mb = obs_mag

                        #collecting spectroscopic observations at peak brightness
                        if line.startswith("SPEC:"):
                            parts_spec = line.split()
                            spectroscopic_lambda.append((float(parts_spec[1]) + float(parts_spec[2]))/2) # lambda_central =(lambda_min +lambda_max)/2
                            spectroscopic_flux.append(float(parts_spec[3]))
                            file_name_spect.append(str(file_name))

                        if line.startswith('REDSHIFT_HELIO:'):
                            parts_z=line.split()
                            z_sn = float(parts_z[1])

                photometric=np.column_stack([photometric_time,photometric_flux])
                spectroscopic=np.column_stack([spectroscopic_lambda,spectroscopic_flux,file_name_spect])
                
                #turning phase into relative phase (to peak brightness phase)
                photometric[:,0]=np.float64(photometric[:,0])-peak_phase
                
                records_photo.append(photometric)
                peak_dat.append([z_sn,photometric_mb])
                records_spect.append(spectroscopic)


#plotting SED of first SN
sn1_spect=records_spect[0]
print('Wavelength range:',min(sn1_spect[:,0]),':',max(sn1_spect[:,0]))

#plotting peak mb (uncorrected) vs z
peak_dat=np.array(peak_dat)
print(peak_dat)
plt.figure()
plt.scatter(peak_dat[:,0],peak_dat[:,1])
plt.axhline(y=19.5)
plt.title('Absolute Magnitude Mb (uncorrected) for '+str(len(match_arr[:,1]))+' SNe Ia')
plt.ylabel('Absolute Magnitude')
plt.xlabel('Redshift')
plt.show()
'''

#g26 model fitting of SED (no dust, peak brightness)
model_g26 = SEDmodel(load_model="/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/bayesn/bayesn/model_files/G26_model/BAYESN.YAML") 
'''
l_g26, spec_g26, params_g26 = model_g26.simulate_spectrum(t=np.array([0.0]), N=1, z=0, mu=0, ebv_mw=0,AV=0, theta=0, eps=0, del_M=0, RV=3.1)
sn1_g26= np.column_stack([l_g26.T, spec_g26[:,:,0].T])

#t21 model fitting of SED (no dust, peak brightness)
model_t21 = SEDmodel(load_model='T21_model')
l_t21, spec_t21, params_t21 = model_t21.simulate_spectrum(t=np.array([0.0]), N=1, z=0, mu=0, ebv_mw=0,AV=0, theta=0, eps=0, del_M=0, RV=3.1)
sn1_t21= np.column_stack([l_t21.T, spec_t21[:,:,0].T])

#plotting REAL SN SED 
plt.figure()
plt.scatter(np.float64(sn1_spect[:,0]),np.float64(sn1_spect[:,1]),marker='.')
plt.title('FIG 2 (A) : SED of SN with SNID :'+str(sn1_spect[0,2]))
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.grid()
plt.show()

#plotting simulated SED with G26
plt.figure()
plt.plot(sn1_g26[:,0],sn1_g26[:,1], 'r',label='G26')
plt.plot(sn1_t21[:,0],sn1_t21[:,1],label='T21')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.grid()
plt.title('FIG 2 (B): MODEL G26 VS T21 SIMULATED SED (peak brightness, no dust)')
plt.legend(loc='best')
plt.show()
'''

'''
FIGURE 3: MODEL REST FRAME LIGHT CURVES IN THE UGRIZ BANDS
'''

#generating data for different phases (theta)
'''
print(inspect.signature(model_g26.simulate_light_curve))
print(model_g26.simulate_light_curve.__annotations__) 
'''
'''
time_range=np.linspace(-10,40,50)
lambda_bands=['u_LSST','g_LSST','r_LSST','i_LSST','z_LSST']
mag_1,magerr_1,params_1 = model_g26.simulate_light_curve(t=time_range,N=1,bands=lambda_bands,z=0,AV=0,mu=0,eps=0,theta=-1,RV=3.1,mag=True)
mag0,magerr0,params0 = model_g26.simulate_light_curve(t=time_range,N=1,bands=lambda_bands,z=0,AV=0,mu=0,eps=0,theta=0,RV=3.1,mag=True)
mag1,magerr1,params1 = model_g26.simulate_light_curve(t=time_range,N=1,bands=lambda_bands,z=0,AV=0,mu=0,eps=0,theta=1,RV=3.1,mag=True)
mag_1=np.array(mag_1)
mag_err_1=np.array(magerr_1)
print(mag_1.shape)
mag0=np.array(mag0)
mag_err0=np.array(magerr0)
mag1=np.array(mag1)
mag_err1=np.array(magerr1)

num_per_band = 50 #number of magnitudes recorded per band

#plotting first SN 
#seperating into bands
fig,ax=plt.subplots(1,1,figsize=(7.5, 7.5))
c= ["#3b0f53","#3e498a",'#35738f','#23ab8b','#82d346' ]
offset_c=[-1,2,4,6,8]
offset_m=[-0.05,0,0.05]
mags=[mag_1,mag0,mag1]
errs = [mag_err_1,mag_err0,mag_err1]
fmts=['--','-',':']
for m in range(len(mags)):
    #splitting by colour band
    mag=mags[m]
    err=errs[m]
    col_mags = np.split(mag,5)
    col_errs = np.split(err,5)
    for cm in range(len(col_mags)):
        col_mag = col_mags[cm]
        col_err = list(col_errs[cm])
        ax.plot(time_range, col_mag + offset_c[cm] + offset_m[m],linestyle=fmts[m], color=c[cm],label=lambda_bands[cm])
        if m==0:
            ax.legend()
    ax.invert_yaxis()
    ax.grid()
ax.text(-12, -8, r"-- $\theta_1 = -1$,  - $\theta_1 = 0$,  : $\theta_1 = 1$",fontsize=12,bbox=dict(facecolor='wheat', alpha=0.5))
fig.supxlabel('Phase')
fig.supylabel('Magnitude (apparent)+offset')
fig.suptitle('FIGURE 3: G26 LIGHTCURVES')
plt.show()
'''

'''
FIGURE 5: THE HUBBLE SCATTER
'''
#print(inspect.signature(SEDmodel))

#loading test data text files
#collecting photometric data from files

#fitting the light curves of SNs from DES-SN5YR

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
print(f"{len(head_clean)} / {len(head_test)} SNe pass all cuts")

# get all SN's photometry
dat=[]
filters_master = {'g': 'g_DES', 'r': 'r_DES', 'i': 'i_DES', 'z': 'z_DES'}
#fig_fit,axs_fit=plt.subplots(2,3)
theta1=unicodedata.lookup('GREEK SMALL LETTER THETA')
head_clean=head_clean[:500] #subject to change with multiple iterations
for i in range(len(head_clean)):
    #getting individual sn flux vs time data
    row = head_clean.iloc[i] #ith sn
    snid = row["SNID"]
    snid = str(snid).split('b')[1]
    ag=1
    #print((snid))
    #if str(snid)=="'1319614'":
    if ag==1:
        redshift = row['REDSHIFT_FINAL']
        peak_phase = row['PEAKMJD']
        print(peak_phase)
        ebv_mw = row['MWEBV']
        low, high = row["PTROBS_MIN"], row["PTROBS_MAX"]
        individual_photo_data = phot_test.iloc[low-1 : high]   #SNANA indexing starts from 1
        t = individual_photo_data["MJD"].to_numpy(dtype=np.float64)
        flux = individual_photo_data["FLUXCAL"].to_numpy(dtype=np.float64)
        flux_err = individual_photo_data["FLUXCALERR"].to_numpy(dtype=np.float64)
        #bands0 = individual_photo_data["BAND"]
        #bands = bands0.str.decode("utf-8").to_numpy()
        #standardise strings
        bands = np.array([str(b.decode("utf-8")) if isinstance(b, bytes) else str(b) for b in individual_photo_data["BAND"]])
        print(bands)
        unique_bands = [str(b) for b in np.unique(bands)]
        filters_master = {'g': 'g_DES', 'r': 'r_DES', 'i': 'i_DES', 'z': 'z_DES'}
        filters = {str(b): str(filters_master[b]) for b in unique_bands if b in filters_master}
        
        #fitting using g26 model
        #print(inspect.signature(model_g26.fit))
        try:
            samples, params = model_g26.fit(t, flux, flux_err, bands, z=redshift, peak_mjd=peak_phase, ebv_mw=ebv_mw, filt_map=filters, mag=False)
        except ValueError as e:
            print('SN with SNID:',snid,' skipped because of error: ',e)
            continue
        samples["RV"] = samples["Rv"]
        
        #finding hubble scatter from inferred parameters
        redshift_lc = params[0]
        ebv_lc = params[1]
        
        
        #for FLAT LAM_CDM COSMOLOGY theoretical/external distance modulus
        print(samples.keys())
        H0=73.24
        omega_m = 0.28
        omega_lambda = 0.72
        c = (3*10**5) #speed of light in km/s
        z_range= np.arange(0,redshift_lc,0.01)
        mu_theo = model_g26.cosmo.distmod(redshift_lc).value
        #marginalising over joint posterior density for distance modulus using SN
        mu_sn = np.mean(samples['mu'])
        mu_std = np.std(samples['mu'])
        theta=np.mean(samples['theta'])
        dust=np.mean(samples['AV'])
        #hubble residual
        hub= mu_sn - mu_theo
        dat.append((snid,redshift,hub,mu_sn,mu_std,mu_theo,theta,dust))

        if i%25==0:
            print('---------------------------------')
            print('STEP ',i)
            print('---------------------------------')
        
        '''
        #plotting 1 lightcurve
        #if i==1
        #getting flux using inferred parameters
        phase_range = np.arange(-10,40, 0.5) #/(1-redshift_lc)
        #print(peak_phase)
        filters_list = ['g_DES','r_DES','i_DES','z_DES']
        redshift_lc= params[0]
        ebv_lc = params[1]
        flux_grid_lc = model_g26.get_flux_from_chains(phase_range, filters_list, samples, zs=np.array([redshift_lc]), ebv_mws=np.array([ebv_lc]), mag=False, num_samples=1000)
        #print(flux_grid_lc.shape)
        flux_lc = flux_grid_lc.mean(axis=(0, 1))
        print(flux_lc.shape)
        flux_err_lc = flux_grid_lc.std(axis=(0, 1))  #mean and std for plot

        #plotting
        cs = ['g', 'r', 'c', 'k'] #colours
        #making phase range of sampled and real SN photometric data equal
        plot_phase = phase_range + peak_phase

        #ax=axs_fit.flatten()[i]
        print(samples.keys())
        theta_samples = np.asarray(samples["theta"])
        AV_samples = np.asarray(samples["AV"])
        theta_mean= np.mean(theta_samples)
        AV_mean= np.mean(AV_samples)
        plt.figure()
        for b_ind, b in enumerate(filters_list):
            #print(b[0])
            b_mjd=[]
            b_flux=[]
            b_flux_err=[]
            for f_ind,f in enumerate(bands):
                if b[0]==f:
                    if peak_phase-10 <= t[f_ind] <= peak_phase+40:
                        b_mjd.append(t[f_ind])
                        b_flux.append(np.float64(flux[f_ind]))
                        b_flux_err.append(np.float64(flux_err[f_ind]))
            plt.errorbar(b_mjd,b_flux,b_flux_err,fmt='x',label=b,color=cs[b_ind])
            plt.plot(plot_phase, flux_lc[b_ind, :], color=cs[b_ind])
            #print(flux_lc[b_ind, :])
            #print(inspect.signature(ax.fill_between))
        plt.legend()
        plt.title('Light Curves of SNe SNID : '+str(snid)+ ' Fit Using G26 Model')
        text= str(" z = " +f"{redshift:.3f}"+"\n "+str(theta1)+' = '+f"{theta_mean:.3f}"+'\n AV = '+f"{AV_mean:.3f}")
        plt.text(max(plot_phase)-5,max(flux_lc[0,:]),text)
        plt.xlabel('Phase')
        plt.ylabel('Flux')
        plt.show()

'''
file_path = '/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/g26model_results.csv'

with open(file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    
    # 3. Write the header row (Optional but recommended)
    writer.writerow(['SNID','Z', 'Hubble Scatter','Calculated Distance Modulus','STD','Theoretical Distance Modulus','Theta','Av'])
    
    # 4. Use zip() to combine lists vertically and write row-by-row
    # zip() turns [item1, item1, item1], [item2, item2, item2]... into rows
    writer.writerows(dat)

print(f"Data successfully saved to {file_path}")

