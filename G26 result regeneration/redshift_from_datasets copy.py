import os, re, glob
import pandas as pd

#SALT3 data
folder = "/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/datasets"
in_folders = ["SALT3TRAIN_K21_Foundation_DR1", "SALT3TRAIN_K21_Hicken2009","SALT3TRAIN_K21_CfA4p1","SALT3TRAIN_K21_CfA4p2","SALT3TRAIN_K21_CSPDR2","SALT3TRAIN_K21_SDSS","SALT3TRAIN_K21_PS1MD","SALT3TRAIN_K21_DESSN3YR","SALT3TRAIN_K21_SNLS3"]

records = []  

for label in range(len(in_folders)):
    d=in_folders[label]
    full_d = os.path.join(folder, d)
    listfile = glob.glob(os.path.join(full_d, "*.LIST"))[0]
    with open(listfile) as f:
        files=[]
        for l in f:
            if l.strip():
                files.append(l.strip())
    for fn in files:
        path = os.path.join(full_d, fn)
        if not os.path.exists(path):
            continue
        snid, z = None, None
        with open(path, errors="ignore") as fh:
            for line in fh:
                if line.startswith("SNID:"):
                    snid = line.split(":",1)[1].strip()
                elif line.startswith("REDSHIFT_HELIO:") or line.startswith("REDSHIFT_FINAL:"):
                    m = re.search(r"REDSHIFT_(?:HELIO|FINAL):\s*([\-0-9.]+)", line)
                    if m and z is None:
                        z = float(m.group(1))
                if snid and z is not None:
                    break
        records.append({"survey": d, "snid": snid, "z": z, "file": fn})

print(f"Total parsed records: {len(records)}")
by_survey = {}
for r in records:
    by_survey.setdefault(r["survey"], []).append(r)
for s, rs in by_survey.items():
    print(f"{s}: {len(rs)} entries, {len(set(r['snid'] for r in rs))} unique SNIDs, missing z: {sum(1 for r in rs if r['z'] is None)}")

df_salt=pd.DataFrame(records)
print(df_salt)
df_salt.to_csv('/Users/gurbanimahajan/Desktop/COSMO_4THYEAR/tasks assigned/bayesn_resultverification/salt3_redshifts.csv',index=False)

