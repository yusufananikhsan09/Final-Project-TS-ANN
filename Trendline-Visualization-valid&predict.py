import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

path2 = glob.glob(r'D:\STUDI\Tugas Akhir\DATASET\PengolahanML\Validasi\Validasi_csv\*.csv')
valid = np.zeros((len(path2),1))
jumlah_epoch = 30

id_pt2 = 5772 #5789 #5779 #5776 #5772
id_pt = 1192 #1209 #1199 #1196 #1192

num_pt = int(2*jumlah_epoch)
all_pt = int(5*jumlah_epoch)
prediksi = np.zeros((num_pt ,1))
disp_data = np.zeros((all_pt, 1))

for i in range(len(path2)):
    df = pd.read_csv(path2[i])
    valid[i,]= df.Disp.loc[id_pt2]

# datavalid = pd.DataFrame(valid, columns=['Disp'])
# datavalid.to_csv(r"D:\STUDI\Tugas Akhir\DATASET\PengolahanML\Validasi{}.csv".format(id_pt2))