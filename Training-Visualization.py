# Visualisasi Hasil Training pada Titik Sampel
path = glob.glob(r'D:\STUDI\Tugas Akhir\DATASET\PengolahanML\NEWDATA_BGT\Pelatihan_Testing\1.Training_dan_prediksi\*.csv')

array = np.zeros((len(path),1))

num_pt = int(2*jumlah_epoch)
all_pt = int(5*jumlah_epoch)
prediksi = np.zeros((num_pt ,1))
disp_data = np.zeros((all_pt, 1))
id_pt = 1207

for i in range(len(path)):
    df = pd.read_csv(path[i])
    array[i,]= float(df.Disp.loc[id_pt])

time = pd.read_csv(r"D:\STUDI\Tugas Akhir\DATASET\PengolahanML\timedf.csv",
                  parse_dates=['new_waktu'])
time = time.copy()
pred = pd.DataFrame(predicted[id_pt,:,], columns = ['Disp'])
datalatih = pd.DataFrame(array[:150], columns=['Disp'])
pred = pred[60:]
Disp = pd.concat([datalatih, pred], axis=0).reset_index()
graf = time.join(Disp)

f, ax = plt.subplots()
ax.plot(graf.new_waktu[120:150], 
        graf.Disp[120:150], 
        label='Displacement', zorder=2)
ax.plot(graf.new_waktu[120:], 
        graf.Disp[120:], 
        label='Predicted Displacement', zorder=1)
ax.set(xlabel='Date', 
        ylabel='Displacement (mm)', 
        title=f'Displacement Trend on Point D')
plt.xticks(rotation ='vertical')
plt.subplots_adjust(bottom=0.1)
ax.axvline(x = graf.new_waktu.loc[149], 
          linestyle='--', color='r')
plt.legend(bbox_to_anchor=(1.02, 1.02), loc='upper left')
plt.xlim()
plt.show()
plt.tight_layout()

# ================================================================================
## Melakukan transformasi dimensi dari prediksi menjadi dimensi raster (waktu, kolom, baris, (depth/value))
## Namun, untuk itu ditransformasikan terlebih dahulu menjadi berurutan sesuai indeks_piksel
data_pred = np.zeros((jumlah_epoch*numdata, 1))
for x in range(jumlah_epoch):
    for y in range(numdata):
        if x > 0:
            data_pred[x+y+x*(numdata-1),] = predicted[y,60+x,]#y+x+x*(numdata-1)
        else:
            data_pred[x+y,] = predicted[y,60+x,] #str(y+x) 
            
width = 68
height = 62
img = np.zeros((jumlah_epoch,height,width))


## kemudian, ditransfromasikan ke dimensi raster
for k in range(jumlah_epoch):
    for j in range(height):
        for i in range(width):
            if j>0:
                img[k,j,i] = data_pred[i+j+k*(numdata)+j*(width-1)]
            else:
                img[k,j,i] = data_pred[i+j+k*(numdata)] 
    
#=====================================================================================
# Membuat Visualisasi pada Interval Waktu Tertentu
f, ax = plt.subplots(nrows=3,ncols=2, figsize=(10,15))

## Visualisasi Prediksi ke-1
ax[0,0].imshow(img[0,:,:], cmap='rainbow')
imgdisp1 = ax[0,0].imshow(img[0,:,:],cmap='rainbow')
f.colorbar(imgdisp1, ax=ax[0,0]).set_label('Displacement (mm)')
ax[0,0].set_title('Prediksi Akuisisi pada \n Januari 2021')
#ax[0,0].set_title('Prediksi Akuisisi pada \n{}'.format(graf.new_waktu.loc[150]))
ax[0,0].axis('off')

## Visualisasi Prediksi ke-6
ax[1,0].imshow(img[5,:,:], cmap='rainbow_r')
imgdisp2 = ax[1,0].imshow(img[5,:,:],cmap='rainbow')
f.colorbar(imgdisp2, ax=ax[1,0]).set_label('Displacement (mm)')
ax[1,0].set_title('Prediksi Akuisisi pada \n Maret 2021')
ax[1,0].axis('off')

## Visualisasi Prediksi ke-12
ax[2,0].imshow(img[11,:,:], cmap='rainbow')
imgdisp3 = ax[2,0].imshow(img[11,:,:], cmap='rainbow')
f.colorbar(imgdisp3, ax=ax[2,0]).set_label('Displacement (mm)')
ax[2,0].set_title('Prediksi Akuisisi pada \n Mei 2021')
ax[2,0].axis('off')

## Visualisasi Prediksi ke-18
ax[0,1].imshow(img[17,:,:], cmap='rainbow')
imgdisp4 = ax[0,1].imshow(img[17,:,:], cmap='rainbow')
f.colorbar(imgdisp4, ax=ax[0,1]).set_label('Displacement (mm)')
ax[0,1].set_title('Prediksi Akuisisi pada \n Juli 2021')
ax[0,1].axis('off')

## Visualisasi Prediksi ke-24
ax[1,1].imshow(img[23,:,:], cmap='rainbow')
imgdisp5 = ax[1,1].imshow(img[23,:,:], cmap='rainbow')
f.colorbar(imgdisp5, ax=ax[1,1]).set_label('Displacement (mm)')
ax[1,1].set_title('Prediksi Akuisisi pada \n Oktober 2021')
ax[1,1].axis('off')

## Visualisasi Prediksi ke-30
ax[2,1].imshow(img[29,:,:], cmap='rainbow')
imgdisp6 = ax[2,1].imshow(img[29,:,:], cmap='rainbow')
f.colorbar(imgdisp6, ax=ax[2,1]).set_label('Displacement (mm)')
ax[2,1].set_title('Prediksi Akuisisi pada \n Desember 2021')
ax[2,1].axis('off')
f.savefig(r"D:\Yusuf_2019_TA\TA\DATASET\PengolahanML\Prediksi_pikbased150.png")