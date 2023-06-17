import csv
import os
import shutil

'''

使用說明: 在 audio_evaluate 目錄下執行

'''

def getTrainData():
    fileroot = "../ESC-50-master/meta/esc50.csv"
    sound_filename = []
    sound_label = []
    # lb = preprocessing.LabelBinarizer()
    # lb.fit([i for i in range(10)])
    skip_first_row = True

    with open(fileroot, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:      # row is a list : filename	fold	target	category	esc10	src_file	take
            if skip_first_row:
                skip_first_row = False
            else:
                # 當前資料夾的檔案數量
                destlen = len(os.listdir(f"GT/{int(row[2])}"))
                # 原始檔案路徑
                sourcefile = f"../ESC-50-master/audio/{row[0]}"
                # 目標檔案路徑，依據類別劃分
                destfile = f"GT/{int(row[2])}/{row[3]}_{destlen}.wav"
                #複製檔案
                shutil.copy2(sourcefile, destfile)


# 創建50個類別的資料夾
for i in range(50):
    os.makedirs(f"GT/{i}", exist_ok=True)

getTrainData()