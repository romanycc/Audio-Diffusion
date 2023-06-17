import matplotlib.pyplot as plt
import numpy as np
import csv

def getTrainData():
    fileroot = "../ESC-50-master/meta/esc50.csv"
    sound_filename = []
    sound_label = []
    skip_first_row = True
    code_dict = {}

    with open(fileroot, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:      # row is a list : filename	fold	target	category	esc10	src_file	take
            if skip_first_row:
                skip_first_row = False
            else:
                code_dict[int(row[2])] = row[3]

    return code_dict


code_dict = getTrainData()
compare_class_num = 10

def get_score(fileroot):
    print("")
    print(f"------ {fileroot} --------")
    fad_list = []
    with open(fileroot, 'r') as file:
        line = file.readline()
        line = file.readline()
        while line != '':
            line = line.split(" ")[1]
            fad_score = float(line.split(":")[1][:-2])

            # print(f"fad_score: {fad_score}")
            fad_list.append(fad_score)
            line = file.readline()
    
    print(f"mean score: {sum(fad_list[:compare_class_num])/compare_class_num}")
    return fad_list[:compare_class_num]

def save_histogram(list1,list2,model1,model2,save_dir):
    categories = np.arange(0, 10)
    label = []
    for i in range(10):
        label.append(code_dict[i])
    width = 0.3
    plt.bar(categories , list1, width, color='green', label=model1)
    plt.bar(categories + width, list2, width, color='red', label=model2)
    plt.xticks(categories + width / 2, label)  
    plt.ylabel('Fad score')
    plt.title(f'Compare {model1} and {model2}')
    plt.legend(bbox_to_anchor=(1,1), loc='upper right')
    plt.savefig(f"result/{save_dir}")
    plt.close()

l1_list = get_score("L1-correct_GT_Fad.txt")
l1_no_atten_list = get_score("L1_no_atten-correct_GT_Fad.txt")
l1_no_norm_list = get_score("L1_no_norm-correct_GT_Fad.txt")
l2_list = get_score("L2_GT_Fad.txt")
l2_no_atten_list = get_score("L2_no_atten_GT_Fad.txt")

# l1 vs l2
save_histogram(l1_list,l2_list,"L1","L2","L1vsL2.png")

# l1 vs no_atten
save_histogram(l1_list,l1_no_atten_list,"atten","w/o atten (L1)","w_wo_atten_L1.png")

# l2 vs no_atten
save_histogram(l2_list,l2_no_atten_list,"atten","w/o atten (L2)","w_wo_atten_L2.png")

# l1 vs no_norm
save_histogram(l1_list,l1_no_norm_list,"norm","w/o norm","w_wo_norm.png")
