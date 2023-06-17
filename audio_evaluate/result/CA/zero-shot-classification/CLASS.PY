import laion_clap
import glob
import json
import torch
import numpy as np
import os

device = torch.device('cpu')

# download https://drive.google.com/drive/folders/1scyH43eQAcrBz-5fAw44C6RNBhC3ejvX?usp=sharing and extract ./ESC50_1/test/0.tar to ./ESC50_1/test/
# esc50_test_dir = 'ESC_test/test'# 'ESC50_1/test'
class_index_dict_path = './class_labels/ESC50_class_labels_indices_space.json'

# Load the model
model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
model.load_ckpt()

# Get the class index dict
class_index_dict = {v: k for v, k in json.load(open(class_index_dict_path)).items()}

# Result
mean_rank = 0
median_rank = 0
r1 = 0
r5 = 0
r10 = 0
map10 = 0


def CA(audio_files ,ground_truth_idx):
    with torch.no_grad():
        ground_truth = torch.tensor(ground_truth_idx).view(-1, 1)
        # print(len(audio_files),len(json_files),len(ground_truth_idx))
        # Get text features
        all_texts = ["This is a sound of " + t for t in class_index_dict.keys()]
        text_embed = model.get_text_embedding(all_texts)
        audio_embed = model.get_audio_embedding_from_filelist(x=audio_files)

        ranking = torch.argsort(torch.tensor(audio_embed) @ torch.tensor(text_embed).t(), descending=True)

        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.cpu().numpy()

        metrics = {}
        metrics[f"mean_rank"] = preds.mean() + 1
        metrics[f"median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"R@{k}"] = np.mean(preds < k)
        # map@10
        metrics[f"mAP@10"] = np.mean(np.where(preds < 10, 1 / (preds + 1), 0.0))

        print(
            f"Zeroshot Classification Results: "
            + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
        )
        global mean_rank
        global median_rank
        global r1
        global r5
        global r10
        global map10
        mean_rank = mean_rank + metrics[f"mean_rank"]
        median_rank = median_rank + metrics[f"median_rank"]
        r1 = r1 + metrics[f"R@{1}"]
        r5 = r5 + metrics[f"R@{5}"]
        r10 = r10 + metrics[f"R@{10}"]
        map10 = map10 + metrics[f"mAP@10"]

def main():
    for i in range(50):
        # file location
        esc50_test_dir = 'audioresultpredict/L1_no_atten-correct-2/' + str(i)

        audio_files = os.listdir(esc50_test_dir)

        for j in range(40):
            audio_files[j] = esc50_test_dir+"/"+audio_files[j]

        ground_truth_idx = [i for j in range(40)]
        print(audio_files)
        print(ground_truth_idx)
        CA(audio_files, ground_truth_idx)
    global mean_rank
    global median_rank
    global r1
    global r5
    global r10
    global map10
    mean_rank = mean_rank/50
    median_rank = median_rank/50
    r1 = r1/50
    r5 = r5/50
    r10 = r10/50
    map10 = map10/50
    print("Result")
    print("mean_rank", mean_rank,"median_rank", median_rank, "r1",r1,"r5",r5,"r10",r10,"map10",map10)

    
if __name__ == '__main__':
    main()