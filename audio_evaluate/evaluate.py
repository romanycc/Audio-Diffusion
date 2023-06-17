import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from frechet_audio_distance import FrechetAudioDistance


label = 1 # 比較哪個類別
model = "L1_no_atten-correct" # 使用哪個model

print(f"L1_no_atten-correct GT Fad_score")

for i in range(50):

    label = i

    dir_1 = f"GT/{label}"
    dir_2 = f"Predict/{model}/{label}"

    frechet = FrechetAudioDistance(
        model_name="vggish",
        use_pca=False, 
        use_activation=False,
        verbose=False
    )

    fad_score = frechet.score(dir_1, dir_2)
    print(f"label{i} fad_score:{fad_score}")