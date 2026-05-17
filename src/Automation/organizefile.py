## 毎日のファイル整理を自動化する ##

import os
import shutil

def organize_files(folder_path):
    # 指定したフォルダ内のオブジェクトであって、かつファイルであるものの名前をリストにまとめる（フォルダはリストには入らない）
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]

    for file in files:  # ファイル名をforで回す
        file_ext = file.split('.')[-1]  # .で分割して配列に直して、拡張子だけを代入する
        target_folder = os.path.join(folder_path, file_ext) # 移動先のその拡張子専用のフォルダーのパス

        # 拡張子ごとのフォルダがない場合は作成
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        # ファイルを移動 shutil.move(移動元, 移動先)
        shutil.move(os.path.join(folder_path, file), os.path.join(target_folder, file))

# 使用例
print(os.getcwd())
organize_files('/home/aikawa/projects/pytorch/src/Lesson9')