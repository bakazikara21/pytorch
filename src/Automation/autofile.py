## ファイル操作の自動化 ##
import os
print(os.getcwd())
folder_path = "src/Lesson9"

# フォルダ内のファイルを取得
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        # .txtで終わるファイルならば
        file_path = os.path.join(folder_path,filename)
        with open(file_path,'r',encoding='utf-8') as file:
            # ファイルの中身を読み込んで、そのまま出力するだけ
            content = file.read()
            print(f"---{filename}---")
            print(content)