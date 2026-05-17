## 指定したウェブページのタイトルを取得するものです。 ##

# requests: 指定したURLにアクセスし、HTMLなどのデータをダウンロードするためのライブラリ
# Beautifulsoup: ダウンロードした大量の文字データをPythonが読み込めるように整理・解析するためのライブラリ

import requests     # ウェブページにHTTPリクエストを送信するためのライブラリ
from bs4 import BeautifulSoup   # 取得したHTMLを解析し、特定の情報を抽出するためのライブラリ

# ウェブページのURL
url = "https://www.youtube.com/"

# ウェブページを取得
response = requests.get(url)    # HTMLのデータなどを取得できる

# BeautifulSoupで解析
soup = BeautifulSoup(response.text, 'html.parser') # response.textでHTMLのテキストデータだけを取り出す

# タイトルを取得
title = soup.title.string   # <title> ~ </title> というタグを探し出して、そのテキスト部分を取り出す
print(f"ページのタイトル: {title}")

# print(soup) # 数100行あるので注意