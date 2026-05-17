## 毎日のメール送信の自動化 ##

import smtplib  # メールサーバーに接続し、メールを送信するためのライブラリ

# メールの「件名」「宛先」「本文」などの構造（MIME形式）を作成するためのライブラリ
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import schedule # タスクスケジューラのようなことを設定できるライブラリ
import time     # プログラムを一時停止させるときに使う

def send_email():
    # 宛先や件名の設定
    sender = 'my@student.jp'
    receiver = 'you@example.com'
    subject = '毎日の定例メール'
    body = 'お疲れ様です。本日の進捗を報告します。'

    # メールの設定
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body,'plain'))

    # メール会社の送信サーバー（例：Gmailなら smtp.gmail.com）に接続します。
    with smtplib.SMTP('smtp.example.com', 587) as server:
        server.starttls()   # 通信を暗号化して、パスワードや本文が盗み見られないように保護
        server.login(sender, 'password')    # メールアドレスとパスワードを使ってサーバーにログイン
        server.sendmail(sender,receiver,msg.as_string()) # 実際にメールを送信

# 毎日9時にメールを送信
schedule.every().day.at("09:00").do(send_email)

# スケジューリングの実行
while True:
    schedule.run_pending()
    time.sleep(60)