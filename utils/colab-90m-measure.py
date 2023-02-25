# 1時間毎に「Google Colab」にアクセスすることで、90分ルールによるリセットを回避できます。

import time
import datetime
import webbrowser

# 1時間毎に任意のノートブックを開く
for i in range(12):
    browse = webbrowser.get('chrome')
    browse.open('https://colab.research.google.com/drive/1r4b_wGBmYUuG4oL5pOI4S7wwgiymAGZ3#scrollTo=bEVPRiXZBliZ')
    print(i, datetime.datetime.today())
    time.sleep(60*60)