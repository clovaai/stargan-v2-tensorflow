## データベースから画像を取得し、ローカルに保存するプログラム
import sqlite3
import io
import os
from pprint import pprint
from PIL import Image
from datetime import datetime

# データベースに接続
conn = sqlite3.connect('study_model.sqlite3')
c = conn.cursor()

# 画像のバイナリデータを取得する
c.execute("SELECT id, image, taken_date FROM image_data")

image_data = c.fetchall()
count = 0

for data in image_data:
  # 撮影時間のうちhourのみを抽出
  date_time = datetime.strptime(data[2], '%Y-%m-%d %H:%M:%S')
  hour = date_time.hour
  dir_num = hour

  # 指定した時間以外の撮影時間であった場合continue
  if dir_num not in[15, 18, 21]:
    continue

  # バイナリデータをImageオブジェクトに変換する
  img = Image.open(io.BytesIO(data[1]))

  # 指定したディレクトリがない場合新しく作成
  dir_path = './data/custom/train/'+ str(dir_num)+'_hour/'
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  # # 画像を保存する
  image_save_path = './data/custom/train/'+ str(dir_num)+'_hour/'+'flickr_'+str(hour)+'_'+str(count).zfill(6) + '.png'

  img.save(image_save_path)

  count+=1