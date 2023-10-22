from transformers import AutoTokenizer, AutoModelForSequenceClassification
checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

#モデル読み込み
import pickle
model = pickle.load(open('finalized_model.sav', 'rb'))

from transformers import TrainingArguments, Trainer
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import requests
import urllib.parse as parse
import csv
sns.set(font='IPAexGothic')

#YouTubeURL
API_KEY = "AIzaSyACjXuJhWm2xXEdHkZIyh9RXhyKHswjsFs"
URL_HEAD = "https://www.googleapis.com/youtube/v3/commentThreads?"
nextPageToken = ''
item_count = 0
items_output = [
    ['videoId']+
    ['textDisplay']+
    ['textOriginal']+
    ['authorDisplayName']+
    ['authorProfileImageUrl']+
    ['authorChannelUrl']+
    ['authorChannelId']+
    ['canRate']+
    ['viewerRating']+
    ['likeCount']+
    ['publishedAt']+
    ['updatedAt']
]

emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']  # 日本語版
num_labels = len(emotion_names)
count = 0
max_list = [0,0,0,0,0,0,0,0]

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

type = st.selectbox('選択してください',['YouTubeURL','csvファイル'])

if type == 'YouTubeURL':
    video_url = st.text_input('URLを入力してください')
    video_id = video_url.replace('https://www.youtube.com/watch?v=', '')
    exe_num = 1

    for i in range(exe_num):

        #APIパラメータセット
        param = {
            'key':API_KEY,
            'part':'snippet',
            #----↓フィルタ（いずれか1つ）↓-------
            #'allThreadsRelatedToChannelId':channelId,
            'videoId':video_id,
            #----↑フィルタ（いずれか1つ）↑-------
            'maxResults':'100',
            'moderationStatus':'published',
            'order':'relevance',
            'pageToken':nextPageToken,
            'searchTerms':'',
            'textFormat':'plainText',
        }
        #リクエストURL作成
        target_url = URL_HEAD + (parse.urlencode(param))

        #データ取得
        res = requests.get(target_url).json()

        #件数
        item_count += len(res['items'])

        #print(target_url)
        #print(str(item_count)+"件")

        #コメント情報を変数に格納
        for item in res['items']:
            items_output.append(
                [str(item['snippet']['topLevelComment']['snippet']['videoId'])]+
                [str(item['snippet']['topLevelComment']['snippet']['textDisplay'].replace('\n', ''))]+
                [str(item['snippet']['topLevelComment']['snippet']['textOriginal'])]+
                [str(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])]+
                [str(item['snippet']['topLevelComment']['snippet']['authorProfileImageUrl'])]+
                [str(item['snippet']['topLevelComment']['snippet']['authorChannelUrl'])]+
                [str(item['snippet']['topLevelComment']['snippet']['authorChannelId']['value'])]+
                [str(item['snippet']['topLevelComment']['snippet']['canRate'])]+
                [str(item['snippet']['topLevelComment']['snippet']['viewerRating'])]+
                [str(item['snippet']['topLevelComment']['snippet']['likeCount'])]+
                [str(item['snippet']['topLevelComment']['snippet']['publishedAt'])]+
                [str(item['snippet']['topLevelComment']['snippet']['updatedAt'])]
            )

        #nextPageTokenがなくなったら処理ストップ
        if 'nextPageToken' in res:
            nextPageToken = res['nextPageToken']
        else:
            break

    #CSVで出力
    f = open('youtube-comments-list.csv', 'w', newline='', encoding='UTF-8')
    writer = csv.writer(f)
    writer.writerows(items_output)
    f.close()

    df = pd.read_csv('youtube-comments-list.csv')

    corpus_list = df['textOriginal'].to_list()

    df = pd.DataFrame([])

    df_time = pd.read_csv('youtube-comments-list.csv')

    time_list = df_time['publishedAt'].to_list()

else:
    uploaded_file = st.file_uploader("csvファイルをアップロードしてください。", type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(
             uploaded_file)
        st.markdown('### アクセスログ（先頭5件）')
        st.write(df.head(5))

    column_name = st.text_input('コラム名')

    # corpus_list（前処理し、データフレームに格納したテキストをリスト化）
    corpus_list = df[column_name].to_list()

    df = pd.DataFrame([])

for text2 in corpus_list:
  # 推論モードを有効か
  model.eval()

  # 入力データ変換 + 推論
  tokens = tokenizer(text2, truncation=True, return_tensors="pt")
  tokens.to(model.device)
  preds = model(**tokens)
  prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
  d = {'テキスト':text2, '喜び':prob[0], '悲しみ':prob[1],'期待':prob[2],
       '驚き':prob[3], '怒り':prob[4], '恐れ':prob[5],'嫌悪':prob[6], '信頼':prob[7]}
  data = pd.DataFrame(d.values(),index=d.keys()).T
  df = df.append( data )
  df = df.reset_index(drop=True)
  #print(text2,prob)
  max=0.0
  max_kanjou_name = '喜び'
  for kanjou in emotion_names_jp:
    if df.loc[count,kanjou] > max:
      max = df.loc[count,kanjou]
      max_kanjou_name = kanjou
  c = 0
  for k in emotion_names_jp:
    if k == max_kanjou_name:
      max_list[c] += 1
    c += 1
  count += 1

df[['喜び','悲しみ','期待','驚き','怒り','恐れ','嫌悪','信頼']] = df[['喜び','悲しみ','期待','驚き','怒り','恐れ','嫌悪','信頼']].astype(float)

#コメント一覧
df

#分析結果一覧
senti = st.selectbox('感情',['喜び','悲しみ','期待','驚き','怒り','恐れ','嫌悪','信頼'])
syou = st.selectbox('昇順/降順',['昇順','降順'])
display_record_number_slider = st.slider('参照データ数', 0, 50)
max_number_slider = st.slider(label='最大値', min_value=0.0, max_value=1.0,step=0.1)
min_number_slider = st.slider(label='最小値', min_value=0.0, max_value=1.0,step=0.1)

#bool_index = df[senti] >= number_slider
bool_index = (df[senti] >= min_number_slider) & (df[senti] <= max_number_slider)
filtered_df = df[bool_index]

if syou == "昇順":
  filtered_df = filtered_df.sort_values(by=senti)
else:
  filtered_df = filtered_df.sort_values(by=senti, ascending=False)

st.write(filtered_df[0:display_record_number_slider])

#棒グラフ
fig = plt.figure()
sns.boxplot(data=df)
st.pyplot(fig)

#最大値円グラフ
df_max = pd.DataFrame([])
d = {'喜び':max_list[0], '悲しみ':max_list[1],'期待':max_list[2],'驚き':max_list[3], '怒り':max_list[4], '恐れ':max_list[5],'嫌悪':max_list[6], '信頼':max_list[7]}
#d = {'喜び':max_list[0]/sum(max_list), '悲しみ':max_list[1]/sum(max_list),'期待':max_list[2]/sum(max_list),'驚き':max_list[3]/sum(max_list), '怒り':max_list[4]/sum(max_list), '恐れ':max_list[5]/sum(max_list),'嫌悪':max_list[6]/sum(max_list), '信頼':max_list[7]/sum(max_list)}
data = pd.DataFrame(d.values(),index=d.keys()).T
df_max = df_max.append( data )
st.dataframe(df_max)
fig = plt.figure()
plt.pie(max_list, startangle=90, counterclock=False,  autopct='%.1f%%', pctdistance=0.8,labels=emotion_names_jp)
st.pyplot(fig)

#時系列データ
#if type == 'YouTubeURL':
  #time = st.selectbox('選択してください',['月','年'])
  #if time == '月':

  #else: