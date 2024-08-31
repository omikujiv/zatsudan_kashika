import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from pprint import pprint
import csv

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import colors
import numpy as np
import collections

import japanize_matplotlib

# モデルの場所 dic下に配置
model_path = 'dic/chive-1.2-mc5_gensim/chive-1.2-mc5.kv'

# model read
wv = gensim.models.KeyedVectors.load(model_path)

# CSVfile パス
csv_file_path = 'words.csv'

# 辞書に含まれない単語を取得
def ck_csv(words, wv):
    
    included_words = []
    excluded_words = []
    
    for row in words:
        for word in row:
            if word!='':
                if word in wv:
                    #print(word)
                    included_words.append(word)
                else:
                    print(word)
                    excluded_words.append(word)
    
    print(excluded_words)
    

# リストの重複を確認
def ck_csv_same(words):
    word_list = []
    word_dict = {}
    for row in words:
        for word in row:
            if word!='':
                word_list.append(word)
    
    print(word_list)
    # リストの要素を辞書のキーとして使用する
    for word in word_list:
        if word in word_dict:
            print(word)
        else:
            word_dict[word] = 1

def read_csv(path):
    # read csv
    words=[]
    with open(path, 'r', encoding='SHIFT_JIS') as f:
        reader = csv.reader(f)
        for row in reader:
            #print(row[0])
            words.append(row)
            
    # 単語が辞書にあるかチェック
    # ここで止まる場合はcsvを編集してください
    # KeyedVectorsの辞書に含まれる単語を格納するリスト
    ck_csv(words, wv)
    ck_csv_same(words)
    i = 0
    # 1行ごと
    time_stamp = [] # 時間データ入れる用
    all_dict ={}
    for row in words:
        i +=1
        my_dict = {}
        # 時間データ取り出し
        time_stamp.append([row[5] , row[6]])
        j = 0 # j = 0, 1, 2, 3, 4 まではワードデータ
        print( str(i) + '行目')
        # 行の要素ごと
        for word in row:
            if j < 5:
                if word!='':
                    if i != len(words):# 最後以外の行
                        for word_next in words[i]:
                            if word_next!='':
                                # 相関値 計算
                                similarity_score = wv.similarity(word,word_next)
                                # 相関値 保存
                                my_dict[similarity_score] = (word,word_next)
                                all_dict[similarity_score] = (word,word_next)
                                # 相関値 表示
                                #print(word+' , '+word_next+' の相関： '+str(similarity_score))
                    else:#最後の行
                        print('last row')
            j = j + 1
        # 次の単語グループと一番相関のある単語ペア
        if i!= len(words):
            max_key = max(my_dict.keys())
            max_value = my_dict[max_key]
            print('**  一番強い相関は'+str(my_dict[max_key])+' で相関は '+str(max_key)+'  **')
            
    
    # 単語のリストを作成
    word_list = []
    for x in all_dict:
        word_list.append(all_dict[x][0])
        word_list.append(all_dict[x][1])
    word_list = list(set(word_list))

    # 相関行列の作成
    corr_matrix = np.zeros((len(word_list), len(word_list)))
    for x in all_dict:
        i = word_list.index(all_dict[x][0])
        j = word_list.index(all_dict[x][1])
        corr_matrix[i][j] = corr_matrix[j][i] = x
    
    return  words, word_list, corr_matrix, time_stamp

def make_path(words):
    '''
    各話題間について
    {w1, w2, w3, w4, w5}{now, next, corr}}
    {話題を代表するワード}{次の話題と最も相関が高い単語の組み合わせ}
    '''
    path_list = []
    
    for i in range(len(words)):
        out_line = []
            
        for w in words[i]:
            out_line.append(w)
            
        if i+1 == len(words): # 最後の行なら
            max_corr = [words[i][0], words[i][0], 1]
        else:
            # now と next の間の 相関が一番高い組み合わせ
            max_corr = [words[i][0], words[i+1][0], wv.similarity(words[i][0],words[i+1][0])] # それぞれの先頭を初期値として与える
            for now in words[i]:
                for next in words[i+1]:
                    if not (now=='' or next==''): # いずれも空でない場合
                        if max_corr[2] < wv.similarity(now,next): # now と next の方が相関が大きかったら
                            # 更新
                            max_corr = [now, next, wv.similarity(now,next)]
        out_line.append(max_corr)
        path_list.append(out_line)

    return path_list

def perf_line(start, end, topN, lim = 55, print_flag=True):
    '''
    start から end までの理想的な単語連想を行うリストを返す
    '''
    # vector
    start_v = wv[start]
    end_v = wv[end]
    # num
    num = lim
    # 
    s2e = []
    s2e.append(start)
    tmp = start
    if print_flag:
        print(f'ワードに近い{topN}件を探索\n')
        print(f'{start}:{wv.similarity(end, start)}')
    for i in range(num):
        # tmp の topN
        top = wv.most_similar(tmp,[],topN)
        # top から s2e との被り除去
        for w in s2e:
            ind = 0
            for t in top:
                if w == t[0]:
                    top.pop(ind)
                ind = ind + 1
        #print(top)
        # top N のうち END に一番近い奴
        near = top[0][0]
        for elm in top[1:]:
            # end 距離が最も近いもの
            if wv.similarity(end, near) < wv.similarity(end, elm[0]):
                near = elm[0]
        s2e.append(near)
        if print_flag:
            print(f'↓:{wv.similarity(near, tmp)}')
            print(f'{near}:{wv.similarity(end, near)}')
        
        tmp = near
        if tmp == end:
            if print_flag:
                print(f'\n')
            break
    s2e = np.array(s2e)
    # print(s2e)
    return s2e   

def hist(words, s2e):
    better_corr = []
    print(f'words {words}')
    for i in range(len(words)):
        if words[i] != words[-1]:# 最後じゃなかったら
            nows = words[i][:-2]
            nexts = words[i+1][:-2]
            max = 0
            for now in nows:
                for next in nexts:
                    if (now != '') and (next != ''):
                        tmp = wv.similarity(now, next)
                        if tmp > max:
                            max = tmp
            better_corr.append(max)
        else:
            pass
    
    better_corr = np.array(better_corr)
    print(f'better_corr{better_corr}')
    Fig, ax = plt.subplots()
    ax.hist(better_corr, bins=20, range=(0, 1))
    ax.set_xlabel('相関係数')
    ax.set_ylabel('度数')
    plt.xlim(0, 1)
    plt.ylim(0, 10)
    plt.savefig(f'img/hist_{s2e[0]}2{s2e[-1]}.png')
    plt.clf()
    plt.cla()
    plt.close()

def plot_kidou(path_list, s2e, time_stamp):
    # 実際の起動関連のリスト
    # 理想のリスト
    # 配信時間[分]
    kidou = []
    for line in path_list:
        kidou.append(line[5][0])
    print(f'{kidou}')
    
    x = np.arange(len(s2e))
    # start goal の 傾き
    start = s2e[0]
    end = s2e[-1]
    s2e_simi = wv.similarity(start, end)
    angle = (1 - s2e_simi) / (len(x)-1)
    y = x * angle + s2e_simi
    # 理想的な軌道Yを 作成
    y = []
    for val in s2e:
        y.append(wv.similarity(val, end))
    y = np.array(y)
    # 時間情報取り出し
    hour = int(time_stamp[-1][0])
    minutes = int(time_stamp[-1][1])
    # hour:min -> min
    minutes = hour*60 + minutes
    # x を 分 に 変換
    min_step = minutes / (len(x)-1)
    x = x * min_step
    # 実際の軌道用の xr を 分 で作成
    xr = []
    t = 0
    for time in time_stamp:
        
        tmp = int(time[0])*60 + int(time[1])
        # print(f'{t} :: {time[0]}:{time[1]} ->\t {tmp}[min] ')
        xr.append(tmp)
        t = t + 1
    xr = np.array(xr)
    # 実際の軌道用の yr を ゴールまでの相関係数で作成
    yr = []
    ends = path_list[-1][:5]
    for line in path_list:
        # line の 各単語について
        # ends を探索し 相関が一番大きい組み合わせ と 相関係数を取得
        pair = [line[0], ends[0], 0]
        for ind_line in range(5):
            if line[ind_line] != '':# word があるとき
                # ends 探索
                for e in ends:
                    if e != '':# word があるとき
                        tmp = wv.similarity(line[ind_line], e)
                        if tmp > pair[2]: # 相関係数大きければ
                            # pair 更新
                            pair = [line[ind_line], e, tmp]
        yr.append(pair[2])
    yr = np.array(yr)
    
    #print(f'path_list {path_list}')
    #print(f's2e {s2e}')
    #print(f'ends {ends}\n')
    t = 0
    for xrr, yrr in zip(xr,yr):
        print(f'{t}: {xrr}, {yrr}')
        t = t + 1
    #print(f'xr{len(xr)} {xr}')
    #print(f'yr{len(yr)} {yr}')
    # show
    Fig, ax = plt.subplots()
    ax.axline((0, (s2e_simi+yr[0])/2), (x[-1], 1), color='Cyan',linestyle=':', label='理想的な雑談軌道')
    #ax.plot(x, y, marker='o', c='blue', label='理想に近い雑談軌道の例')
    ax.plot(xr, yr, marker='o', c='magenta', label='実際の雑談軌道')
    #ax.scatter(xr[-1], yr[-1], marker='o', c='magenta', label='実際の雑談軌道')
    ax.set_xlim(0,x[-1])
    #ax.set_ylim(0,1)
    ax.set_xlabel('時間経過[min]')
    ax.set_ylabel('最後の話題との相関係数')
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02,), borderaxespad=0, ncol=2)
    #plt.show()
    ax.set_box_aspect(0.6) # 小さくすれば横長
    plt.savefig(f'img/plot_{start}2{end}_.png')
    plt.clf()
    plt.cla()
    plt.close()
    

def main():
    # CSVfile パス
    # csv_file_path = 'words.csv'
    csv_path = '240728.csv'
    #csv_path = 'words.csv'
    
    words, word_list, corr_matrix, time_stamp = read_csv(csv_path)
    path_list = make_path(words)
    
    
    if csv_path=='240728.csv':
        s2e = perf_line('喜び', '全員', 24 , 27) # 最後2個の引数は調整要素
        s2e_ = perf_line(s2e[-1],s2e[-2], 10 , 27) # 最後2個の引数は調整要素
        s2e = s2e[:-2].tolist()
        for i in range(len(s2e_)):
            s2e.append(s2e_[-1-i])
        s2e = np.array(s2e)
    
    elif csv_path=='words.csv':
        s2e = perf_line('WBC', '周年', 1370 , 27)
        
    else: 
        return 0
    
    hist(words, s2e)
    
    plot_kidou(path_list, s2e, time_stamp)
    
    print(s2e)


if __name__ == "__main__":
    main()