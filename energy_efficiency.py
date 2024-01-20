# -*- coding: utf-8 -*-
"""
빅데이터 기말프로젝트: 건물의 에너지 효율(냉/난방 부하) 평가

@author: 20190962 컴퓨터공학과 김수현
"""
# In[1]: 워드클라우드 생성

import json
import re
from konlpy.tag import Okt
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

inputFileName = '난방비_naver_news'
data = json.loads(open(inputFileName+'.json', 'r', encoding='utf-8').read())

description = ''

for item in data:
    if isinstance(item, dict) and 'description' in item.keys():
        description = description + re.sub(r'[^\w]', ' ', item['description']) + ''

nlp = Okt()
description_N = nlp.nouns(description)

count = Counter(description_N)
word_count = dict()

########################
for tag, counts in count.most_common(80):
    if(len(str(tag))>1):
        word_count[tag] = counts
        print("%s : %d" % (tag, counts))
########################
font_path = "c:/Windows/fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname = font_path).get_name()
matplotlib.rc('font', family=font_name)

wc = WordCloud(font_path, background_color='ivory', width=800, height=600)
cloud=wc.generate_from_frequencies(word_count)

plt.figure(figsize=(8,8))
plt.imshow(cloud)
plt.axis('off')
plt.show()

# In[2]: 상관분석

import pandas as pd
import seaborn as sns
import numpy as np

EE = pd.read_csv('energy_efficiency_data.csv', header=0, engine='python')
EE.info()
EE.isnull().sum() #비어있는 데이터 있는지 확인 => 없음
#상관계수 계산
EE_corr = EE.corr(method = 'pearson').round(10) #소숫점 아래 10자리에서 반올림한다.
#########################히트맵으로 시각화
colormap = plt.cm.RdBu
sns.heatmap(EE_corr, linewidths = 0.1, vmax = 1.0, square = True, cmap = colormap, linecolor = 'white', annot = True)
########################

# In[3]: 회귀분석

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data_df = pd.read_csv('energy_efficiency_data.csv', header=0, engine='python')

Y = data_df[['HeatingLoad', 'CoolingLoad']]
X = data_df.drop(['Orientation', 'GlazingAreaDistribution', 'HeatingLoad', 'CoolingLoad'], axis=1, inplace=False)

#################################################
# test_size를 8:2로 했을때 모델평가
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))

# test_size를 7:3로 했을때 모델평가
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)
mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse) ##############
print('MSE : {0:.3f}, RMSE : {1:.3f}'.format(mse, rmse))
print('R^2(Variance score) : {0:.3f}'.format(r2_score(Y_test, Y_predict)))
#################################################

print('Y 절편 값: ',  np.round(lr.intercept_, 2))
print('회귀 계수 값: ', np.round(lr.coef_, 2))
#7:3이 성능이 좋음. test_size=0.3으로 선택.

#회귀 계수 값 큰 항목 확인 
coef = pd.Series(data=np.round(lr.coef_[0], 2), index=X.columns)
coef.sort_values(ascending=False)

# 산점도 + 선형 회귀 그래프로 시각화
fig, axs = plt.subplots(figsize=(20, 20), ncols=3, nrows=2)
x_features = ['RelativeCompactness', 'SurfaceArea', 'WallArea', 'RoofArea', 'OverallHeight', 'GlazingArea']
plot_color = ['r', 'b', 'y', 'c', 'm', 'brown']
for i, feature in enumerate(x_features):
      row = int(i/3)
      col = i%3
      sns.regplot(x=feature, y='HeatingLoad', data=data_df, ax=axs[row][col], color=plot_color[i])
      
####################################
#에너지 효율 예측      
print("난방/냉방 부하를 예측하고 싶은 건물의 정보를 입력해주세요.")
X1 = float(input("Relative Compactness : "))
X2 = float(input("Surface Area : "))
X3 = float(input("Wall Area : "))
X4 = float(input("Roof Area : "))
X5 = float(input("Overall Height : "))
X6 = float(input("Glazing Area : "))

Load_predict = lr.predict([[X1, X2, X3, X4, X5, X6]])

print("이 건물의 예상 난방 부하는 %.4f 입니다." %Load_predict[0, 0])
print("이 건물의 예상 냉방 부하는 %.4f 입니다." %Load_predict[0, 1])    
      
      
      
      
      