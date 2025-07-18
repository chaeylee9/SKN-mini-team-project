import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

titanic = pd.read_csv('C:/Users/yongchae/Desktop/SKN/titanic_kor.csv')

# 결측치 비율 70% 이상이면 컬럼 삭제
missing_ratio = titanic.isnull().mean()
columns_to_drop = missing_ratio[missing_ratio > 0.7].index.tolist()

titanic_drop = titanic.drop(columns=columns_to_drop, axis=1)

# 쓸 컬럼만 남겨
titanic_drop = titanic_drop.drop(columns = ['등급', '성별', '성인남성', '탑승지명', '생존여부'])

# 결측치 대체
mode_val = titanic_drop['탑승지코드'].mode()[0]
titanic_drop['탑승지코드'] = titanic_drop['탑승지코드'].fillna(mode_val)

children_age_mean = titanic_drop.loc[titanic_drop['성인여부'] == 'child', '나이'].mean()
adult_age_mean = titanic_drop.loc[titanic_drop['성인여부'].isin(['man', 'woman']), '나이'].mean()

def fill_age(row):
    if pd.isna(row['나이']):
        if row['성인여부'] == 'child':
            return children_age_mean
        elif row['성인여부'] in ['man', 'woman']:
            return adult_age_mean
    return row['나이']

titanic_drop['나이'] = titanic_drop.apply(fill_age, axis=1)

# 변수 인코딩
def dummy_encoding(df, col_list):
    cate_df = titanic_drop[col_list]

    df_enc = pd.get_dummies(cate_df)

    return df_enc

cate_columns = ['탑승지코드', '객실등급', '성인여부']

cate_df = dummy_encoding(titanic_drop, cate_columns)

train_df = pd.concat([cate_df, titanic_drop[['개인탑승자', '나이', '형제자매', '부모자녀', '요금']]], axis=1)
test_df = titanic_drop['생존']

from sklearn.model_selection import train_test_split

# y_titanic_df = df_train['Survived']
# X_titanic_df = df_train.drop('Survived', axis = 1, inplace = False)
X_train, X_test, y_train, y_test = train_test_split(train_df, titanic_drop['생존'], test_size = 0.2, 
                                                    stratify=titanic_drop['생존'], random_state = 11)



def find_best_model(X_train, X_test, y_train, y_test):

    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score


    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=11),
        'RandomForestClassifier': RandomForestClassifier(random_state=11),
        'LogisticRegression': LogisticRegression(random_state=11, max_iter=1000)
    }

    accuracies = {}

    for name, model in models.items():
        # 스케일링이 필요한 모델만 스케일 적용
        if name in ['KNeighborsClassifier']:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        accuracies[name] = acc
        print(f'{name} 정확도 : {acc:.4f}')

    # 정확도 기준으로 베스트 모델 선정
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    print(f'\n>>> 베스트 모델은 "{best_model_name}"이며 정확도는 {accuracies[best_model_name]:.4f}입니다.')

    return best_model_name, best_model, accuracies

best_name, best_model, model_accuracies = find_best_model(X_train, X_test, y_train, y_test)


y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]  # ROC-AUC 계산용


from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# 평가 지표 출력
print(f"✅ 정확도: {accuracy_score(y_test, y_pred):.4f}")
print(f"✅ 정밀도: {precision_score(y_test, y_pred):.4f}")
print(f"✅ 재현율: {recall_score(y_test, y_pred):.4f}")
print(f"✅ F1 스코어: {f1_score(y_test, y_pred):.4f}")
print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print("\n✅ 분류 리포트:\n", classification_report(y_test, y_pred))

# Confusion Matrix 시각화
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("예측값")
plt.ylabel("실제값")
plt.title("Confusion Matrix")
plt.show()

# 전체 데이터 다시 훈련
best_model.fit(train_df, titanic_drop['생존'])

# 테스트 데이터 생성
def make_test():
    made_test = pd.DataFrame([{
        '탑승지코드_C': False,
        '탑승지코드_Q': False,
        '탑승지코드_S': False,
        '객실등급_First': False,
        '객실등급_Second': False,
        '객실등급_Third': False,
        '성인여부_child': False,
        '성인여부_man': False,
        '성인여부_woman': False,
        '개인탑승자': True,
        '나이': 0,
        '형제자매': 0,
        '부모자녀': 0,
        '요금': 0.0
    }])

    # 나이 입력
    try:
        age = int(input("나이 입력: "))
        made_test.at[0, '나이'] = age
    except:
        print("나이는 정수로 입력해야 합니다. 기본값 0 유지")
        age = 0

    # 성인여부 결정
    if age <= 18:
        # 무조건 child
        made_test.at[0, '성인여부_man'] = False
        made_test.at[0, '성인여부_woman'] = False
        made_test.at[0, '성인여부_child'] = True
    else:
        # 19세 이상은 man 또는 woman 입력 받기
        adult_status = input("성인여부 (man, woman) 입력: ").strip().lower()
        if adult_status in ['man', 'woman']:
            made_test.at[0, '성인여부_man'] = (adult_status == 'man')
            made_test.at[0, '성인여부_woman'] = (adult_status == 'woman')
            made_test.at[0, '성인여부_child'] = False
        else:
            print("성인여부는 man 또는 woman이어야 합니다. 기본값 유지")

    # 탑승지코드 입력받기
    boarding_code = input("탑승지코드 (S, C, Q) 입력: ").strip().upper()
    if boarding_code in ['C', 'S', 'Q']:
        for code in ['C', 'S', 'Q']:
            made_test.at[0, f'탑승지코드_{code}'] = (boarding_code == code)
    else:
        print("탑승지코드는 S, C, Q 중 하나여야 합니다. 기본값 유지")

    # 객실등급 입력받기
    room_grade = input("객실등급 (First, Second, Third) 입력: ").strip().capitalize()
    if room_grade in ['First', 'Second', 'Third']:
        for grade in ['First', 'Second', 'Third']:
            made_test.at[0, f'객실등급_{grade}'] = (room_grade == grade)
    else:
        print("객실등급은 First, Second, Third 중 하나여야 합니다. 기본값 유지")

    # 나머지 입력
    try:
        siblings = int(input("형제자매 수 입력: "))
        made_test.at[0, '형제자매'] = siblings
    except:
        print("형제자매 수는 정수로 입력해야 합니다. 기본값 유지")

    try:
        parents = int(input("부모자녀 수 입력: "))
        made_test.at[0, '부모자녀'] = parents
    except:
        print("부모자녀 수는 정수로 입력해야 합니다. 기본값 유지")

    try:
        fare = float(input("요금 입력: "))
        made_test.at[0, '요금'] = fare
    except:
        print("요금은 숫자로 입력해야 합니다. 기본값 유지")

    return made_test


df_test = make_test()

y_pred = best_model.predict(df_test)

if y_pred[0] == 1:
    print('테스트 데이터 생존')
else:
    print('테스트 데이터 사망')