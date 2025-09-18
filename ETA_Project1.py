## Data Preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
#import movingpandas as mpd
from shapely.geometry import Point
import numpy as np
import os
file_path = r"C:\Users\slab\Desktop\Slab Project\Stage2 ETA\Raw Data\477769500_VesselHistoryLineInfo.csv"

try:
    df
except NameError:
    df = pd.read_csv(file_path, low_memory=False)
df = pd.read_csv(file_path, low_memory=False)
# 我只想畫出477769500這艘船的航線圖，先把其他的洗掉
mmsi_target = 477769500
df_filtered = df[df['MMSI'] == mmsi_target].copy()
print(len(df_filtered))
print(df_filtered.head())
print("有效data數量約為:", len(df_filtered)/len(df))
df_filtered = df_filtered[
    (df_filtered['Lat'] >= -90) & (df_filtered['Lat'] <= 90) &
    (df_filtered['Lng'] >= -180) & (df_filtered['Lng'] <= 180)
].copy()

# 再檢查經緯度是否都合理
print(df_filtered[['Lat', 'Lng']].describe())
df_filtered['Lat'] = df_filtered['Lat'].astype(float)
df_filtered['Lng'] = df_filtered['Lng'].astype(float)
df_filtered['CreateTime'] = pd.to_datetime(df_filtered['CreateTime'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['Lat', 'Lng', 'CreateTime'])
# 為了不讓圖片上線斷掉，將經度[-180, 180] 轉成 [0, 360]
df_filtered['Lng_360'] = df_filtered['Lng'] % 360
# 計算地圖中心
map_center_lat = df_filtered['Lat'].mean()
map_center_lon = df_filtered['Lng_360'].mean()
df_filtered = df_filtered.sort_values('CreateTime').reset_index(drop=True)
## Trajectory Reconstruction
m = folium.Map(
    location=[map_center_lat, map_center_lon],
    zoom_start=4,
    tiles='OpenStreetMap'
)
import folium

coords = df_filtered[['Lat', 'Lng_360']].iloc[::1].values.tolist()  # ::10 可抽樣10倍
folium.PolyLine(
    coords,
    color='blue',
    weight=3,
    opacity=0.7,
    popup=f"航線點數: {len(coords)}"
).add_to(m)

# 起訖點標記

if len(coords) > 0:
    # 起點
    folium.Marker(
        location=coords[0],
        popup=f"起點\n時間: {df_filtered.iloc[0]['CreateTime']}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    # 終點
    folium.Marker(
        location=coords[-1],
        popup=f"終點\n時間: {df_filtered.iloc[-1]['CreateTime']}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
html_file = "ship_trajectory_360.html"
m.save(html_file)
# 自動用瀏覽器開啟
import webbrowser
webbrowser.open('file://' + os.path.realpath(html_file))
## DBSCAN clustering
#from sklearn.cluster import DBSCAN
#from haversine import haversine
""" stop_df = df_filtered[df_filtered["Sog"] < 0.5].copy()
len(stop_df) """

""" ## DBSCAN clustering (前半航程)
# 取前半段資料
half_index = len(df_filtered) // 2
df_half = df_filtered.iloc[:half_index].copy()

# 篩選停泊點 (假設 Sog 欄位存在)
stop_df = df_half[df_half["Sog"] < 0.5].copy()

if stop_df.empty:
    print("前半航程沒有偵測到停泊點")
else:
    # 經緯度轉弧度
    coords = stop_df[['Lat', 'Lng_360']].to_numpy()
    coords_rad = np.radians(coords)

    # DBSCAN 聚類
    # eps 調整成合適值（弧度）
    db = DBSCAN(eps=0.001, min_samples=5, metric='haversine').fit(coords_rad)
    stop_df['cluster'] = db.labels_

    # 計算 cluster 中心與半徑
    port_list = []
    for cluster_id in stop_df['cluster'].unique():
        if cluster_id == -1:
            continue  # 忽略噪聲點

        cluster_points = stop_df[stop_df['cluster'] == cluster_id][['Lat', 'Lng_360']]
        center_lat = cluster_points['Lat'].mean()
        center_lon = cluster_points['Lng_360'].mean()

        distances = cluster_points.apply(
            lambda row: haversine((center_lat, center_lon), (row['Lat'], row['Lng_360'])),
            axis=1
        )
        radius_km = distances.max()

        port_list.append({
            'MMSI': mmsi_target,
            'cluster': cluster_id,
            'lat': center_lat,
            'lon': center_lon,
            'radius_km': radius_km
        })

    port_df = pd.DataFrame(port_list)
    print(f"共偵測到 {port_df.shape[0]} 個停泊點 / 港口 (前半航程)")
    print(port_df.head())
"
 """
## Mooring state classification + Add New Columns
import numpy as np
import pandas as pd

# -----------------------------------
# 確保時間排序
# -----------------------------------
df_filtered = df_filtered.sort_values('CreateTime').reset_index(drop=True)

# -----------------------------------
# 計算經緯度差分
# -----------------------------------
df_filtered['Delta_Lat'] = df_filtered['Lat'].diff().abs()
df_filtered['Delta_Long'] = df_filtered['Lng_360'].diff().abs()

# 計算時間差（秒）
df_filtered['Delta_Time'] = df_filtered['CreateTime'].diff().dt.total_seconds()

# -----------------------------------
# 閾值設定
# -----------------------------------
short_time_thresh = 1.0   # 秒
sog_threshold = 0.5       # knots
rate_threshold = 0.0001   # 經緯度變化閾值

# 修正 delta 經緯度
df_filtered['Adj_Delta_Lat'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Lat'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Lat']
)
df_filtered['Adj_Delta_Long'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Long'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Long']
)

# -----------------------------------
# 停泊判斷
# -----------------------------------
df_filtered['Is_Stop'] = (
    (df_filtered['Sog'] < sog_threshold) &
    (df_filtered['Adj_Delta_Lat'].abs() < rate_threshold) &
    (df_filtered['Adj_Delta_Long'].abs() < rate_threshold)
)

# -----------------------------------
# 找停泊區段（容忍短暫中斷）
# -----------------------------------
max_gap_sec = 120  # 允許 2 分鐘內 False 不切斷停泊

stop_segments = []
current_start_idx = None
last_stop_idx = None

for i in range(len(df_filtered)):
    if df_filtered.loc[i, 'Is_Stop']:
        if current_start_idx is None:
            current_start_idx = i
        last_stop_idx = i
    else:
        if current_start_idx is not None:
            gap = (df_filtered.loc[i, 'CreateTime'] - df_filtered.loc[last_stop_idx, 'CreateTime']).total_seconds()
            if gap > max_gap_sec:
                segment_idx = range(current_start_idx, last_stop_idx + 1)
                stop_segments.append(segment_idx)
                current_start_idx = None
                last_stop_idx = None

# 最後一段停泊若持續到資料尾端
if current_start_idx is not None:
    segment_idx = range(current_start_idx, last_stop_idx + 1)
    stop_segments.append(segment_idx)

# -----------------------------------
# 過濾短暫停泊 (<30分鐘)
# -----------------------------------
min_stop_sec = 1800  # 30 分鐘
filtered_segments = []
for segment in stop_segments:
    start_time = df_filtered.loc[segment[0], 'CreateTime']
    end_time = df_filtered.loc[segment[-1], 'CreateTime']
    duration = (end_time - start_time).total_seconds()
    if duration >= min_stop_sec:
        filtered_segments.append(segment)

# -----------------------------------
# 航程標註與 Real_ETA_sec
# -----------------------------------
df_filtered['voyage_id'] = np.nan
df_filtered['Real_ETA_sec'] = np.nan  # 剩餘到港時間（秒）

voyage_id = 1
for i in range(len(filtered_segments) - 1):
    # 航程 = 當前停泊區段的下一點到下一停泊區段的起點
    start_idx = filtered_segments[i][-1] + 1
    end_idx = filtered_segments[i+1][0] - 1
    if start_idx > end_idx:
        continue  # 沒有航程點就跳過

    df_filtered.loc[start_idx:end_idx, 'voyage_id'] = voyage_id

    # Real_ETA_sec = 下一停泊區段時間均值 - 當前點時間
    eta_segment = filtered_segments[i+1]
    eta_time = df_filtered.loc[eta_segment, 'CreateTime'].mean()

    df_filtered.loc[start_idx:end_idx, 'Real_ETA_sec'] = (
        (eta_time - df_filtered.loc[start_idx:end_idx, 'CreateTime']).dt.total_seconds()
    )

    voyage_id += 1

# 可選：將第一段航程沒有停泊前的點標記為 voyage_id = 1
if len(filtered_segments) > 1 and filtered_segments[0][0] > 0:
    start_idx = 0
    end_idx = filtered_segments[0][-1]
    df_filtered.loc[start_idx:end_idx, 'voyage_id'] = 1
    eta_time = df_filtered.loc[filtered_segments[1], 'CreateTime'].mean()
    df_filtered.loc[start_idx:end_idx, 'Real_ETA_sec'] = (
        (eta_time - df_filtered.loc[start_idx:end_idx, 'CreateTime']).dt.total_seconds()
    )

# -----------------------------------
# 檢查結果
# -----------------------------------
print(df_filtered[['CreateTime','Lat','Lng_360','Sog','Is_Stop','voyage_id','Real_ETA_sec']].head(20))

# 檢查停泊段數量與詳細資訊
print(f"總共停泊段數: {len(filtered_segments)}\n")

stop_info = []
for i, segment in enumerate(filtered_segments, 1):
    start_time = df_filtered.loc[segment[0], 'CreateTime']
    end_time = df_filtered.loc[segment[-1], 'CreateTime']
    duration_sec = (end_time - start_time).total_seconds()
    stop_info.append({
        'Stop_ID': i,
        'StartTime': start_time,
        'EndTime': end_time,
        'Duration_sec': duration_sec,
        'Num_Points': len(segment)
    })

stops_df = pd.DataFrame(stop_info)
print(stops_df)

# 看看哪些行是 voyage_id 為 NaN（通常是停泊點）
df_filtered.loc[df_filtered['voyage_id'].isna(), ['CreateTime','Is_Stop','voyage_id','Real_ETA_sec']].tail(10)

#篩掉點數小於10的航程
# -----------------------------------
# 篩選航程點數 >= 10
# -----------------------------------
voyage_counts = df_filtered.groupby('voyage_id').size()

# 找出點數 >= 10 的 voyage_id
valid_voyages = voyage_counts[voyage_counts >= 10].index

# 只保留有效航程
df_filtered = df_filtered[df_filtered['voyage_id'].isin(valid_voyages)].copy()

# 丟掉 voyage_id 或 Real_ETA_sec 為 NaN 的資料
df_filtered = df_filtered.dropna(subset=['voyage_id', 'Real_ETA_sec']).reset_index(drop=True)

# -----------------------------------
# 檢查結果
# -----------------------------------
print("篩選後資料數量:", len(df_filtered))
print(df_filtered[['CreateTime','Lat','Lng_360','Sog','Is_Stop','voyage_id','Real_ETA_sec']].head(20))

df_filtered["Destination"].value_counts()
## For Testing
import numpy as np
import pandas as pd
from haversine import haversine

# -----------------------------------
# 確保時間排序
# -----------------------------------
df_filtered = df_filtered.sort_values('CreateTime').reset_index(drop=True)

# -----------------------------------
# 先清理 SOG 異常值
# -----------------------------------
sog_min, sog_max = 0, 50  # 合理範圍
before_len = len(df_filtered)
df_filtered = df_filtered[(df_filtered['Sog'] >= sog_min) & (df_filtered['Sog'] <= sog_max)].reset_index(drop=True)
after_len = len(df_filtered)
print(f"[INFO] 移除異常 SOG 筆數: {before_len - after_len}, 剩餘: {after_len}")

# -----------------------------------
# 計算經緯度差分
# -----------------------------------
df_filtered['Delta_Lat'] = df_filtered['Lat'].diff().abs()
df_filtered['Delta_Long'] = df_filtered['Lng_360'].diff().abs()

# 計算時間差（秒）
df_filtered['Delta_Time'] = df_filtered['CreateTime'].diff().dt.total_seconds()

# -----------------------------------
# 閾值設定
# -----------------------------------
short_time_thresh = 1.0   # 秒
sog_threshold = 0.5       # knots
rate_threshold = 0.0001   # 經緯度變化閾值

# 修正 delta 經緯度
df_filtered['Adj_Delta_Lat'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Lat'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Lat']
)
df_filtered['Adj_Delta_Long'] = np.where(
    df_filtered['Delta_Time'] > short_time_thresh,
    df_filtered['Delta_Long'] / df_filtered['Delta_Time'],
    df_filtered['Delta_Long']
)

# -----------------------------------
# 停泊判斷
# -----------------------------------
df_filtered['Is_Stop'] = (
    (df_filtered['Sog'] < sog_threshold) &
    (df_filtered['Adj_Delta_Lat'].abs() < rate_threshold) &
    (df_filtered['Adj_Delta_Long'].abs() < rate_threshold)
)

# -----------------------------------
# 找停泊區段（容忍短暫中斷）
# -----------------------------------
max_gap_sec = 120  # 允許 2 分鐘內 False 不切斷停泊

stop_segments = []
current_start_idx = None
last_stop_idx = None

for i in range(len(df_filtered)):
    if df_filtered.loc[i, 'Is_Stop']:
        if current_start_idx is None:
            current_start_idx = i
        last_stop_idx = i
    else:
        if current_start_idx is not None:
            gap = (df_filtered.loc[i, 'CreateTime'] - df_filtered.loc[last_stop_idx, 'CreateTime']).total_seconds()
            if gap > max_gap_sec:
                segment_idx = range(current_start_idx, last_stop_idx + 1)
                stop_segments.append(segment_idx)
                current_start_idx = None
                last_stop_idx = None

# 最後一段停泊若持續到資料尾端
if current_start_idx is not None:
    segment_idx = range(current_start_idx, last_stop_idx + 1)
    stop_segments.append(segment_idx)

# -----------------------------------
# 過濾短暫停泊 (<30分鐘)
# -----------------------------------
min_stop_sec = 1800  # 30 分鐘
filtered_segments = []
for segment in stop_segments:
    start_time = df_filtered.loc[segment[0], 'CreateTime']
    end_time = df_filtered.loc[segment[-1], 'CreateTime']
    duration = (end_time - start_time).total_seconds()
    if duration >= min_stop_sec:
        filtered_segments.append(segment)

# -----------------------------------
# 航程標註與 Real_ETA_sec
# -----------------------------------
df_filtered['voyage_id'] = np.nan
df_filtered['Real_ETA_sec'] = np.nan  # 剩餘到港時間（秒）

voyage_id = 1
for i in range(len(filtered_segments) - 1):
    # 航程 = 當前停泊區段的下一點到下一停泊區段的起點
    start_idx = filtered_segments[i][-1] + 1
    end_idx = filtered_segments[i+1][0] - 1
    if start_idx > end_idx:
        continue  # 沒有航程點就跳過

    df_filtered.loc[start_idx:end_idx, 'voyage_id'] = voyage_id

    # Real_ETA_sec = 下一停泊區段時間均值 - 當前點時間
    eta_segment = filtered_segments[i+1]
    eta_time = df_filtered.loc[eta_segment, 'CreateTime'].min()

    df_filtered.loc[start_idx:end_idx, 'Real_ETA_sec'] = (
        (eta_time - df_filtered.loc[start_idx:end_idx, 'CreateTime']).dt.total_seconds()
    )

    voyage_id += 1

# -----------------------------------
# 新增「距離目的港」與「靠港判定」
# -----------------------------------
df_filtered['dist_to_dest_km'] = np.nan
df_filtered['near_port_flag'] = np.nan

# 建立 voyage_id → 目的港座標 mapping
dest_map = {}
for v in range(1, voyage_id):
    if v < len(filtered_segments):
        seg = filtered_segments[v]  # 下一停泊
        dest_lat = df_filtered.loc[seg, 'Lat'].mean()
        dest_lon360 = df_filtered.loc[seg, 'Lng_360'].mean()
        # 把 0-360 轉 -180~180
        dest_lon = dest_lon360 if dest_lon360 <= 180 else dest_lon360 - 360
        dest_map[v] = (dest_lat, dest_lon)

# 計算距離
for idx, row in df_filtered.iterrows():
    v = row['voyage_id']
    if pd.notna(v):
        v = int(v)
        if v in dest_map:
            lat1 = row['Lat']
            lon1 = row['Lng_360']
            lon1 = lon1 if lon1 <= 180 else lon1 - 360
            lat2, lon2 = dest_map[v]
            df_filtered.at[idx, 'dist_to_dest_km'] = haversine((lat1, lon1), (lat2, lon2))

# 判定靠港 flag
df_filtered.loc[df_filtered['dist_to_dest_km'] < 1.0, 'near_port_flag'] = 1.0
df_filtered.loc[(df_filtered['dist_to_dest_km'] >= 1.0) & (df_filtered['dist_to_dest_km'] <= 5.0), 'near_port_flag'] = 0.5
df_filtered.loc[df_filtered['dist_to_dest_km'] > 5.0, 'near_port_flag'] = 0.0

# -----------------------------------
# 檢查結果
# -----------------------------------
print(df_filtered[['CreateTime','Lat','Lng_360','Sog','Is_Stop',
                   'voyage_id','Real_ETA_sec',
                   'dist_to_dest_km','near_port_flag']].head(20))


#篩掉點數小於10的航程
# -----------------------------------
# 篩選航程點數 >= 10
# -----------------------------------
#voyage_counts = df_filtered.groupby('voyage_id').size()

# 找出點數 >= 10 的 voyage_id
#valid_voyages = voyage_counts[voyage_counts >= 10].index

# 只保留有效航程
#df_filtered = df_filtered[df_filtered['voyage_id'].isin(valid_voyages)].copy()

# 丟掉 voyage_id 或 Real_ETA_sec 為 NaN 的資料
df_filtered = df_filtered.dropna(subset=['voyage_id', 'Real_ETA_sec']).reset_index(drop=True)

# -----------------------------------
# 檢查結果
# -----------------------------------
print("篩選後資料數量:", len(df_filtered))
print(df_filtered[['CreateTime','Lat','Lng_360','Sog','Is_Stop','voyage_id','Real_ETA_sec']].head(20))
df_filtered['near_port_flag'].value_counts()

## Random Forest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# 選特徵 (示範可選)
# -----------------------------
numeric_features = [
    'Lat', 'Lng_360', 'Sog','near_port_flag'
]

X = df_filtered[numeric_features]
y = df_filtered['Real_ETA_sec']

# -----------------------------
# 切分訓練與測試集
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 訓練模型
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,  # 可調整
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# -----------------------------
# 預測
# -----------------------------
y_pred = rf_model.predict(X_test)

# -----------------------------
# 評估
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} sec")
print(f"RMSE: {rmse:.2f} sec")
print(f"R2: {r2:.3f}")

# -----------------------------
# 預測示例
# -----------------------------
df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'] = y_pred
print(df_filtered[['CreateTime','voyage_id','Real_ETA_sec','Pred_Real_ETA_sec']].head(20))

test_results = X_test.copy()
test_results['Real_ETA_sec'] = y_test
test_results['Pred_Real_ETA_sec'] = y_pred
print(test_results.head(20))
import matplotlib.pyplot as plt

# -----------------------------
# 安全填回 Pred_Real_ETA_sec
# -----------------------------
# 確保索引對齊
df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'] = y_pred

# 檢查 NaN
print("Pred_Real_ETA_sec NaN 數量:", df_filtered['Pred_Real_ETA_sec'].isna().sum())

# -----------------------------
# 畫散點圖：實際 ETA vs 預測 ETA
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df_filtered.loc[X_test.index, 'Real_ETA_sec'],
            df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'],
            alpha=0.5, s=10, color='blue')
plt.plot([0, df_filtered['Real_ETA_sec'].max()],
         [0, df_filtered['Real_ETA_sec'].max()],
         color='red', linestyle='--', linewidth=1)  # 理想對角線
plt.xlabel('Actual Real_ETA_sec (s)')
plt.ylabel('Predicted Real_ETA_sec (s)')
plt.title('Actual ETA vs Predicted ETA Scatter Plot')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 特徵重要性
# -----------------------------
importances = rf_model.feature_importances_
feature_names = X.columns

# 排序 (可視化時比較好看)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# -----------------------------
# 選特徵 (示範可選)
# -----------------------------
numeric_features = [
    'Lat', 'Lng_360', 'Sog', 'dist_to_dest_km', 'near_port_flag'
]

X = df_filtered[numeric_features]
y = df_filtered['Real_ETA_sec']

# -----------------------------
# 切分訓練與測試集
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 訓練模型
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,  # 可調整
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

# -----------------------------
# 預測
# -----------------------------
y_pred = rf_model.predict(X_test)

# -----------------------------
# 評估
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} sec")
print(f"RMSE: {rmse:.2f} sec")
print(f"R2: {r2:.3f}")

# -----------------------------
# 預測示例
# -----------------------------
df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'] = y_pred
print(df_filtered[['CreateTime','voyage_id','Real_ETA_sec','Pred_Real_ETA_sec']].head(20))

import matplotlib.pyplot as plt

# -----------------------------
# 安全填回 Pred_Real_ETA_sec
# -----------------------------
# 確保索引對齊
df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'] = y_pred

# 檢查 NaN
print("Pred_Real_ETA_sec NaN 數量:", df_filtered['Pred_Real_ETA_sec'].isna().sum())

# -----------------------------
# 畫散點圖：實際 ETA vs 預測 ETA
# -----------------------------
plt.figure(figsize=(8,6))
plt.scatter(df_filtered.loc[X_test.index, 'Real_ETA_sec'],
            df_filtered.loc[X_test.index, 'Pred_Real_ETA_sec'],
            alpha=0.5, s=10, color='blue')
plt.plot([0, df_filtered['Real_ETA_sec'].max()],
         [0, df_filtered['Real_ETA_sec'].max()],
         color='red', linestyle='--', linewidth=1)  # 理想對角線
plt.xlabel('Actual Real_ETA_sec (s)')
plt.ylabel('Predicted Real_ETA_sec (s)')
plt.title('Actual ETA vs Predicted ETA Scatter Plot')
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 特徵重要性
# -----------------------------
importances = rf_model.feature_importances_
feature_names = X.columns

# 排序 (可視化時比較好看)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# 分航程類型
short_mask = df_filtered['Real_ETA_sec'] <= 86400   # 24 小時內為短航程
long_mask = df_filtered['Real_ETA_sec'] > 86400     # 超過 24 小時為長航程

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_group(mask, label):
    y_true = df_filtered.loc[mask, 'Real_ETA_sec']
    y_pred = df_filtered.loc[mask, 'Pred_Real_ETA_sec']
    
    # 過濾 NaN
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"=== {label} ===")
    print(f"MAE: {mae:.2f} sec")
    print(f"RMSE: {rmse:.2f} sec")
    print(f"R2: {r2:.3f}\n")


# 評估
evaluate_group(short_mask, "短航程")
evaluate_group(long_mask, "長航程")

import matplotlib.pyplot as plt
import numpy as np

# 只取有預測值的資料
df_plot = df_filtered.dropna(subset=['Pred_Real_ETA_sec', 'Real_ETA_sec']).copy()

# 設定靠近港口的閾值，例如 1 小時內
close_port_thresh = 3600  # 秒

# 分兩種顏色：靠近港口 vs 其他
df_plot['Color'] = np.where(df_plot['Real_ETA_sec'] <= close_port_thresh, 'red', 'blue')

plt.figure(figsize=(10, 6))
plt.scatter(df_plot['Real_ETA_sec'], df_plot['Pred_Real_ETA_sec'], 
            c=df_plot['Color'], alpha=0.5, s=20)

# 畫理想線 y=x
max_eta = df_plot[['Real_ETA_sec', 'Pred_Real_ETA_sec']].max().max()
plt.plot([0, max_eta], [0, max_eta], color='black', linestyle='--', label='Ideal')

plt.xlabel('Actual Real_ETA_sec')
plt.ylabel('Predicted Real_ETA_sec')
plt.title('Random Forest: Actual vs Predicted ETA')
plt.legend(['Ideal line', 'Other points', 'Close to port'])
plt.ylim(0, max_eta)
plt.xlim(0, max_eta)
plt.show()

# 複製一份 df_filtered
df_plot = df_filtered.copy()

# 對全部資料做預測
numeric_features = ['Lat', 'Lng_360', 'Sog', 'near_port_flag']
X_all = df_plot[numeric_features]

# 使用已訓練好的模型做預測
df_plot['Pred_Real_ETA_sec'] = rf_model.predict(X_all)

# 檢查前幾筆
df_plot[['CreateTime', 'voyage_id', 'Real_ETA_sec', 'Pred_Real_ETA_sec']].head(10)

# 條件：實際ETA < 10000，預測ETA > 30000
condition_idx = df_plot[
    (df_plot['Real_ETA_sec'] < 50000) &
    (df_plot['Pred_Real_ETA_sec'] > 100000)
].index

# 印出每個點及其前後五筆資料
for idx in condition_idx:
    start = max(idx - 5, 0)
    end = min(idx + 5 + 1, len(df_plot))  # +1 因為slice不包含end
    print(f"\n=== Index {idx} (Real_ETA_sec={df_plot.loc[idx, 'Real_ETA_sec']:.1f}, "
          f"Pred_Real_ETA_sec={df_plot.loc[idx, 'Pred_Real_ETA_sec']:.1f}) ===")
    display(df_plot.loc[start:end, ['CreateTime','Lat','Lng_360','Sog',
                                        'voyage_id','Real_ETA_sec','Pred_Real_ETA_sec']])

## Time To Predict!
#先建立換座標程式碼
#目前資訊 2025/9/17 /15:47
# sog :12.0节 纬度： 25-41.181N 经度： 121-56.643E 目的地： WAKAYAMA,JP
from math import radians, sin, cos, sqrt, atan2

# 轉換為弧度
lat1 = radians(27.677572)
lon1 = radians(125.52714)
lat2 = radians(34.21535)
lon2 = radians(135.13684)

# 哈弗辛公式
dlat = lat2 - lat1
dlon = lon2 - lon1
a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
c = 2 * atan2(sqrt(a), sqrt(1-a))

# 地球半徑（公里）
R = 6371.0

# 計算距離
distance = R * c
print(f"剩餘距離：{distance:.2f} 公里")

import numpy as np


# 假設這是從 API 獲取的即時船舶資料
new_data = {
    'Lat': 27.61578,
    'Lng_360': 125.419655,
    'Sog': 13.11,
    'dist_to_dest_km': 1168.73,
    'near_port_flag': 0.0
}

# 將資料轉換為 DataFrame
new_df = pd.DataFrame([new_data])

# 使用標準化器進行資料標準化


# 使用模型進行預測
predicted_eta = rf_model.predict(new_df)

# 顯示預測結果
print(f"預測的剩餘到港時間：{predicted_eta[0]:.2f} 秒")

from datetime import datetime, timedelta

# 假設 predicted_eta 是模型的輸出（秒）
predicted_eta = predicted_eta.astype(float)[0]  

# 1. 取得現在時間
now = datetime.now()

# 2. 換成 timedelta（四捨五入到分鐘）
eta_timedelta = timedelta(seconds=round(predicted_eta))

# 3. 加到現在時間
predicted_arrival = now + eta_timedelta

# 4. 輸出結果
print("現在時間:", now.strftime("%Y-%m-%d %H:%M:%S"))
print("預測到港時間:", predicted_arrival.strftime("%Y-%m-%d %H:%M"))

#model_1 預測到港時間大約 2025/09/20 19:15:00。
#model_2 預測到港時間大約為 2025/09/20 16:25:52
#搜船網預測到港時間為 2025/09/20 15:30:00