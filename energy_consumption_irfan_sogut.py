################## Energy Consumption Forecasting PROJECT ##############################


################### Görev ##############################
# Elimizdeki veri seti üzerinden minimum hata ile enerji tüketim miktarlarını tahmin eden bir makine öğrenmesi modeli geliştireceğiz.
# Projemiz bir regresyon vakası olacaktır

################### Libraries ##############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

########################## Adjustments ###########################################
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

######################
# GÖREV 1: Veri setine EDA işlemlerini uygulayınız
# #########################
# 1. Genel Resim
# 2. NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
# 3. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 4. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 5. Hedef Değişken Analizi (Analysis of Target Variable)
# 6. Korelasyon Analizi (Analysis of Correlation)

########################### Veri Setini Yükleme ################################
df = pd.read_csv("DAYTON_hourly.csv"),

if isinstance(df, tuple):
    df = df[0]  # Tuple içindeki DataFrame'i al

################################## # 1. Genel Resim ######################################
def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    numeric_df = dataframe.select_dtypes(include=['number'])
    print(numeric_df.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

################ Erken Aşama Özellik Çıkarımı ############

df["Datetime"] = pd.to_datetime(df["Datetime"])  # String -> Datetime dönüşümü

df["year"] = df["Datetime"].dt.year
df["month"] = df["Datetime"].dt.month
df["day"] = df["Datetime"].dt.day
df["hour"] = df["Datetime"].dt.hour

df = df.drop(columns=["Datetime"])

#################### 2. NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI ############

def grab_col_names(dataframe, cat_th=200, car_th=400):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # Dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

################# 3. Kategorik Değişken Analizi (Analysis of Categorical Variables) ###############

def cat_summary(dataframe, col_name, plot=False):
    summary_df = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                               "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})
    print(summary_df)

    if plot:
        dataframe[col_name].value_counts().plot.bar(color="skyblue", edgecolor="black", figsize=(8, 4))
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col_name}")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    print("#####################################")

# Fonksiyonu tüm kategorik değişkenler için çalıştır
for col in cat_cols:
    cat_summary(df, col, plot=True)

################## 4. Sayısal Değişken Analizi (Analysis of Numerical Variables) ###############

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=60)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)

################## 5. Hedef Değişken Analizi (Analysis of Target Variable)

def cat_summary_with_target(dataframe, cat_col, target_col="DAYTON_MW", plot=False):
    summary_df = dataframe.groupby(cat_col)[target_col].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
    print(summary_df)

    if plot:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=dataframe[cat_col], y=dataframe[target_col])
        plt.xticks(rotation=45)
        plt.title(f"{cat_col} Kategorisine Göre {target_col} Dağılımı")
        plt.show()

    print("#####################################")


# Kategorik değişkenleri bağımlı değişkene göre analiz et ve görselleştir
for col in cat_cols:
    df[col] = df[col].astype(str)
    cat_summary_with_target(df, col, plot=True)

################## 6. Korelasyon Analizi (Analysis of Correlation)

# Sayısal özellikleri gruplama
target_feature = ["DAYTON_MW"]  # Tahmin edilmek istenen değişken
time_features = ["year", "month", "day", "hour"]  # Zamanla ilgili değişkenler

# Korelasyon için tüm sayısal değişkenleri birleştirme
numeric_features = target_feature + time_features

# Korelasyon matrisi oluştur
corr = df[numeric_features].corr()

# Korelasyonları görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu", center=0)
plt.title("Feature Correlation Heatmap")
plt.show()

# Yüksek korelasyonlu değişkenleri belirleme fonksiyonu
def high_correlated_cols(dataframe, corr_th=0.70, plot=False):
    corr_matrix = dataframe.corr().abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    drop_list = [column for column in upper_triangle_matrix.columns if any(upper_triangle_matrix[column] > corr_th)]

    if plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu", center=0)
        plt.title("Correlation Matrix")
        plt.show()

    return drop_list



# Yüksek korelasyonlu sütunları belirleme
drop_list = high_correlated_cols(df[numeric_features], plot=True)
print("Highly Correlated Columns:", drop_list)


######################################
# Görev 2 : Feature Engineering
######################################

########################### Aykırı Değer Analizi

# Aykırı değer sınırlarının hesaplanması
def outlier_thresholds(dataframe, variable, low_quantile=0.25, up_quantile=0.75):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
      print(col, check_outlier(df, col))

# Aykırı değer raporlama (Aşama 1)
def report_outliers(dataframe, num_columns):
    for col in num_columns:
        low, up = outlier_thresholds(dataframe, col)
        outliers = dataframe[(dataframe[col] < low) | (dataframe[col] > up)]
        print(f"{col} için {outliers.shape[0]} aykırı değer bulundu.")

report_outliers(df, num_cols)

# Aykırı değerlerin görselleştirilmesi
def plot_outliers(dataframe, num_columns):
    plt.figure(figsize=(12, len(num_columns) * 4))  # Dinamik boyutlandırma
    for i, col in enumerate(num_columns, 1):
        plt.subplot(len(num_columns), 1, i)  # Her değişken için alt grafik oluştur
        sns.boxplot(x=dataframe[col])
        plt.title(f"{col} - Aykırı Değerler")
        plt.xlabel("")  # X etiketini kaldır
        plt.grid(True)  # Izgara ekleyerek daha okunaklı hale getir
    plt.tight_layout()
    plt.show()

plot_outliers(df, num_cols)

# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df,col)

# Aykırı değer raporlama (Aşama 2)
def report_outliers(dataframe, num_columns):
    for col in num_columns:
        low, up = outlier_thresholds(dataframe, col)
        outliers = dataframe[(dataframe[col] < low) | (dataframe[col] > up)]
        print(f"{col} için {outliers.shape[0]} aykırı değer bulundu.")

report_outliers(df, num_cols)

################## Rare analizi yapınız ve rare encoder uygulayınız.

# Kategorik kolonların dağılımının incelenmesi

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "DAYTON_MW", cat_cols)


# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc, cat_cols):
    temp_df = dataframe.copy()

    rare_columns = [col for col in cat_cols if (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df, 0.01, cat_cols)

print(df["year"].value_counts())
print(df["month"].value_counts())
print(df["day"].value_counts())
print(df["hour"].value_counts())

##################################
# Görev 3: MODELLEME
##################################

############# Model kurma

np.random.seed(17)
random_indices = np.random.choice(df.index, size=100, replace=False)
df.loc[random_indices, 'DAYTON_MW'] = np.nan  # 'DAYTON_MW' sütununda boş değerler oluşturuyoruz

#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
train_df = df[df['DAYTON_MW'].notnull()]
test_df = df[df['DAYTON_MW'].isnull()]

y = train_df['DAYTON_MW'] # np.log1p(df['DAYTON_MW'])
X = train_df.drop(["DAYTON_MW"], axis=1)

# Train verisi ile model kurup, model başarısını değerlendiriniz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

int_cols = ["year", "month", "day", "hour"]
X_train[int_cols] = X_train[int_cols].astype(int)
X_test[int_cols] = X_test[int_cols].astype(int)

X["year"] = X["year"].astype(int)
X["month"] = X["month"].astype(int)
X["day"] = X["day"].astype(int)
X["hour"] = X["hour"].astype(int)

models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          # ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror', enable_categorical=True)),
          ("LightGBM", LGBMRegressor())]
          #("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")



################### hiperparametre optimizasyonlarını gerçekleştiriniz.

int_cols = ["year", "month", "day", "hour"]
df[int_cols] = df[int_cols].astype(int)


lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]
               #"colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))

############ Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)

########################################
# test dataframeindeki boş olan DAYTON değerlerini tahminleyiniz
########################################
test_df.loc[:, ["year", "month", "day", "hour"]] = test_df.loc[:, ["year", "month", "day", "hour"]].astype(int)

model = LGBMRegressor(force_col_wise=True)
model.fit(X, y)
predictions = model.predict(test_df.drop(["DAYTON_MW"], axis=1))

dictionary = {"ID":test_df.index, "DAYTON_MW":predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("DAYTON_MW_Predictions1.csv", index=False)

