
import numpy as np
from itertools import product
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import re
import pickle

# ------------------------------------------
# 
# ------------------------------------------
# Функция для расчета RMSE (негативного)
# -------------------------------------
# Принимает:
#   y_true: реальные значения целевой переменной (таргета)
#   y_pred: прогнозные значения таргета (model.predict(X_test))
# Воззвращает:
#   Отрицательный RMSE
# Так как метрика rmse должна уменьшаться при оптимизации, то в алгоритмах настроенных на увеличение метрики, удобнее использовать ее со знаком "минус"
# 
def rmse(y_true, y_pred):
    err = (y_true - y_pred)**2
    return -np.sqrt(np.mean(err))

# ------------------------------------------
# 
# ------------------------------------------
# Функция для создания всех комбинаций словаря
# -------------------------------------
# Принимает:
#   Словарь
# Возвращает:
#   Все возможные комбинации словаря
# 
def CombDict(dct):
    keys = dct.keys()
    values = (dct[key] for key in keys)
    dict_combs = [dict(zip(keys, comb)) for comb in product(*values)]
    return (dict_combs)

# ------------------------------------------
# 
# ------------------------------------------
def reg_filter(list_str, clear=False):
    ''' 
    В процессе создания!
    
    Функция создает строку (reg_str) для использования в регулярных выраженияx внутри функций фильтрации столбцов:
    df.filter(regex=reg_str)
    df.iloc[: ,df.columns.str.contains(reg_str, regex=True)]

    и список (list_filter) для использования внутри функций фильтрации столбцов:
    df_new = df.filter(items=list_filter)
    
    Получает: 
    массив уникальных строк.
    clear: False - не очищать, True - очищать

    Возвращает: 
    - очищенную строку, которую можно использовать для фильтра столбцов df. 
    Формат строки: 'подстрока|подстрока|подстрока|подстрока|..'
    - очиищеный список для использования там же    
    '''
    # Создать регулярное выражение из списка строк
    reg_str = r'|'.join(list_str)

    if clear:
        # Найти все символы которые могут расцениваться как метасимволы регулярного выражения
        re.findall(r"[\.\^\$\*\+\?\{\[\(]", reg_str)

        # Замена скобок на разделитель подстрок
        reg_str = re.sub(r"[\(]","|",reg_str)
        reg_str = re.sub(r"[\)]","",reg_str)
        re.findall(r"[\(\)]", reg_str) # посмотрим

        # Замена "or" "with" "and" ":" на разделитель подстрок
        reg_str = re.sub(r" or ","|",reg_str)
        reg_str = re.sub(r" with ","|",reg_str)
        reg_str = re.sub(r" and ","|",reg_str)
        reg_str = re.sub(r": ","|",reg_str)

        # После замен мог появиться двойной разделитель
        if re.findall(r"\|{2}", reg_str):
            # Замена двойного разделителя на один
            reg_str = re.sub(r"\|{2}","|",reg_str)

        re.findall(r"\|{2}", reg_str)
        
        # Создана очищенная строка. Теперь из этой строки заново создать
        # список
        list_filter = list(set(reg_str.split('|')))
    else:
        list_filter = list_str

    return (reg_str, list_filter)

# ------------------------------------------
# 
# ------------------------------------------
# https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
# 
# Функция удаляет признаки с коэффициентом корреляции больше 0.8. 
# Эту операцию нужно проводить перед применением алгоритмов отбора 
# признаков по значимости, типа RFECV.
# Получает:
#   X: df только признаков, без таргета
#   coef: коэффициент корреляции, который считается высоким
# 
# Пример:
#  X = drop_corr(X, 0.8)
# 
def drop_corr(X, coef):
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > coef)]
    res = X.drop(X.columns[to_drop], axis=1)
    return res

    # correlated_features = set()
    # correlation_matrix = X.corr()

    # for i in range(len(correlation_matrix.columns)):
    #     for j in range(i):
    #         if abs(correlation_matrix.iloc[i, j]) > 0.8:
    #             colname = correlation_matrix.columns[i]
    #             # print(colname)
    #             correlated_features.add(colname)
    
    # X.drop(correlated_features)

# ------------------------------------------
# 
# ------------------------------------------
# https://machinelearningmastery.ru/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15/
# 
# Функция отбирает наиболее важные признаки. Перед применением функции 
# нужно применить функцию удаления сильно коррелированных признаков 
# drop_corr().
# Получает:
#   X: признаки
#   y: таргет
#   estim: алгоритм ML
#   cv: количество фолдов для кросс-валидации
#   scoring: оптимизируемая метрика при выборе признаков
# Возвращает: 
#   обученный объект
# Пример:
    # estim = Ridge(random_state=21)
    # scr='neg_root_mean_squared_error'
    # rfecv = feature_select(X, y, estim, 5, scr)

    # print("Сколько осталось колонок", sum(rfecv.support_))
    # print("Номера лишних колонок", np.where(rfecv.support_ == False)[0])
# 
def feature_select(X, y, estim, cv, scoring):
    # Инициализация/настройка селектора
    rfecv = RFECV(
        estimator=estim, 
        step=1, # количество функций, которые нужно удалить на каждой итерации
        cv=cv, 
        scoring=scoring
        )
        
    rfecv.fit(X, y)
    # Удалить лищние колонки
    X.drop(X.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

    return rfecv

# ------------------------------------------
# 
# ------------------------------------------
'''
https://habr.com/ru/post/475552/

Оценивание эффективности выполнения каждого алгоритма
---------------------------------------------------

R2 – коэффициент детерминации – это доля дисперсии зависимой переменной, объясняемая рассматриваемой моделью 
( http://www.machinelearning.ru/wiki/index.php?title=%D0%9A%D0%BE%D1%8D%D1%84%D1%84%D0%B8%D1%86%D0%B8%D0%B5%D0%BD%D1%82_%D0%B4%D0%B5%D1%82%D0%B5%D1%80%D0%BC%D0%B8%D0%BD%D0%B0%D1%86%D0%B8%D0%B8 ).

«Коэффициент детерминации для модели с константой принимает значения от 0 до 1. Чем ближе значение коэффициента к 1, тем сильнее зависимость. При оценке регрессионных моделей это интерпретируется как соответствие модели данным. Для приемлемых моделей предполагается, что коэффициент детерминации должен быть хотя бы не меньше 50% (в этом случае коэффициент множественной корреляции превышает по модулю 70%). Модели с коэффициентом детерминации выше 80% можно признать достаточно хорошими (коэффициент корреляции превышает 90%). Равенство коэффициента детерминации единице означает, что объясняемая переменная в точности описывается рассматриваемой моделью»

Пример использования
---------------------
models = []
models.append(('LR', LinearRegression()))
models.append(('R', Ridge()))
models.append(('L', Lasso()))
models.append(('ELN', ElasticNet()))
models.append(('LARS', LarsCV()))
# models.append(('BR', BayesianRidge(n_iter=n_iter)))
models.append(('BR', BayesianRidge()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('DTR', DecisionTreeRegressor()))
models.append(('LSVR', LinearSVR()))
models.append(('SVR', SVR()))
models.append(('ABR', AdaBoostRegressor()))
models.append(('BR', BaggingRegressor()))
models.append(('ETR', ExtraTreesRegressor())) 
models.append(('GBR', GradientBoostingRegressor()))
models.append(('RFR', RandomForestRegressor()))

algorithm_select(
    models,
    X_train, 
    y_train, 
    X_test, 
    y_test,
    scoring = 'r2',
    num_folds = 10,
    seed = 21
    )
'''
def algorithm_select(
        models, # список проверяемых ML алгоритмов
        X_train, 
        y_train, 
        X_test, 
        y_test,
        scoring = 'r2',
        num_folds = 10,
        seed = 21
    ):
    scores = []
    names = []
    results = []
    predictions = []
    msg_row = []
    for name, model in models:
        # kfold = KFold(n_splits=num_folds, random_state=seed)
        kfold = KFold(n_splits=num_folds, random_state=seed,shuffle=True)
        # 
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        names.append(name)
        results.append(cv_results)
        # 
        m_fit = model.fit(X_train, y_train)
        m_predict = model.predict(X_test)
        predictions.append(m_predict)
        m_score = model.score(X_test, y_test)
        scores.append(m_score)
        # 
        msg = "%s: train = %.3f (%.3f) / test = %.3f" % (name, cv_results.mean(), cv_results.std(), m_score)
        msg_row.append(msg)
        print(msg)
    # Диаграмма размаха («ящик с усами»)
    fig = plt.figure()
    fig.suptitle('Сравнение результатов выполнения алгоритмов')
    ax = fig.add_subplot(111)
    red_square = dict(markerfacecolor='r', marker='s')
    plt.boxplot(results, flierprops=red_square)
    ax.set_xticklabels(names, rotation=45)
    plt.show()

# ------------------------------------------
# 
# ------------------------------------------
# Функция для оценки модели
# ------------------------------------------
# 
def modelEstimate(
    model, 
    X_train, 
    y_train, 
    X_test, 
    y_test,
    prt = True, 
    write=False
    ):
    models_estimate = []
    model = model.fit(X_train, y_train)
    # ----------------------------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    # ----------------------------------------
    if prt:
        print('accuracy is', round(accuracy, 5))
        print('precision is', round(precision, 5))
        print('recall is', round(recall, 5))

    if write:
        models_estimate.append({
            'name':type(model).__name__,
            'accuracy':accuracy, 
            'precision':precision, 
            'recall':recall
          })

    return {'accuracy':accuracy, 'precision':precision, 'recall':recall}

# ------------------------------------------
# 
# ------------------------------------------
# Создание вектора признаков для проверки работы модели 
# 
def feature_vector(feature, check):
    '''
    Принимает:
        feature - полный список признаков (элементов) - имена колонок пространства признаков
        check - искомый набор признаков (элементов) - вектор с именами тех признаков которые у нас есть - какое то подмножество feature 
    Возвращает:
        feature_check: список длинной len(feature) из 0 и 1, причем 1 располложены по индексам найденных в списке feature элементов из списка check
    Для проверки feature_check используй:
        inx_check = [i for i in range(len(feature_check)) if feature_check[i] == 1]
    '''
    # индексы элементов списка check в списке feature
    res = [([idx for idx, val in enumerate(feature) if val == sub] if sub in feature else [None]) for sub in check]
    # вектор нулей с единицами по индексам res
    feature_check = [1 if [i] in res else 0  for i in range(len(feature))]
    return feature_check

# ------------------------------------------
# 
# ------------------------------------------
# Функция сохранения модели
# 
def saving_model(model, filepath):
    ''' 
    Принимает:
        model: модель
        filepath: путь к файлу в котором будет сохранена модель
    '''
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print('Models saved!')

# ------------------------------------------
# 
# ------------------------------------------
# Функция сохранения модели
# 
def loading_model(filepath):
    ''' 
    Принимает:
        filepath: путь к файлу в котором будет сохранена модель
    Возвращает:
        model: модель
    '''
    f = open(filepath, "rb")
    model = pickle.load(f)
    f.close()

    # with open(filepath, "rb") as f:
    #     model = pickle.load(f)

    return model
    
# ------------------------------------------
# 
# ------------------------------------------
# Функция для отрисовки наиболее важных признаков
# См. проект d02
# Примеры:
# Для LogRegression
# f_importances(
#   abs(mod_clf_m[1].coef_[0]), 
#   X_train.columns, 
#   top=20
#   )
# 
# Для GradientBoostingClassifier
# f_importances(
#   abs(xgb_clf_new.feature_importances_[0:10]), 
#   X_train.columns, 
#   top=10
#   )
# 
def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)
    plt.figure(figsize=(12,5))
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()
# ------------------------------------------
# 
# ------------------------------------------
