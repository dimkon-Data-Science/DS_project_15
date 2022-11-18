import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from tqdm import trange, tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from timeit import default_timer as timer
from .utils import *

# PYTHONDONTWRITEBYTECODE = True

# ------------------------------------------
# 
# ------------------------------------------
def gsList(models, param_grid, gs_type = 'gscv', scoring=None, cv=None, verbose=None, n_jobs=None):
    '''
    Функция создает список объектов GridSearch()
    Получает:
        models: список алгоритмов
        param_grid: соответствующий models список параметров для поиска
        gs_type: какой тип объектов для поиска гиперпараметров создавать - gscv:GridSearchCV, gspb: GridSearchPB, gsrn: RandomizedSearchCV
        остальные параметры - известны
    Возвращает:
        grids: список объектов GridSearch()
        grid_dict: словарь алгоритмов типа {0: 'BayesianRidge', 1: 'LarsCV'}
    '''
    grids = []
    grid_dict = {}
    i = 0
    for model, param_gs in zip(models, param_grid):
        if gs_type == 'gscv':
            gs = GridSearchCV(model,param_gs,scoring=scoring,cv=cv,verbose=verbose,n_jobs=n_jobs)
            grids.append(gs)
        if gs_type == 'gspb':
            gs = GridSearchPB(model,param_gs,scoring=scoring,cv=cv,n_jobs=n_jobs)
            grids.append(gs)
        if gs_type == 'gsrn':
            gs = RandomizedSearchCV(model,param_gs,scoring=scoring,cv=cv,n_jobs=n_jobs)
            grids.append(gs)

        grid_dict[i] = type(model).__name__
        i += 1

    return grids, grid_dict

# ------------------------------------------
# 
# ------------------------------------------
class FeatureExtractor:
    ''' 
    Принимает на вход исследуемый датафрейм и описание ингредитентов (датаффрейм). Отфильтровывает поля: удаляет те столбцы, которые не относятся к названиями ингридиентов 

    Метод fit
    Принимает:
    df: исследуемый датафрейм
    y: имя колнки с целевой переменной (таргет)

    __init__
    df_xl: датафрейм с одной колонкой, содержащей описание ингредиентов
    add_clmns: список колонок которые должны остаться после фильтрации, например, поле таргета
    clear: флаг, управляющий очисткой фильтра:
        False - не очищать
        True - очищать
    strong: флаг, управляющий типом фильтрации столбцов:
        False - не строгая фильтрация: имя колонки должно включать в себя подстроку фильтра.
        True - строгая фильтрация: имя колонки должно быть равно подстроке фильтра 
    select_imp: флаг, True - применить алгоритм RFECV для отбора наиболее значимых признаков
    '''
    def __init__(self, 
        df_xl, 
        add_clmns, 
        clear=False, 
        strong=False, 
        select_imp=False
        ):

        self.df_xl = df_xl.copy()
        self.add_clmns = add_clmns
        self.strong = strong
        self.clear = clear
        self.select_imp = select_imp

    def fit(self, df, y=None):
        self.df = df.copy()
        return self
    
    def extract_ingredients(self):
        ''' 
        Извлекает названия ингредиентов из датафрейма с описанием ингредиентов.
        Получает датафрейм, где в 0 столбце находится описание нигредиентов.
        Возвращает: 
            filter_str: строка для фильтрации столбцов формата "name|name|name|..."
            filter_list: список строк-имён для фильтрации столбцов
        '''
        ingredient_descr = self.df_xl.drop_duplicates()
        # Получим первое-основное определение любого ингредиента в первом столбце и дополнительное описание в остальных столбцах
        ingredients = ingredient_descr.iloc[:,0].str.split(',',expand=True)

        # Создать фильтр 
        filter_1 = list(ingredients[0].unique())
        filter_1 = [f.lower() for f in filter_1]

        # Очистить и преобразовать фильтр 
        self.filter_str, self.filter_list = reg_filter(filter_1, self.clear)

    def select_imp_feature(self):
        ''' 
        Выбирает наиболее важные признаки с точки зоения алгоритмов ML
        '''
        # Удаление сильнокоррелированных признаков
        drop_corr(self.df_recipes)
        # Отбор признаков по важности (RFECV)
        estim = Ridge(random_state=21)#BayesianRidge()#
        scr='neg_root_mean_squared_error'
        rfecv = feature_select(self.df_recipes, self.df['rating'], 
            estim, 5, scr)

    def extract_feature(self):
        ''' 
        Отфильтровывает столбцы исходного датафрейма: оставляет только те которые соответствуют полученному списку.
        Получает фильтры с возможными названиями столбцов.
        Возвращает датафрейм в котором остались только те столбцы, имена которых в максимальной степени соответствуют списку названий.
        '''
        # список уникальных имен столбцов исходного df
        col_recipes = list(self.df.columns.unique())
        col_recipes = [f.lower() for f in col_recipes]

        # Выборка столбцов по очищенным фильтрам
        if self.strong:
            # Строгое соответствие - название столбца должно быть равно переданной подстроке.
            self.df_recipes = self.df.filter(items=self.filter_list)
        else:
            # Не строгое соответствие - название столбца должно включать в себя переданную подстроку.
            self.df_recipes = self.df.filter(regex=self.filter_str)
        
        # Отбор наиболее важных признаков
        if self.select_imp:
            # Выбирает наиболее важные признаки с точки зоения алгоритмов ML
            # self.select_imp_feature()
            self.df_recipes.fillna(0, inplace=True)
            # Удаление сильнокоррелированных (0.8) признаков
            self.df_recipes = drop_corr(self.df_recipes, 0.8)
            # Отбор признаков по важности (RFECV)
            # estim = RandomForestClassifier(random_state=21)
            estim = Ridge(random_state=21)
            scr='neg_root_mean_squared_error'
            rfecv = feature_select(
                self.df_recipes, 
                self.df['rating'], 
                estim, 
                5, 
                scr
                )
                
        # Добавить необходимые колонки
        self.df_recipes = self.df_recipes.join(self.df[self.add_clmns])

    def transform(self, df, y=None): 
        self.extract_ingredients()
        self.extract_feature()
        return self.df_recipes 

# ------------------------------------------
# 
# ------------------------------------------
class DatasetCleaner:
    ''' 
    Получает на вход: 
        датасет  
        y_name: имя столбца таргета (y)
        ignor_clmns: список игнорируемых при некторых операциях столбцов
        zero_del: флаг удаления строк где рейтинг равен 0
            False - не удалять
            True - удалять

    Очищает датасет: 
    - Удаляет Дубликаты строк. 
    - Удаляет Строки, содержащие в признаках только 0. 
    - Заполняет нулями пропущенные значения.
    - Удаляет Строки где таргет равен 0.
    Возвращает 
        X (признаки) 
        y (y_name)
        ignor (ignor_clmns) (Переделать!) 
            вернет ignor_clmns если указан, 
            если - нет, то вернет еще раз y
    '''
    def __init__(self, y_name, ignor_clmns='', zero_del=False):
        self.y_name = y_name # имя таргета
        self.ignor_clmns = ignor_clmns # игнорируемые столбцы
        self.zero_del = zero_del

    def fit(self, df, y=None):
        self.df = df.copy()
         # имена признаков
        self.x_name = list(self.df.columns)
        self.x_name.remove(self.y_name)
        if self.ignor_clmns:
            self.x_name.remove(self.ignor_clmns)
        return self

    def cleaner(self):
        ''' 
        Очищает датасет
        '''
        # Дубликаты
        self.df = self.df.drop_duplicates()
        # Пропуски
        # заполняет нулями пропущенные значения, если таие есть
        if self.df.isnull().values.any():
            self.df.fillna(0, inplace=True)
        # строки, содержащие в признаках только нули
        self.df = self.df[self.df[self.x_name].sum(axis=1)>0]
        # строки, где таргет равен 0
        if self.zero_del:
            self.df = self.df[self.df[self.y_name]>0]

    def downcast(self):
        '''
        Преобразовать признаки из типа float в тип int, так как значения столбцов - целые числа
        '''
        self.df[self.x_name] = self.df[self.x_name].astype('int8')

    def transform(self, df, y=None): 
        self.cleaner()
        self.downcast()
        return (
            self.df[self.x_name], 
            self.df[self.y_name], 
            self.df[self.ignor_clmns] if self.ignor_clmns else self.df[self.y_name]
            )

# ------------------------------------------
# 
# ------------------------------------------
class TrainValidationTest:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train_val_test(self):
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            self.X, self.y, stratify=self.y, test_size=0.2, random_state=21)
    
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X_train, y_train, stratify=y_train, test_size=0.2, random_state=21)

    def get_TVT(self):
        self.train_val_test()
        return (self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test)

# ------------------------------------------
# 
# ------------------------------------------
class GridSearchPB():
    def __init__(
                self, 
                model, 
                grid_params, 
                cv=5, 
                scoring=None, 
                n_jobs=None, 
                # random_state=21
            ):
        self.model = model
        self.grid_params = grid_params
        self.cv=cv 
        self.scoring=scoring 
        self.n_jobs=n_jobs
        # self.random_state=random_state

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.searchPB()
        # Обучим модель на лучших параметрах
        self.model.set_params(**self.best_params_)
        m_fit = self.model.fit(X, y)
        
        # return self
        
    def searchPB(self):
        '''
        Функция побдбора гиперпараметров
        -------------------------------------
        Принимает:
        model: объект модели
        grid_params: словарь исследуемых параметров для модели
        X:  пространство признаков для обучения модели
        y:  таргет для обучения модели
        cv: количество фолдов для кроссвалидации
        scoring: метод оценки модели (чем больше - тем лучше)
        n_jobs: число параллельных процессов при исследовании параметров(чем больше - тем быстрее)
        Возвращает:
        Лучшую комбинацию параметров модели
        Значение метрики для лучшей комбинации
        Датафрейм из списка параметров
        Вектор средних оценок
        Вектор std
        '''
        # Все комбинации гиперпараметров
        combinations = CombDict(self.grid_params)
        # Обучение на каждой комбинации и подсчет cross_val_score
        scor_mean = []
        scor_std = []

        # kfold = KFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        kfold = KFold(n_splits=self.cv) # GridSearcCV

        with trange(len(combinations), desc='Param_Search') as pbar: # прогрессбар
            for i in range(len(combinations)):
                self.model.set_params(**combinations[i])
                cv_results = cross_val_score(
                    self.model, 
                    self.X, 
                    self.y, 
                    cv=kfold, 
                    scoring=self.scoring, 
                    n_jobs=self.n_jobs
                    )
                scor_mean.append(cv_results.mean())
                scor_std.append(cv_results.std())
                pbar.update(1)

        # Наилучший score
        self.best_score_ = max(scor_mean)
        # Наилучшме параметры
        idx = scor_mean.index(self.best_score_)
        self.best_params_ = combinations[idx]

        # Датафрейм из списка параметров
        df_param = pd.DataFrame(combinations)

        return (self.best_params_, self.best_score_, df_param, scor_mean, scor_std)

    def score_rmse(self, y_true, y_pred):
        '''neg_root_mean_squared_error'''
        error =  (y_true - y_pred) ** 2
        return -np.sqrt(np.mean(error))

    def score(self, X, y):
        if self.scoring == "neg_root_mean_squared_error":
            y_pred = self.model.predict(X)
            m_score = self.score_rmse(y, y_pred)
        else: # R2
            m_score = self.model.score(X, y)
        return m_score

# ------------------------------------------
# 
# ------------------------------------------
class ModelSelection:
    '''
    Принимает на вход 
        grids: список экземпляров GridSearchCV или GridSearchPB
        
        grid_dict: словарь, в котором ключами являются индексы из этого списка, а значениями – названия моделей.
        
        scor_test: Флаг. Делать ли оценку метрики на тестовой выборке
            True - делать
            False - не делать

        prt_mode: тип вываода информации на экран 
            True -  все
            False - по минимуму 
    '''
    def __init__(self, grids, grid_dict, scor_test=True, prt_mode=True):
        self.grids = grids
        self.grid_dict = grid_dict
        self.scor_test = scor_test
        self.prt_mode = prt_mode

    def choose(self, X_train, y_train, X_test, y_test):
        '''
        Принимает на вход X_train, y_train, X_test, y_test и возвращает название лучшего классификатора среди всех моделей на тестовой выборке
        '''
        self.result = {'model':[], 'test_score':[], 'train_score':[], 'time':[],'params':[]}
        res_scors = []

        for i, grid in enumerate(self.grids):
            print('\nEstimator:', self.grid_dict[i])
            # dct = grid.get_params()['param_grid']
            # num_combs = len(CombDict(dct))
            # ----------------
            # Поиск
            # ----------------
            start = timer()
            grid.fit(X_train, y_train)# Поиск
            end = timer()
            learning_time = end - start
            print("learning time", round(learning_time,3))
            # ----------------
            # Проводить ли проверку на тестовых данных
            # ----------------
            print_test = "--" # что выводить в итогах
            if self.scor_test: 
                res_scor = grid.score(X_test, y_test)
                print_test = round(res_scor, 5)
            else:
                res_scor = grid.best_score_

            res_scors.append(res_scor)
            # ----------------
            # Итоги по всем моделям
            # ----------------
            self.result['model'].append(self.grid_dict[i])
            self.result['test_score'].append(print_test)
            self.result['train_score'].append(round(grid.best_score_, 5))
            self.result['time'].append(round(learning_time,3))
            self.result['params'].append(grid.best_params_)
            # ----------------
            # Текущий результат
            # ----------------
            if self.prt_mode:
                print('Best params: ', grid.best_params_)
                print('Best training score:', round(grid.best_score_, 5))
                print('Test score for best parameters:', print_test)

        max_idx = res_scors.index(max(res_scors))

        if self.prt_mode:
            print('\nModel with best test set score:',
            self.grid_dict[max_idx], round(max(res_scors), 5))
        
        # self.best_results()
         
    def best_results(self):
        '''
        Возвращает датафрейм со столбцами model, params, test_score, где строки – это модели, являющиеся лучшими в своем классе моделей
        '''
        if self.scor_test:
            sort_column = 'test_score'
        else:
            sort_column = 'train_score'

        df_results = pd.DataFrame(self.result).sort_values(
                by=sort_column, ascending=False
                ).round({sort_column: 5})
                
        return df_results

# ------------------------------------------
# 
# ------------------------------------------