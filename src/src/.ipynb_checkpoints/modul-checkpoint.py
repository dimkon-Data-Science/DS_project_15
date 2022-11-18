import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from itertools import product

# __all__=["FeatureExtractor","DatasetCleaner"]
# ------------------------------------------
# 
# ------------------------------------------
class FeatureExtractor:
    ''' 
    Принимает на вход исследуемый датафрейм и описание ингредитентов (датаффрейм). Отфильтровывает поля: удаляет те столбцы, которые не относятся к названиями ингридиентов 
    '''

    def __init__(self, df_xl):
        self.df_xl = df_xl.copy()

    def fit(self, df, y=None):
        self.df = df.copy()
        return self
    
    def extract_ingredients(self):
        ''' 
        Извлекает названия ингредиентов из датафрейма с описанием ингредиентов 
        '''
        ingredient_descr = self.df_xl.drop_duplicates()
        # Получим первое-основное определение любого ингредиента в первом столбце и дополнительное описание в остальных
        ingredients = ingredient_descr['Ingredient description'].str.split(',',expand=True)
        # Создать фильтр по первому слову "Ingredient description"
        self.filter_1 = list(ingredients[0].unique())
        self.filter_1 = [f.lower() for f in self.filter_1]
        # `fat` может быть как нутриентом, так и ингредиентом, поэтому лучше его удалить из списка ингредиентов, так как непонятно в каком качестве он используется в исходном датафрейме.
        self.filter_1.remove('fat')

    def extract_feature(self):
        ''' 
        Отфильтровывает столбцы исходного датафрейма: оставляет только те которые соответствуют названиям ингредиентов
        '''
        # список уникальных имен столбцов
        col_recipes = list(self.df.columns.unique())
        col_recipes = [f.lower() for f in col_recipes]
        # Фильтр имен столбцов  
        self.ingredients_recipes = list(set(col_recipes) & set(self.filter_1))
        # Выборка столбцов по отфильтрованным именам 
        self.df_recipes = self.df[[*self.ingredients_recipes,'rating']]

    def transform(self, df, y=None): 
        self.extract_ingredients()
        self.extract_feature()
        return self.df_recipes 

# ------------------------------------------
# 
# ------------------------------------------

class DatasetCleaner:
    # pass
    ''' 
    Получает на вход датасет и имя столбца таргета (y). Очищает датасет. 
    - Удаляет Дубликаты строк. 
    - Удаляет Строки, содержащие в признаках только 0. 
    - Заполняет нулями пропущенные значения.
    - Удаляет Строки где таргет равен 0.
    Возвращает X и y
    '''
    def __init__(self, y_name):
        self.y_name = y_name # имя таргета

    def fit(self, df, y=None):
        self.df = df.copy()
        self.x_name = list(self.df.columns) # имена признаков
        self.x_name.remove(self.y_name)
        return self

    def cleaner(self):
        ''' 
        Очищает датасет
        '''
        # дубликаты
        self.df = self.df.drop_duplicates()
        # заполняет нулями пропущенные значения, если таие есть
        if self.df.isnull().values.any():
            self.df.fillna(0, inplace=True)
        # строки, содержащие в признаках только нули
        self.df = self.df[self.df[self.x_name].sum(axis=1)>0]
        # строки, где таргет равен 0
        self.df = self.df[self.df[self.y_name]>0]

    def downcast(self):
        '''
        Преобразовать признаки из типа float в тип int, так как значения столбцов - целые числа
        '''
        self.df[self.x_name] = self.df[self.x_name].astype('int8')

    def transform(self, df, y=None): 
        self.cleaner()
        self.downcast()
        return (self.df[self.x_name], self.df[self.y_name])

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
class ModelSelection:
    '''
    Принимает на вход список экземпляров `GridSearchCV` и словарь, в котором ключами являются индексы из этого списка, а значениями – названия моделей.
    '''
    def __init__(self, grids, grid_dict):
        self.grids = grids
        self.grid_dict = grid_dict

    def choose(self, X_train, y_train, X_valid, y_valid):
        '''
        Принимает на вход X_train, y_train, X_valid, y_valid и возвращает название лучшего классификатора среди всех моделей на валидационной выборке
        '''
        self.result = {'model':[], 'params':[], 'valid_score':[]}
        val_tests = []

        for i, grid in enumerate(self.grids):
            print('\nEstimator:', self.grid_dict[i])
            dct = grid.get_params()['param_grid']
            # num_combs = len(CombDict(dct))
            
            # Поиск
            grid.fit(X_train, y_train)

            val_test = grid.score(X_valid, y_valid)
            val_tests.append(val_test)

            self.result['model'].append(self.grid_dict[i])
            self.result['params'].append(grid.best_params_)
            self.result['valid_score'].append(val_test)

            print('Best params: ', grid.best_params_)
            print('Best training score:', round(grid.best_score_, 3))
            print('Validation set score for best params:', round(val_test, 3))

        max_idx = val_tests.index(max(val_tests))
        print('\nModel with best validation set score:',self.grid_dict[max_idx])
         
    def best_results(self):
        '''
        Возвращает датафрейм со столбцами model, params, valid_score, где строки – это модели, являющиеся лучшими в своем классе моделей
        '''
        return pd.DataFrame(self.result)
# ------------------------------------------
# 
# ------------------------------------------