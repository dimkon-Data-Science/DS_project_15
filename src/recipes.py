import pickle
import pandas as pd 
from random import randrange
''' мой модуль'''
from src.projmodul import *
''' подавление предупреждений '''
import warnings
warnings.filterwarnings('ignore')
# ------------------------------------------------
# 
# ------------------------------------------------
class Forecast:
    """
    Предсказание рейтинга блюда или его класса
    """
    def __init__(self, list_of_ingredients):
        """
        Добавьте сюда любые поля и строчки кода, которые вам покажутся нужными.
        """
        self.check = list_of_ingredients

            
    def preprocess(self):
        """
        Этот метод преобразует список ингредиентов в структуры данных, 
        которые используются в алгоритмах машинного обучения, чтобы сделать предсказание.
        """
        # Загрузка ингредиентов
        self.feature = loading_model('./data/feature.d15')

        vector = feature_vector(self.feature, self.check)
        return vector

    def predict_rating(self):
        """
        Этот метод возвращает рейтинг для списка ингредиентов, используя регрессионную модель, 
        которая была обучена заранее. Помимо самого рейтинга, метод также возвращает текст, 
        который дает интерпретацию этого рейтинга и дает рекомендацию, как в примере выше.
        """
        
        # return rating, text
        pass

    def predict_rating_category(self):
        """
        Этот метод возращает рейтинговую категорию для списка ингредиентов, используя классификационную модель, 
        которая была обучена заранее. Помимо самого рейтинга, метод возвращает также и текст, 
        который дает интерпретацию этой категории и дает рекомендации, как в примере выше.
        """
        # Загрузка модели
        self.model = loading_model('./models/ETClf(0.95880)_1.sav')

        texts = {
            'great': 'Вкусное. Многим, кто это блюдо пробовал, оно нравиться. Скорее всего, понравиться и вам.',
            'so-so': 'Нормальное. Кому то нравиться, кому то - нет',
            'bad': 'Невкусное. Хоть конкретно вам может быть и понравится блюдо из этих ингредиентов, но, на наш взгляд, это плохая идея – готовить блюдо из них. Хотели предупредить.'
        }

        rating = self.model.predict([self.preprocess()])[0]

        return texts[rating]
# ------------------------------------------------
# 
# ------------------------------------------------
class NutritionFacts:
    """
    Выдает информацию о пищевой ценности ингредиентов.
    """
    def __init__(self, list_of_ingredients):
        """
        Добавьте сюда любые поля и строчки кода, которые вам покажутся нужными.
        """
        self.ingred = list_of_ingredients.split(', ')

    def retrieve(self):
        """
        Этот метод получает всю имеющуюся информацию о пищевой ценности из файла с заранее собранной информацией по заданным ингредиентам. 
        Он возвращает ее в том виде, который вам кажется наиболее удобным и подходящим.
        """
        self.nutr = pd.read_csv('./data/NUTR_FRAME.csv', index_col='ingr')

        ret = self.nutr.loc[self.ingred].T

        return ret

    def filter(self, n):
        """
        Этот метод отбирает из всей информации о пищевой ценности только те нутриенты, которые были заданы в must_nutrients (пример в PDF-файле ниже), 
        а также топ-n нутриентов с наибольшим значением дневной нормы потребления для заданного ингредиента. 
        Он возвращает текст, отформатированный как в примере выше.
        """
        df_ret = self.retrieve()

        nutr_dict = {}
        
        for ingr in df_ret.columns:
            nutr_dict[ingr] = []

            df_ret.sort_values(by=ingr, inplace=True, ascending=False)
  
            for i, nutr in enumerate(df_ret.index):
                value = round(df_ret.at[nutr, ingr])
                if value > 0 and i < n:
                    nutr_str = f"{nutr} - {value}% of Daily Value"
                    nutr_dict[ingr].append(nutr_str)

        return nutr_dict
# ------------------------------------------------
# 
# ------------------------------------------------
class SimilarRecipes:
    """
    Рекомендация похожих рецептов с дополнительной информацией
    """
    def __init__(self, list_of_ingredients):
        """
        Добавьте сюда любые поля и строчки кода, которые вам покажутся нужными.
        """    
        self.ingred = list_of_ingredients.split(', ')

    def find_all(self):
        """
        Этот метод возвращает список индексов рецептов, которые содержат заданный список ингредиентов. 
        Если нет ни одного рецепта, содержащего все эти ингредиенты, то сделайте обработку ошибки, чтобы программа не ломалась.
        """
        self.full_df = pd.read_csv('./data/DF_FINE.csv', index_col="title")
        res = self.full_df[self.full_df[self.ingred].sum(axis=1)==len(self.ingred)].index

        return res
   
    def top_similar(self, n):
        """
        Этот метод возвращает текст, форматированный как в примере выше: с заголовком, рейтингом и URL. 
        Чтобы это сделать, он вначале находит топ-n наиболее похожих рецептов с точки зрения количества дополнительных ингредиентов, 
        которые потребуются в этих рецептах. Наиболее похожим будет тот, в котором не требуется никаких других ингредиентов. 
        Далее идет тот, у которого появляется 1 доп. ингредиент. Далее – 2. 
        Если рецепт нуждается в более, чем 5 доп. ингредиентах, то такой рецепт не выводится.
        """
        text_with_recipes = "\nIII. ТОП-3 ПОХОЖИХ РЕЦЕПТА:\n"
        if len(self.find_all())==0:
            text_with_recipes += "Похожих рецептов не найдено.\n\n"
            return text_with_recipes

        df=self.full_df.loc[self.find_all()]
        df['sum'] = self.full_df.drop(columns=['url', 'rating', 'calories', 'protein', 'fat', 'sodium', 'breakfast', 'lunch', 'dinner']).sum(axis=1)

        res = df[['sum','rating','url']].sort_values(by=['sum', 'rating'], ascending=[True, False])
        res = res[res['sum']<=(len(self.ingred)+5)]
        res = res[:n]
        if len(res.index)==0:
            text_with_recipes += "Похожих рецептов не найдено.\n\n"
            return text_with_recipes
        for ind in res.index:
            text_with_recipes+=f"- {ind.strip()}, рейтинг: {res.at[ind, 'rating']}, URL: {res.at[ind, 'url']}\n"
        text_with_recipes+='\n'
        return text_with_recipes
# ------------------------------------------------
# 
# ------------------------------------------------
        