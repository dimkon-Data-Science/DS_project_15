from recipes import Forecast, NutritionFacts, SimilarRecipes
import pickle
import sys

''' мой модуль '''
from src.projmodul import *
# -------------------------------
# 
# -------------------------------
def main():
    # -------------------------------
    # 
    # -------------------------------
    print("----------------------")
    print("Программа Nutritionist")
    print("----------------------")
    ingrs = loading_model('./data/feature.d15')
    print(f"Доступно {len(ingrs)} ингредиентов")  
    print("----------------------")  

    # print(ingrs)
    # -------------------------------
    #   Аргументы пользователя - список ингредиентов
    # -------------------------------
    check = input("Введите через запятую ингредиенты: ").lower()
    print("Вы ввели: ", check)
    # -------------------------------
    #  Если введенных ингредиентов нет в списке
    # -------------------------------
    check_list = check.split(', ')
    search_check = feature_vector(ingrs, check_list)
    if sum(search_check) < len(check_list):
        print("Введенные ингредиенты не найдены, попробуйте снова.")
        return 
    # -------------------------------
    #   
    # -------------------------------
    print("\nI. НАШ ПРОГНОЗ")
    fc = Forecast(check_list)
    rating = fc.predict_rating_category()
    print(f"\n{rating}")
    # -------------------------------
    # 
    # -------------------------------
    print("\nII. ПИЩЕВАЯ ЦЕННОСТЬ")
    nf = NutritionFacts(check)
    nutr_dict = nf.filter(30)
    for key in nutr_dict:
        print(f"\n{key.title()}")
        for nf in nutr_dict[key]:
            print(nf)
    # -------------------------------
    # 
    # -------------------------------
    # print("\nIII. ТОП-3 ПОХОЖИХ РЕЦЕПТА:")
    sr = SimilarRecipes(check)
    print(sr.top_similar(3))
    # -------------------------------
    # 
    # -------------------------------
    
main()

    