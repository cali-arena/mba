def solution(N, ratings):
    dish_ratings = {}
    
    for i in range(N):
        dish_id = ratings[i][0]
        rating = ratings[i][1]
        
        if dish_id not in dish_ratings:
            dish_ratings[dish_id] = []
        dish_ratings[dish_id].append(rating)
    
    best_dish_id = None
    best_average = -1
    
    for dish_id in dish_ratings:
        ratings_list = dish_ratings[dish_id]
        average = sum(ratings_list) / len(ratings_list)
        
        if average > best_average or (average == best_average and (best_dish_id is None or dish_id < best_dish_id)):
            best_average = average
            best_dish_id = dish_id
    
    return best_dish_id
