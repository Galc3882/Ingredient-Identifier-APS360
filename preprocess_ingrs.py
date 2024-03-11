# data/layer1.json
# ```js
# {
#   id: String,  // unique 10-digit hex string
#   title: String,
#   instructions: [ { text: String } ],
#   ingredients: [ { text: String } ],
#   partition: ('train'|'test'|'val'),
#   url: String
# }
# ```
#
# data/det_ingrs.json
# '''js
# [
#   {
#     "valid": [Boolean, ...],
#     "id": String, // unique 10-digit hex
#     "ingredients": [ {
#       "text": String
#     } ]
#   }
# ]
# '''
#
# ### layer2+.json
#
# ```js
# {
#   id: String,   // refers to an id in layer 1
#   images: [ {
#     id: String, // unique 10-digit hex + .jpg
#     url: String
#   } ]
# }
# ```

# First initialize the statistics dictionary, which will contain the number of recipes,
# the invalid ingredients removed, the recipes removed, the number of whitespace invalid ingredients,
# and the invalid ingredients turned into valid ingredients.
# Then, we will remove the recipes with too few images, and store the ids of the removed recipes in the statistics dictionary,
# by going through each recipe in the layer2+.json file and checking if the number of images is less than the threshold.
# We will get a list of all the valid ingredients in all the recipes, and count the number of times each ingredient appears in the recipes.
# We will then remove the ingredients that appear less than a certain number of times, and store the removed ingredients in a set.
# Then, to treat the invalid ingredients, we will go through each recipe and check if it is in the list of removed recipes, and if it is, we will skip it.
# If not, we will go through each ingredient in the recipe, and if it is valid, we will add it to a set of valid ingredients for the recipe.
# If it is not valid, we will check if it is a substring of any valid ingredient, and if it is, we will replace it with the valid ingredient,
# but only if the valid ingredient is not already present in the recipe. If it is, we will remove the invalid ingredient.
# If the invalid ingredient is not a substring of any valid ingredient, we will remove the corresponding recipe.
# We will also remove the valid field from the recipe, and add the partition information from the layer1.json file to the recipe.
# Finally, we will save the data to a new JSON file, in the same format as the original file, without the valid field,
# and with the invalid ingredients and recipes removed, and the partition information added.


import json
import os
import ijson
from tqdm import tqdm
import re


def process_recipes(input_file, output_file, layer1_file, layer2_file, dataset_files, num_words_threshold, count_threshold, num_ingrs_per_recipe, num_images_threshold):
    '''Process recipes and remove invalid ingredients.'''
    # Initialize statistics
    stats = {
        'num_recipes': 0,
        'invalid_ingredients_removed': [],
        'recipes_removed': [],
        'whitespace_invalid_ingredients': 0,
        'invalid_ingredients_to_valid': []
    }

    # Get number of recipes
    stats['num_recipes'] = get_num_recipes(input_file)

    # Remove based on number of pictures
    low_image_recipes = process_recipe_images(layer2_file, dataset_files, stats['num_recipes'], num_images_threshold)

    # Process valid ingredients by adding them to a set
    valid_ingredients, low_count_ingredients = process_valid_ingredients(
        input_file, stats['num_recipes'], low_image_recipes, num_words_threshold, count_threshold, num_ingrs_per_recipe)

    # Process invalid ingredients and update recipes
    stats, ingredient_count = process_invalid_ingredients(input_file, output_file, valid_ingredients, stats, layer1_file,
                                                          low_image_recipes, low_count_ingredients, num_words_threshold, num_ingrs_per_recipe)

    # Remove low count ingredients
    stats, ingredient_count = remove_low_count_ingredients(output_file, stats, ingredient_count, count_threshold, num_ingrs_per_recipe)

    # Write statistics to files
    write_statistics(stats, sorted(ingredient_count.keys(), key=lambda x: ingredient_count[x], reverse=True))


def remove_low_count_ingredients(output_file, stats, ingredient_count, count_threshold, num_ingrs_per_recipe):
    '''Remove whole recipe from output file if there is at least one low count ingredient and update statistics.'''
    print("Removing low count ingredients")
    recipes_to_remove = []
    with open(output_file.split('/')[0] + '/temp_' + output_file.split('/')[1], 'r') as f:
        recipes = ijson.items(f, 'item')
        for recipe in tqdm(recipes, total=stats['num_recipes'] - len(stats['recipes_removed'])):
            remove_recipe = False
            for ingredient in recipe['ingredients']:
                if ingredient['text'] in ingredient_count and ingredient_count[ingredient['text']] <= count_threshold:
                    ingredient_count[ingredient['text']] -= 1
                    if not remove_recipe:
                        stats['recipes_removed'].append(recipe['id'])
                        recipes_to_remove.append(recipe['id'])
                    remove_recipe = True
    
    # Remove all ingredients with 0 count from ingredient_count
    for ingredient in list(ingredient_count.keys()):
        if ingredient_count[ingredient] == 0:
            ingredient_count.pop(ingredient)

    # Remove recipes from output file
    with open(output_file.split('/')[0] + '/temp_' + output_file.split('/')[1], 'r') as f, open(output_file, 'w') as out_f:
        recipes = ijson.items(f, 'item')
        out_f.write('[')
        first_recipe = True
        for recipe in recipes:
            if recipe['id'] not in recipes_to_remove and len(recipe['ingredients']) >= num_ingrs_per_recipe:
                if not first_recipe:
                    out_f.write(',')
                json.dump(recipe, out_f)
                first_recipe = False
        out_f.write(']')

    # Remove temp file
    os.remove(output_file.split('/')[0] + '/temp_' + output_file.split('/')[1])
    return stats, ingredient_count


def get_num_recipes(input_file):
    num_recipes = 0
    with open(input_file, 'r') as f:
        recipes = ijson.items(f, 'item')
        num_recipes = sum(1 for _ in recipes)
    return num_recipes


def get_partitions(layer1_file, num_recipes):
    '''Get partitions from layer1.json.'''
    print("Getting partitions from layer1.json")
    layer1_dict = {}
    with open(layer1_file, 'r') as f:
        for recipe in tqdm(ijson.items(f, 'item'), total=num_recipes):
            layer1_dict[recipe['id']] = recipe['partition']
    return layer1_dict


def process_recipe_images(layer2_file, dataset_files, num_recipes, num_images_threshold):
    '''Process recipes and remove invalid ingredients.'''
    print("Processing recipe images")

    low_image_recipes = set()
    with open(layer2_file, 'rb') as f:
        for recipe in tqdm(ijson.items(f, 'item'), total=num_recipes):
            if len(recipe['images']) <= num_images_threshold or sum([1 if any([img_path[:1] in dataset_files for img_path in recipe['images'][i]['id']]) else 0 for i in range(len(recipe['images']))]) <= num_images_threshold:
                low_image_recipes.add(recipe['id'])

    return low_image_recipes


def process_valid_ingredients(input_file, num_recipes, low_image_recipes, num_words_threshold, count_threshold, num_ingrs_per_recipe):
    '''Process valid ingredients and map them to recipes.'''
    print("Processing valid ingredients and mapping them to recipes")

    valid_ingredients = set()
    counts = {}
    with open(input_file, 'r') as f:
        recipes = ijson.items(f, 'item')
        for recipe in tqdm(recipes, total=num_recipes):
            if recipe['id'] not in low_image_recipes:
                if sum(recipe['valid']) >= num_ingrs_per_recipe:
                    for i, ingredient in enumerate(recipe['ingredients']):
                        if recipe['valid'][i] and ingredient['text'] != "" and len(ingredient['text'].split(" ")) <= num_words_threshold:
                            # Count unique ingredients and frequency
                            if ingredient['text'] not in valid_ingredients:
                                valid_ingredients.add(ingredient['text'])
                                counts[ingredient['text']] = 1
                            else:
                                counts[ingredient['text']] += 1

    # Merge ingredients with similar names if they are a substring of each other
    low_count_ingredients = set()
    for ingredient in valid_ingredients:
        if ingredient not in low_count_ingredients:
            for other_ingredient in valid_ingredients:
                if other_ingredient not in low_count_ingredients:
                    if ingredient != other_ingredient and ingredient in other_ingredient and abs(len(ingredient) - len(other_ingredient)) <= 3:
                        counts[ingredient] += counts[other_ingredient]
                        counts.pop(other_ingredient)
                        low_count_ingredients.add(other_ingredient)

    # Reduce ingredients based on occurrences
    freq = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    # Remove ingredients where freq is below count_threshold
    valid_ingredients = set()
    for i in range(len(freq)):
        if freq[i][1] >= count_threshold:
            valid_ingredients.add(freq[i][0])
        else:
            low_count_ingredients.add(freq[i][0])

    return valid_ingredients, low_count_ingredients


def process_invalid_ingredients(input_file, output_file, valid_ingredients, stats, layer1_file, low_image_recipes, low_count_ingredients, num_words_threshold, num_ingrs_per_recipe):
    '''Process invalid ingredients and update recipes.'''
    # Get partitions from layer1.json
    layer1_dict = get_partitions(layer1_file, stats['num_recipes'])

    # Preprocess valid ingredients for faster processing
    preprocessed_valid_ingredients = {vi: set(vi.lower().split(" ")) for vi in valid_ingredients if vi}

    # Count number of times each ingredient is in a recipe
    ingredient_count = {vi: 0 for vi in valid_ingredients}

    # Process invalid ingredients and update recipes
    print("Processing invalid ingredients")
    with open(input_file, 'r') as f, open(output_file.split('/')[0] + '/temp_' + output_file.split('/')[1], 'w') as out_f:
        recipes = ijson.items(f, 'item')
        out_f.write('[')
        first_recipe = True

        # Process each recipe
        for recipe in tqdm(recipes, total=stats['num_recipes']):
            valid_ingredients_in_recipe = set()

            # Skip recipes with too few images
            if recipe['id'] in low_image_recipes or sum(recipe['valid']) < num_ingrs_per_recipe:
                stats['recipes_removed'].append(recipe['id'])
                continue

            # Process each ingredient
            for i, ingredient in enumerate(recipe['ingredients']):
                if recipe['valid'][i] and ingredient['text'] != "" and len(ingredient['text'].split(" ")) <= num_words_threshold and ingredient['text'] not in low_count_ingredients:
                    valid_ingredients_in_recipe.add(ingredient['text'])
                else:
                    # Check for valid substrings
                    ingredient_text = re.sub(r'[^a-zA-Z\s]', ' ', ingredient['text'].lower())
                    ingredient_words = set(ingredient_text.split(" "))
                    ingredient_words.discard("")
                    found = False
                    for valid_ingredient, valid_ingredient_words in preprocessed_valid_ingredients.items():
                        if valid_ingredient not in valid_ingredients_in_recipe:
                            # Check if all words in valid_ingredient are in the ingredient's text
                            if valid_ingredient_words.issubset(ingredient_words):
                                valid_ingredients_in_recipe.add(valid_ingredient)
                                stats['invalid_ingredients_to_valid'].append(f"Invalid ingredient: {ingredient['text']} | Valid ingredient: {valid_ingredient}")
                                found = True
                                break

                    if not found:
                        if ingredient['text'] == "":
                            stats['whitespace_invalid_ingredients'] += 1
                        else:
                            stats['invalid_ingredients_removed'].append(ingredient['text'])

                            # remove whole recipe
                            valid_ingredients_in_recipe.clear()
                            break

            # Update recipe
            recipe['ingredients'] = [{'text': ingredient} for ingredient in valid_ingredients_in_recipe]

            # Add partition information from layer1
            partition = layer1_dict.get(recipe['id'])
            if partition:
                recipe['partition'] = partition

            # Remove valid field
            recipe.pop('valid')
            # Remove recipes with too few ingredients, even if they are all valid
            if valid_ingredients_in_recipe and len(valid_ingredients_in_recipe) >= num_ingrs_per_recipe:
                if not first_recipe:
                    out_f.write(',')
                json.dump(recipe, out_f)
                first_recipe = False

                # Update ingredient count
                for vi in valid_ingredients_in_recipe:
                    ingredient_count[vi] += 1
            else:
                stats['recipes_removed'].append(recipe['id'])
        out_f.write(']')
    return stats, ingredient_count


def write_statistics(stats, valid_ingredients):
    '''Write statistics to files.'''
    # Print statistics
    print(f"Number of valid ingredients: {len(valid_ingredients)}")
    print(f"Number of recipes: {stats['num_recipes']}")
    print(f"Number of invalid ingredients turned into valid ingredients: {len(stats['invalid_ingredients_to_valid'])}")
    print(f"Number of invalid ingredients removed: {len(stats['invalid_ingredients_removed'])}")
    print(f"Number of recipes removed: {len(stats['recipes_removed'])}")
    print(f"Number of recipes kept: {stats['num_recipes'] - len(stats['recipes_removed'])}")
    print(f"Number of whitespace invalid ingredients: {stats['whitespace_invalid_ingredients']}")

    # file to write the statistics
    statistics_file = 'data/statistics.txt'
    # file to write the valid ingredients
    valid_ingredients_file = 'data/valid_ingredients.txt'
    # file to write the invalid ingredients removed
    invalid_ingredients_removed_file = 'data/invalid_ingredients_removed.txt'
    # file to write the invalid ingredients turned into valid ingredients
    invalid_ingredients_to_valid_file = 'data/invalid_ingredients_to_valid.txt'
    # file to write the recipes removed
    recipes_removed_file = 'data/recipes_removed.txt'

    with open(statistics_file, 'w') as f:
        # Write the statistics to the file
        f.write(f"Number of valid ingredients: {len(valid_ingredients)}\n")
        f.write(f"Number of recipes: {stats['num_recipes']}\n")
        f.write(f"Number of invalid ingredients turned into valid ingredients: {len(stats['invalid_ingredients_to_valid'])}\n")
        f.write(f"Number of invalid ingredients removed: {len(stats['invalid_ingredients_removed'])}\n")
        f.write(f"Number of recipes removed: {len(stats['recipes_removed'])}\n")
        f.write(f"Number of recipes kept: {stats['num_recipes'] - len(stats['recipes_removed'])}\n")
        f.write(f"Number of whitespace invalid ingredients: {stats['whitespace_invalid_ingredients']}\n")

    with open(valid_ingredients_file, 'w') as f:
        # Write the valid ingredients to the file
        for ingredient in valid_ingredients:
            f.write(ingredient + '\n')

    with open(invalid_ingredients_to_valid_file, 'w') as f:
        # Write the invalid ingredients turned into valid ingredients to the file
        for ingredient in stats['invalid_ingredients_to_valid']:
            f.write(ingredient + '\n')

    with open(invalid_ingredients_removed_file, 'w') as f:
        # Write the invalid ingredients removed to the file
        for ingredient in stats['invalid_ingredients_removed']:
            f.write(ingredient + '\n')

    with open(recipes_removed_file, 'w') as f:
        # Write the recipes removed to the file
        for recipe_id in stats['recipes_removed']:
            f.write(recipe_id + '\n')


if __name__ == "__main__":
    # Set hyperparameters

    # List of dataset files
    dataset_files = ['0', '1', '2']
    # dataset_files = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']

    # The maximum number of words an ingredient can have to be considered valid
    num_words_threshold = 1  # 2
    # The number of times an ingredient must appear in the recipes to be considered valid
    count_threshold = 1200  # 600
    # The number of images a recipe must have to be considered valid
    num_images_threshold = 2  # 4
    # The number of ingredients a recipe must have to be considered valid
    num_ingrs_per_recipe = 10  # 5

    process_recipes('data/det_ingrs.json', 'data/det_ingrs_processed.json', 'data/layer1.json',
                    'data/layer2+.json', dataset_files, num_words_threshold, count_threshold, num_images_threshold, num_ingrs_per_recipe)
