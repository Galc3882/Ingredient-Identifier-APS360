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
import ijson
from tqdm import tqdm
import re


def process_recipes(input_file, output_file, layer1_file, layer2_file, num_words_threshold, count_threshold, num_ingrs_per_recipe, num_images_threshold):
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
    low_image_recipes = process_recipe_images(layer2_file, stats['num_recipes'], num_images_threshold)

    # Process valid ingredients by adding them to a set
    valid_ingredients, low_count_ingredients = process_valid_ingredients(
        input_file, stats['num_recipes'], low_image_recipes, num_words_threshold, count_threshold, num_ingrs_per_recipe)

    # Process invalid ingredients and update recipes
    stats = process_invalid_ingredients(input_file, output_file, valid_ingredients, stats, layer1_file,
                                        low_image_recipes, low_count_ingredients, num_words_threshold, num_ingrs_per_recipe)

    # Write statistics to files
    write_statistics(stats, valid_ingredients)


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


def process_recipe_images(layer2_file, num_recipes, num_images_threshold):
    '''Process recipes and remove invalid ingredients.'''
    print("Processing recipe images")

    low_image_recipes = set()
    with open(layer2_file, 'rb') as f:
        for recipe in tqdm(ijson.items(f, 'item'), total=num_recipes):
            if len(recipe['images']) < num_images_threshold:
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
                if len(recipe['ingredients']) >= num_ingrs_per_recipe:
                    for i, ingredient in enumerate(recipe['ingredients']):
                        if recipe['valid'][i] and ingredient['text'] != "" and len(ingredient['text'].split(" ")) <= num_words_threshold:
                            # Count unique ingredients and frequency
                            if ingredient['text'] not in valid_ingredients:
                                valid_ingredients.add(ingredient['text'])
                                counts[ingredient['text']] = 1
                            else:
                                counts[ingredient['text']] += 1

    # Reduce ingredients based on occurrences
    freq = sorted(counts.items(), key=lambda x: x[1])
    # Remove ingredients where freq is below count_threshold
    low_count_ingredients = set()
    for i in range(len(freq)):
        if freq[i][1] >= count_threshold:
            break
        else:
            valid_ingredients.remove(freq[i][0])
            low_count_ingredients.add(freq[i][0])

    return valid_ingredients, low_count_ingredients


def process_invalid_ingredients(input_file, output_file, valid_ingredients, stats, layer1_file, low_image_recipes, low_count_ingredients, num_words_threshold, num_ingrs_per_recipe):
    '''Process invalid ingredients and update recipes.'''
    # Get partitions from layer1.json
    layer1_dict = get_partitions(layer1_file, stats['num_recipes'])

    # Preprocess valid ingredients for faster processing
    preprocessed_valid_ingredients = {vi: set(vi.lower().split(" ")) for vi in valid_ingredients if vi}

    # Process invalid ingredients and update recipes
    print("Processing invalid ingredients")
    with open(input_file, 'r') as f, open(output_file, 'w') as out_f:
        recipes = ijson.items(f, 'item')
        out_f.write('[')
        first_recipe = True

        # Process each recipe
        for recipe in tqdm(recipes, total=stats['num_recipes']):
            valid_ingredients_in_recipe = set()

            # Skip recipes with too few images
            if recipe['id'] in low_image_recipes or len(recipe['ingredients']) <= num_ingrs_per_recipe:
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
            if valid_ingredients_in_recipe and len(valid_ingredients_in_recipe) > 5:
                if not first_recipe:
                    out_f.write(',')
                json.dump(recipe, out_f)
                first_recipe = False
            else:
                stats['recipes_removed'].append(recipe['id'])
        out_f.write(']')
    return stats


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
    num_words_threshold = 2
    count_threshold = 600
    num_images_threshold = 4
    num_ingrs_per_recipe = 5
    process_recipes('data/det_ingrs.json', 'data/det_ingrs_processed.json', 'data/layer1.json',
                    'data/layer2+.json', num_words_threshold, count_threshold, num_images_threshold, num_ingrs_per_recipe)
