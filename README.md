
# APS360: Ingredient Identifier from Images

## Project Description

This project was developed for the APS360 (Introduction to Artificial Intelligence) course at the University of Toronto, during the winter semester of 2024. We achieved a grade of **100%** for this project.

Our convolutional neural network (CNN) model automates the recognition of ingredients from images of food, greatly enhancing the process of recipe discovery and dietary management. The model showcases a robust ability to identify and classify a diverse range of ingredients, demonstrating promising results through extensive testing on various dishes.

The increasing importance of digital solutions in the culinary space, driven by online food services and social media, underscores the relevance of our project. By automating ingredient identification, we provide a valuable tool for home cooks, professional chefs, and individuals with dietary restrictions, making culinary exploration more accessible and informed.

Our model utilizes the strengths of CNNs, renowned for their efficiency in image recognition tasks. By processing large datasets, the model learns to recognize complex patterns in ingredients, regardless of the dish's complexity. This technical foundation not only powers our current model but also sets the stage for future enhancements that could further revolutionize how we interact with food digitally.

## Installation and Setup
1. Clone the repository
2. Run the following command in the terminal to install the required packages:
```bash
# Clone the repository
git clone https://github.com/Galc3882/Ingredient-Identifier-APS360.git

# Navigate to the project directory
cd Ingredient-Identifier-APS360

# Install required Python packages
pip install -r requirements.txt
```
3. Download the following files and place them in the `data` folder:
    - det_ingrs.json
    - layer1.json
    - layer2+.json

4. Preprocess the data by running the following command:
```bash
python preprocess_ingrs.py
```
5. Train the model by running the following command:
```bash
python train.py
```

## Model Details

### Architecture

![architecture](https://github.com/Galc3882/Ingredient-Identifier-APS360/assets/86870298/c6b95b83-370a-4c9c-9e0e-9bb39eb5c73a)

Our model is a simplified adaptation of the VGG architecture tailored for multi-class classification tasks. It is composed of three convolutional layers followed by ReLU activation functions, three max pooling layers, and three fully connected layers also employing ReLU. The architecture processes input images by initially passing them through the convolutional layers that extract various features from the images. The max pooling layers interspersed between the convolutional layers serve to distill essential features while reducing computational load. ReLU activations help introduce non-linearity, aiding in mitigating the vanishing gradient problem. The features extracted by the convolutional layers are then flattened and fed into fully connected layers, which utilize the features to classify and predict the probability of each of the 34 ingredients present in the image. If the probability exceeds 0.5, an ingredient is considered present; otherwise, it is deemed absent.

For model training, we utilize ZLPR loss, a function designed for multi-class tasks with known target labels, which considers the correlation between labels. This choice is based on findings that suggest ZLPR provides superior performance compared to binary cross-entropy, with comparable computational complexity.

### Data Processing and Training

Our dataset was derived from the extensive "Recipe1M+" dataset, focusing on a subset that includes diverse and balanced recipes. 

![example](https://github.com/Galc3882/Ingredient-Identifier-APS360/assets/86870298/a92455d1-2067-490f-bfd5-23e668226dc8)

The model training involved meticulous data preprocessing to enhance model training efficacy:

- **Image Resizing and Normalization:** Standardizing image dimensions and scale to ensure uniform input.
- **Data Augmentation:** Random modifications to the training images (like rotation and color adjustment) to improve the model's generalizability.

The training process was rigorously monitored to optimize performance metrics such as accuracy, precision, and recall, ensuring that the model performs well on unseen data.

### Performance

The model's performance was extensively evaluated against a baseline established through user studies. The model demonstrates a high degree of accuracy in identifying ingredients, closely matching human performance in controlled tests. Detailed performance results are discussed in our project report.

## Results

| **Metric**     | **Result** |
|----------------|------------|
| **Accuracy**   | 0.8329     |
| **Precision**  | 0.6453     |
| **Recall**     | 0.6647     |
| **Specificity**| 0.8856     |
| **F1 Score**   | 0.6549     |

The initial tests using our designed setup and data processing strategy yielded an accuracy of 83%. This robust performance is illustrated in the learning curve shown below, which depicts a best parameter set achieved around epoch 45, with subsequent epochs showing minimal improvements. This outcome indicates that our model fits well, effectively identifying ingredients from the images.

The model's performance closely matches that of our human baseline, with F1 scores comparable to average human performance. The significant number of true negatives did influence accuracy positively, but this was anticipated and considered in our data processing approach. We also used multiple performance metrics, such as F1 and precision, to provide a balanced view of model efficacy.

|                               | **True Class** |           |
|-------------------------------|----------------|-----------|
| **Predicted Class**           | **Positive**   | **Negative** |
| **Positive**                  | 167,322 (63.4%)| 98,340 (37.3%) |
| **Negative**                  | 88,976 (33.7%) | 704,328 (88.5%) |

Our ingredient identification model demonstrates promising capabilities in recognizing a wide array of ingredients from food images. Qualitatively, the model generally performs well with simple dishes where ingredients are clearly visible, such as in the case of salads. However, it occasionally fails to detect less distinguishable ingredients like white onion and feta cheese.

Challenges arise particularly with complex dishes where ingredients overlap or are hidden, such as in wraps or layered dishes. While the model effectively identifies visible primary ingredients and even seasoning like salt and pepper, it may overlook components like tomatoes and onions that are not immediately apparent.

Despite these challenges, the model's performance aligns closely with human benchmarks, particularly in terms of the bimodal distribution of human accuracy depending on familiarity with the dish.

## Credits

This project is the result of collaborative work by:
- **Gal Cohen** (gal.cohen@mail.utoronto.ca)
- **Hannah Kim** (hannahh.kim@mail.utoronto.ca)
- **Terrence Zhang** (terrencez.zhang@mail.utoronto.ca)

We also acknowledge the contributions from the academic and open-source community whose insights and tools have facilitated this research.

## License

This project is licensed under the MIT License. For more details, see the `LICENSE.md` file in the repository.
