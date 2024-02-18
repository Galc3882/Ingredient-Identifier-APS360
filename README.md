# Ingredient-Identifier-APS360

## How to run the code
1. Clone the repository
2. Run the following command in the terminal to install the required packages:
```bash
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