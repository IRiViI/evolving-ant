# Walking Ant using Genetic algorithm

Code for evolving and rendering an ant using the openai ant roboschool environment.

## Dependencies

Roboschool Ant enviroment
https://gym.openai.com/envs/RoboschoolAnt-v1/

Tensorflow
https://www.tensorflow.org/

Python3 (ofcourse)

Pandas (optional)

jupyter (for training)

## Run pretrained model

python3 run.py

There are multiple keyarguments
Example with default model:
python3 run.py --creatures=300 --steps=600 --hidden_layers=128,32 --models_dir='./agents_weights/128_32'

Example with larger model:
python3 run.py --creatures=300 --steps=600 --hidden_layers=64,32 --models_dir='./agents_weights/64_32'

## Training model

Use training.ipynb jupter notebook

## Analysis

Requires pandas

Use the analysis.ipynb jupyter notebook