# Fine-Tuning a Vision-Language Model for Game Scene Analysis

This project demonstrates a data-centric approach to fine-tuning a Vision-Language Model (VLM) to understand and answer questions about dynamic game environments. Using data from the game *SuperTuxKart*, this repository provides scripts to build a question-answering dataset, train a model, and evaluate its performance.

![SuperTuxKart](https://supertuxkart.net/assets/wiki/STK0.8.1_3.jpg)
*(Image from supertuxkart.net)*

## Key Features

- **Data Generation Pipeline**: A Python script (`src/generate_qa.py`) that parses game metadata (kart positions, track details) to automatically generate a large-scale, high-quality dataset of image-question-answer pairs.
- **Efficient Fine-Tuning**: Utilizes Parameter-Efficient Fine-Tuning (PEFT) with LoRA to efficiently adapt a pre-trained VLM (`HuggingFaceTB/SmolVLM-256M-Instruct`) to the new task.
- **Multiple Question Types**: The generated dataset can include questions about object identification, counting, spatial relationships (left/right, front/behind), and general scene information.
- **Modular Code**: The project is structured with clear, separate modules for data handling, model definition, training, and QA generation.

## How It Works

The core workflow is as follows:

1.  **Parse Data**: The `generate_qa.py` script reads JSON files containing vision labels from the game.
2.  **Generate QA Pairs**: It then generates a series of questions and corresponding answers for each game screenshot based on the parsed labels.
3.  **Train Model**: The `finetune.py` script uses this new dataset to perform Supervised Fine-Tuning on the VLM, teaching it to answer questions about the game.

## Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

1.  **Create and activate a conda environment:**
    ```bash
    conda create --name vlm-project python=3.12 -y
    conda activate vlm-project
    ```

2.  **Install PyTorch:**
    Follow the instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/) for your specific hardware (CPU/GPU).

3.  **Install project dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

#### 1. Download and Unzip Data

First, you need the SuperTuxKart vision dataset.

```bash
wget [https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip](https://utexas.box.com/shared/static/qubjm5isldqvyimfj9rsmbnvnbezwcv4.zip) -O supertux_data.zip
unzip supertux_data.zip
```
 This creates a 'data' directory which should be in the project's root folder.


#### 2. Generate the Question-Answer Dataset

Use the provided script to generate the training data from the downloaded files.

```bash
python -m src.generate_qa generate --data_split "train"
```

This will create a generated_qa_pairs.json file inside the data/train/ directory.

#### 3. Train the Model

Fine-tune the VLM on the dataset you just created.

```Bash
python -m src.finetune train
```
This will start the training process and save the fine-tuned model checkpoints in the vlm_sft/ directory.

#### 4. Test a Trained Model

You can benchmark your trained model against a validation set.

```Bash
# The path should point to the directory containing your adapter_config.json
python -m src.finetune test path/to/your/checkpoint
```


#### Project Structure

## Project Structure

```
.
├── data/                 # Game data (after download)
├── src/                  # Main source code for the project
│   ├── generate_qa.py    # --- SCRIPT TO GENERATE QA DATASET ---
│   ├── finetune.py       # --- SCRIPT TO TRAIN THE MODEL ---
│   ├── data.py           # VQA dataset and benchmark logic
│   ├── base_vlm.py       # Base VLM model setup
│   └── ...
├── grader/               # Evaluation and grading scripts
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers**: For model and processor loading.
- **Hugging Face PEFT**: For LoRA-based fine-tuning.
- **NumPy & Pillow**: For data and image manipulation.
- **Fire**: For creating command-line interfaces.
