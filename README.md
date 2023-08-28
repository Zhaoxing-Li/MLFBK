# MLFBK
This repository is for the research **Broader and Deeper: A Multi-Features with Latent Relations BERT Knowledge Tracing Model.**

## Abstract
Knowledge tracing aims to estimate students' knowledge state or skill mastering level over time, which is evolving into an essential task in educational technology. Traditional knowledge tracing algorithms generally use one or a few features to predict students' behaviour and do not consider the latent relations between these features, which could be limiting and disregarding important information in the features. In this paper, we propose MLFBK: A Multi-Features with Latent Relations BERT Knowledge Tracing model, which is a novel BERT based Knowledge Tracing approach that utilises multiple features and mines latent relations between features to improve the performance of the Knowledge Tracing model. Specifically, our algorithm leverages four data features student_id, skill\_id, item\_id, and response\_id, as well as three meaningful latent relations among features to improve the performance: individual skill mastery, ability profile of students (learning transfer across skills), and problem difficulty. By incorporating these explicit features, latent relations, and the strength of the BERT model, we achieve higher accuracy and efficiency in knowledge tracing tasks. We use t-SNE as a visualisation tool to analyse different embedding strategies. Moreover, we conduct ablation studies and activation function evaluation to evaluate our model. Experimental results demonstrate that our algorithm outperforms baseline methods and demonstrates good interpretability. 

## Setup 

pip install -r requirements.txt


## Model Architecture
![Architecture](https://github.com/Zhaoxing-Li/MLFBK/blob/main/Architecture.jpg)
The preprocessed data contains a list of student interactions.
In each interaction a student answers a question correctly or incorrectly.

These features are being used as inputs for the model:
- `user_id` the user id of the student relevant to the interaction
- `item_id` the question ID that is being answered in the interaction
- `correct` 0 or 1 if the student answered correctly
- `skill_id` the skill ID that corresponds to the question

After loading the initial data, to avoid data leakage the data is split into 70% training data, 20% testing data and 10% validation data.

Then using the `IKT_HANDLER` class, the following new features are generated from the input features:
- `problem_difficulty` which represents the difficulty of the question in a number between 1 and 10
- `ability_profile` represents the transfer learning across different skills
- `skill_mastery` represents the current knowledge state of a given student


The BERT layer uses embedding for all features and adds these together:
- `question embedding` +
- `response embedding` +
- `item embedding` +
- `ability profile embedding` +
- `problem difficulty embedding` +
- `skill mastery embedding` +
- `positional embedding` +

After the encoder stack and embedding, the input is passed to a feed forward layer for prediction.


## Errata
If you have any question or find error in the code, you can send me a mail.

Contact: Zhaoxing Li (zhaoxing.li0808@outlook.com).
