# Knowledge Tracing Research
This repository contains all the code for the Knowledge Tracing research project that combines multi features with a BERT architecture.

## Model Architecture

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

These new features are generated using `BKT` and clustering. It is the exact same as described in the [IKT paper](https://arxiv.org/pdf/2112.11209.pdf).

Now the data is split into interaction sequences of length 100. <br />
If a student has less than 100 interactions, the rest is padded with 0's. <br />
If a student has more that 100 interactions, these are split into multiple sequences.

After the BKT / clustering layer to generate additional features, the new and original features are passed to a BERT layer. <br />
This layer uses an encoder stack with a `MonotonicConvolutionalMultiheadAttention` attention layer. This is the same as in the original [MonaCoBERT paper](https://arxiv.org/abs/2208.12615).
This layer learns by masking and guessing 15% of the input, just as the original BERT paper by google.

The BERT layer uses embedding for all features and adds these together:
- `question embedding` +
- `response embedding` +
- `item embedding` +
- `ability profile embedding` +
- `problem difficulty embedding` +
- `skill mastery embedding` +
- `positional embedding` +

After the encoder stack and embedding, the input is passed to a feed forward layer for prediction.

## Hyperparameters

Here are the hyperparameters that I used to train the current model:
- `number of epochs` 50
- `learning rate` 0.001
- `optimizer` adam
- `crit` binary cross entropy
- `number of encoders` 12
- `hidden size` 512
- `number of attention heads` 8
