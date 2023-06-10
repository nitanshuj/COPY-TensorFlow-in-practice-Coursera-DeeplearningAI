# **Notes**



### <font color='purple'> **Questions based on Reasons:**</font>

**`Question 1: When predicting words to generate poetry, the more words predicted the more likely it will end up gibberish. Why?`** <br>
***Answer:*** Because the probability that each word matches an existing phrase goes down the more words you create

**`Question 2: What is a major drawback of word-based training for text generation instead of character-based generation?`**<br>
***Answer:***
Because there are far more words in a typical corpus than characters, it is much more memory intensive


**`Question 3: What are the critical steps in preparing the input sequences for the prediction model?`**<br>
***Answers:***
- Pre-padding the subprhases sequences.
- Generating subphrases from each line using n_gram_sequences.

**`Question 4: In  natural language processing, predicting the next item in a sequence is a classification problem.Therefore, after creating inputs and labels from the subphrases, we one-hot encode the labels. What function do we use to create one-hot encoded arrays of the labels?`**<br>
***Answers:*** ```tf.keras.utils.to_categorical```

**`Question 5: When building the model, we use a sigmoid activated Dense output layer with one neuron per word that lights up when we predict a given word.`**<br>
***Answers: FALSE***


