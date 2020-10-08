# Text_Classifier

This program is used to classify essays as being written by a person who is either fluent or non-fluent in English.

The file trigram-model.py uses the class Trigram Model to train one model on the train_high.txt file, representing essays scored as "high" and thus written by a fluent English writer, and another model on train_low, which are essays written by non-fluent writers. The model uses unigrams, bigrams, and trigrams to calculate probabilities, and uses this information to calculate the perplexity of an unseen essay. The essay is classsified within the model that calculates the lower perplexity. The program then uses the essay in test_high.zip and test_low.zip to calculate the correctness of the model.

The program can be run using python3 without any command line arguments.
