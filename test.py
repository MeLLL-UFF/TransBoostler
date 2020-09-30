from boostsrl import boostsrl

'''Step 1: Background Knowledge'''

# Sample data is built in from the 'Toy Cancer' Dataset, retrieve it with example_data
bk = boostsrl.example_data('background')

# Create the background knowledge or 'Modes,' where 'cancer' is the target we want to predict.
background = boostsrl.modes(bk, ['cancer'], useStdLogicVariables=True, treeDepth=4, nodeSize=2, numOfClauses=8)

'''Step 2: Training a Model'''

# Retrieve the positives, negatives, and facts.
train_pos = boostsrl.example_data('train_pos')
train_neg = boostsrl.example_data('train_neg')
train_facts = boostsrl.example_data('train_facts')

# Train a model using this data:
model = boostsrl.train(background, train_pos, train_neg, train_facts)

# How many seconds did training take?
#model.traintime()

'''Step 3: Test Model on New Data'''

# Retrieve the positives, negatives, and facts.
test_pos = boostsrl.example_data('test_pos')
test_neg = boostsrl.example_data('test_neg')
test_facts = boostsrl.example_data('test_facts')

# Test the data
results = boostsrl.test(model, test_pos, test_neg, test_facts)

'''Step 4: Observe Performance'''

# To see the overall performance of the model on test data:
print(results.summarize_results())



