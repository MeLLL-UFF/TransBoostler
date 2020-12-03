
GOOGLE_WORD2VEC_PATH = 'word2vec/GoogleNews-vectors-negative300.bin'
WIKIPEDIA_FASTTEXT_PATH = 'fasttext/wiki.en.bin'
EMBEDDING_DIMENSION = 300
METHOD = 'CONCATENATE'

# Validation parameters
N_FOLDS = 3
AMOUNTS = [0.2, 0.4, 0.6, 0.8, 1.0]

# BoostSRL parameters
NODESIZE = 2
NUMOFCLAUSES = 8
MAXTREEDEPTH = 3
TREES = 10
SEED = 441773
REFINE = None
ALLOW_SAME_TARGET_MAP = False

# Filenames
TRANSFER_FILENAME = 'boostsrl/transfer.txt'
REFINE_FILENAME = 'boostsrl/refine.txt'
WILLTHEORIES_FILENAME = 'boostsrl/train/models/WILLtheories/{}_learnedWILLregressionTrees.txt'

TEST_OUTPUT = 'boostsrl/test/results_{}.db'
TEST_NEGATIVES = 'boostsrl/test/test_neg.txt'
TEST_POSITIVES = 'boostsrl/test/test_pos.txt'