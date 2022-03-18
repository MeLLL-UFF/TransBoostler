import os

ROOT_PATH = os.getcwd() + '/'

FASTTEXT = 'fasttext'
WORD2VEC = 'word2vec'

GOOGLE_WORD2VEC = ROOT_PATH + 'resources/word2vec/GoogleNews-vectors-negative300.bin'
GOOGLE_WORD2VEC_SPACY = ROOT_PATH + 'resources/word2vec/spacy'

WIKIPEDIA_FASTTEXT = ROOT_PATH + 'resources/fasttext/wiki.en.vec'
WIKIPEDIA_FASTTEXT_SPACY = ROOT_PATH + 'resources/fasttext/spacy'

WORD_VECTOR_SIMILARITIES = ['cosine','euclidean']

USE_HUNGARIAN_METHOD = False

EMBEDDING_DIMENSION = 300
#METHOD = 'CONCATENATE'
METHOD = None

TOP_N = 2

# Validation parameters
N_FOLDS = 3
AMOUNTS = [0.2, 0.4, 0.6, 0.8, 1.0]
AMOUNTS_SMALL = [5, 10, 15, 20, 25]

# BoostSRL parameters
NODESIZE = 2
NUMOFCLAUSES = 8
MAXTREEDEPTH = 3
TREES = 10
SEED = 441773
REFINE = None
ALLOW_SAME_TARGET_MAP = False
SEARCH_PERMUTATION = True
SEARCH_EMPTY = False

# Filenames
TRANSFER_FILENAME = ROOT_PATH + 'boostsrl/transfer.txt'
REFINE_FILENAME = ROOT_PATH + 'boostsrl/refine.txt'
REFINE_REVISION_FILENAME = ROOT_PATH + 'boostsrl/refine-revision.txt'
WILLTHEORIES_FILENAME = ROOT_PATH + 'boostsrl/train/models/WILLtheories/{}_learnedWILLregressionTrees.txt'
SOURCE_TREE_NODES_FILES = 'source_tree_nodes.pkl'
STRUCTURED_TREE_NODES_FILES = 'source_structured_nodes.pkl'

TEST_OUTPUT = ROOT_PATH + 'boostsrl/test/results_{}.db'
TEST_NEGATIVES = ROOT_PATH + 'boostsrl/test/test_neg.txt'
TEST_POSITIVES = ROOT_PATH + 'boostsrl/test/test_pos.txt'

TRAIN_OUTPUT_FILE = ROOT_PATH + 'boostsrl/train_output.txt'
TEST_OUTPUT_FILE = ROOT_PATH + 'boostsrl/test_output.txt'
BACKGROUND_FILE = ROOT_PATH + 'boostsrl/background.txt'

MAX_REVISION_ITERATIONS = 1

#Folders
TRAIN_FOLDER = ROOT_PATH + 'boostsrl/train'
TEST_FOLDER = ROOT_PATH + 'boostsrl/test'
BEST_MODEL_FOLDER = ROOT_PATH + 'boostsrl/best'

TRAIN_FOLDER_FILES = ROOT_PATH + 'boostsrl/train/*'
TEST_FOLDER_FILES = ROOT_PATH + 'boostsrl/test/*'
BEST_MODEL_FOLDER_FILES = ROOT_PATH + 'boostsrl/best/*'
