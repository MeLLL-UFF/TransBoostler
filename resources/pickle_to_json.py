
import os, pickle
import utils as utils
import parameters as params
from experiments import experiments, bk, setups

def save_json_file(filename, data):
  """
    Save JSON file

    Args:
        filename(str): name of the file
  """
  def myconverter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

  with open(filename, 'w') as outfile:
    json.dump(data, outfile, default=myconverter)

def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


for experiment in experiments:
	_id = experiment['id']
	source = experiment['source']
	target = experiment['target']
	predicate = experiment['predicate']
	to_predicate = experiment['to_predicate']

	experiment_title = experiment['id'] + '_' + experiment['source'] + '_' + experiment['target']
	nodes = load_pickle_file(os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, params.SOURCE_TREE_NODES_FILES.replace('.json', '.pkl')))
	structured = load_pickle_file(os.getcwd() + '/resources/{}_{}_{}/{}'.format(_id, source, target, params.STRUCTURED_TREE_NODES_FILES).replace('.json', '.pkl'))

	utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.SOURCE_TREE_NODES_FILES), nodes)
	utils.save_json_file(os.getcwd() + '/experiments/{}_{}_{}/{}'.format(_id, source, target, params.STRUCTURED_TREE_NODES_FILES), structured)
