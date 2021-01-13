from sklearn.metrics import confusion_matrix
import numpy as np
import subprocess
import glob
import json
import sys
import os
import re

def build_triples(data):
  """
      Splits predicates and its literals

      Args:
          data(str): relation to be split
      Returns:
          relation as a three-element array

      Example:
          movie(A) -> movie, A, ''
          father(A,B) -> father, A, B
  """
  output = []
  for d in data:
    d = d.replace('.','').split('(')
    d_predicate = d[0]

    if(',' in d[1]):
      d_literal_1 = d[1].split(',')[0].replace('(', '').replace('+', '').replace('-', '')
      d_literal_2 = d[1].split(',')[1].split(')')[0].replace('+', '').replace('-', '')
    else:
      d_literal_1 = d[1].split(',')[0].replace(')', '').replace('+', '').replace('-', '')
      d_literal_2 = ''

    output.append([d_predicate, d_literal_1, d_literal_2])
  return output

def sweep_tree(structure, preds=[]):
  """
      Sweep through the relational tree 
      to get all predicates learned 
      using recursion

      Args:
          structure(list/dict/str/float): something to be added to the list
      Returns:
          all predicates learned by the model
  """
  if(isinstance(structure, list)):
    for element in structure:
      preds = sweep_tree(element, preds)
    return preds
  elif(isinstance(structure, dict)):
    for key in structure:
      if(isinstance(structure[key], str)):
        temp = re.split(r',\s*(?![^()]*\))', structure[key])
        for t in temp:
          t = t.split('(')[0]	
          preds = sweep_tree(t, preds) 
      else:
        preds = sweep_tree(structure[key], preds)
    return preds
  elif(isinstance(structure, str) and ("false" not in structure or "true" not in structure)):
    preds.append(structure.split('(')[0])
    #preds.append(structure)
    return preds
  else:
    return preds

def get_next_node(node, next):
  """
      Add next nodes of tree to list of rules

      Args:
          node(list): current branch to be added to list
          next(list): next branch of tree given the current node
      Returns:
          all rules from a node
  """

  if not node:
    return next
  b = node.split(',')
  b.append(next)
  return ','.join(b)


def get_rules(structure, treenumber=1):
  """
      Sweep through a branch of the tree 
      to get all rules

      Args:
          structure(list): tree struct
          treenumber(int): number of the tree to be processed
      Returns:
          all rules learned in the given branch
  """

  target = structure[0]
  nodes = structure[1]
  tree = treenumber-1

  rules = []
  for path, value in nodes.items():
    node = target + ' :- ' + value + '.' if not path else value + '.'
    true =  'true'  if get_next_node(path, 'true')  in nodes else 'false'
    false = 'true'  if get_next_node(path, 'false') in nodes else 'false'
    rules.append(';'.join([str(tree), path, node, true, false]))
  return rules

def get_all_rules_from_tree(structures):
  """
      Sweep through the relational tree 
      to get all relational rules

      Args:
          structure(list): tree struct
      Returns:
          all rules learned by the model
  """

  rules = []
  for i in range(len(structures)):
    rules += get_rules(structures[i], treenumber=i+1)
  return rules

def write_to_file(data, filename, op='w'):
  """
      Write data to a specific file

      Args:
          data(list): information to be written
          filename(str): name of file in which the data will be written
          op(str): 'w' to create a new file or 'a' to append data to a new file if exists
  """
  with open(filename, op) as f:
      for line in data:
          f.write(line + '\n')

def read_file(filename):
  """
      Read data from a specific file

      Args:
          filename(str): name of file in which is the data to be read
      Returns:
          data(array): arrays containing all information found in file
  """
  f = open(filename, 'r')
  data = f.readlines()
  f.close()

  return data

def fill_dimension(source, target, dimension):
  """
     Fix source and target to be the same dimension

      Args:
          source(list): source embedding vector
          target(str): target embedding vector
          dimension(int): size of dimension of embedding vector
  """

  if(source.shape[0] > target.shape[0]):
    zeros = np.zeros(source.shape[0] - target.shape[0])
    return source, np.append(target, zeros)
  elif(source.shape[0] < target.shape[0]):
    zeros = np.zeros(target.shape[0] - source.shape[0])
    return np.append(source, zeros), target
  else:
    print("Something went wrong while filling dimensions") 

def fill_missing_dimensions(source, target, dimension):
  """
     Add zero arrays to source and target belong to the same feature space

      Args:
          source(list): source embedding vector
          target(str): target embedding vector
          dimension(int): size of dimension of embedding vector
  """
  if(len(source) > len(target)):
    temp = [[0]* dimension] * len(source)
    for i in range(len(target)):
      temp[i] = target[i].copy()
    return source, temp
  elif(len(target) > len(source)):
    temp = [[0]* dimension] * len(target)
    for i in range(len(target)):
      temp[i] = source[i].copy()
    return temp, target
  else:
    print("Something went wrong while fixing space of word vector")

def convert_db_to_txt(predicate, path):
  """
     Converts the db file containing test outputs to txt

      Args:
          predicate(str): name of the predicate to be learned
          path(str): path to text file
  """
  cmd = 'less {} > {}'
  process = subprocess.Popen(cmd.format(path.format(predicate), path.format(predicate).replace('.db', '.txt')), shell=True)
  output, error = process.communicate()

  if(error):
    print('Something went wrong while converting db file to txt file')


def read_results(filename):
  """
     Reads the file containing test results 

      Args:
          filename(str): the name of the file
      Returns:
          y_true(array): real values of each test example
          y_pred(array): predicted values of each test example
  """
  y_true, y_pred = [], []
  with open(filename, 'r') as file:
    for line in file:
      example, score = line.replace(', ', ',').split()

      if('!' in example):
        y_true.append(0)
        boolean = 0 if float(score) > 0.500 else 1
        y_pred.append(boolean)
      else:
        y_true.append(1)
        boolean = 1 if float(score) > 0.500 else 0
        y_pred.append(boolean)
  return y_true, y_pred

def get_confusion_matrix(y_true, y_pred):
  """
     Returns the confusion matrix for each experiment

      Args:
          y_true(array): real values of each test example
          y_pred(array): predicted values of each test example
      Returns:
          confusion matrix
  """
  # True Negatives, False Positives, False Negatives, True Positives
  return confusion_matrix(y_true, y_pred).ravel()

def delete_folder_files(folder_name):
  """
    Deletes files from a specific folder

    Args:
        folder_name(str): name of the folder to empty
  """
  files = glob.glob(folder_name)
  for f in files:
      try:
        os.remove(os.getcwd() + '/' + f)
      except PermissionError as e:
        print('In utils, delete_folder_files function: ', e)
        os.rmdir(os.getcwd() + '/' + folder_name)


def delete_file(filename):
  """
    Deletes file

    Args:
        filename(str): name of the file to be deleted
  """
  try:
    os.remove(os.getcwd() + '/' + filename)
  except FileNotFoundError as e:
    print('In utils, delete_file function: ', e)

def save_json_file(filename, data):
  """
    Save JSON file

    Args:
        filename(str): name of the file
  """
  with open(filename, 'w') as outfile:
    json.dump(data, outfile)

#y_true, y_pred = read_results('boostsrl/test/results_{}.db'.format('advisedby'))
#print(get_confusion_matrix(y_true, y_pred))

#print(len(y_true), len(y_pred))