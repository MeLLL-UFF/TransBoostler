import numpy as np
import sys
import os

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
    d = d.split('(')
    d_predicate = d[0]

    if(',' in d[1]):
      d_literal_1 = d[1].split(',')[0]
      d_literal_2 = d[1].split(',')[1].split(')')[0]
    else:
      d_literal_1 = d[1].split(',')[0]
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
        temp = structure[key].split(', ')
        for t in temp:
          if((')' not in t and '(' in t) or ('(' in t and ')' in t)):
            preds = sweep_tree(t, preds) 
      else:
        preds = sweep_tree(structure[key], preds)
    return preds
  elif(isinstance(structure, str) and ("false" not in structure or "true" not in structure)):
    preds.append(structure)
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

def write_to_file(data, filename):
  """
      Write data to a specific file

      Args:
          data(list): information to be written
          filename(str): name of file in which the data will be written
  """
  with open(filename, 'w') as f:
      for line in data:
          f.write(line + '\n')
  f.close()