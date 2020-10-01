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
