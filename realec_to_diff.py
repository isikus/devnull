# -*- coding: utf-8 -*-

from collections import OrderedDict

class AnnEntry:
  def __new__(self, ann_line):
    line = ann_line.split("\t")
    line = [line[0]] +line[1].split(" ") + line[2:]
    return line

class TextPatch:
  def __init__(self, ann_entry):
    self.start = int(ann_entry[2])
    self.end = int(ann_entry[3])
    self.err_type = ann_entry[1]
    self.orig_str = None
    self.corr_str = None
    self.name = ann_entry[0]
    if self.name[0] == "T":
      self.orig_str = ann_entry[4]

def rectify_patch(patch_dict):
  pd = patch_dict
  patch_list = sorted(pd.items(), key=lambda k: (k[1].start, k[1].end, int(k[0][1:])))
  del_i = []
  for i, el in enumerate(patch_list[1:]):
    prev = patch_list[i][1]
    curr = el[1]
    if prev.start == curr.start and prev.end == curr.end:
      del_i.append(i)
    elif prev.end >= curr.end:
      del_i.append(i+1)
    elif prev.start < curr.start and prev.end < curr.end and prev.end > curr.start:
      del_i.append(i)
  del_i = sorted(list(set(del_i)), reverse=True)
  for i in del_i:
    del patch_list[i]
  patch_list = [el[1] for el in patch_list]
  return patch_list

def pl_to_fl(patch_list):
  fix_list = []
  for patch in patch_list:
    fix_list.append([patch.start, patch.end, patch.corr_str])
  return fix_list

def ann_to_fl(ann_lines):
  ann_lines = [AnnEntry(line) for line in ann_lines]
  patch_dict = OrderedDict()
  for entry in ann_lines:
    if entry[0][0] == "T":
      tp = TextPatch(entry)
      patch_dict[tp.name] = tp
    elif entry[0][0] == "#":
      patch_dict[entry[2]].corr_str = entry[3]
    elif entry[0][0] == "A" and entry[1] == "Delete":
      patch_dict[entry[2]].corr_str = ""
    else:
      pass
  patch_list = rectify_patch(patch_dict)
  fix_list = pl_to_fl(patch_list)
  return fix_list
