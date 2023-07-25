import os

def decomment(csvfile):
    for row in csvfile:
        raw = row.split('#')[0].strip()
        if raw: yield raw

def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

