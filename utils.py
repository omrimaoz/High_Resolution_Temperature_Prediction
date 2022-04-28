import csv
import json


def csv_to_json(path):
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        fields = next(csv_reader)
        for row in csv_reader:
            data_json = {}


csv_to_json("./resources/")