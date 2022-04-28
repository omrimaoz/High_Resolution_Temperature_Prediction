import csv
import json
import os


def csv_to_json(path):
    rows = []
    # read csv file
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        fields = next(csv_reader)
        for row in csv_reader:
            rows.append(row)

    # generate json file for each row in the appropriate directory
    for row in rows:
        data_json = {}
        for i in range(1, len(row)):
            data_json[fields[i]] = row[i]
        for option in ["", "_W", "_E"]:  # TODO: maybe make it more elegant
            if row[0] + option in os.listdir("./resources"):
                with open("./resources/{dir}/station_data.json".format(dir=row[0] + option), 'w') as data_file:
                    json.dump(data_json, data_file, indent=4, separators=(',', ': '))



csv_to_json("./resources/data_table.csv")