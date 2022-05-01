import csv
import json
import os


def csv_to_json(path):
    rows = []
    locations = {
        'desert': 0.,
        'mediterranean': 1.
    }
    # read csv file
    with open(path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        fields = next(csv_reader)
        for row in csv_reader:
            rows.append(row)

    # generate json file for each row in the appropriate directory
    for row in rows:
        data_json = {}
        time = row[0].split('_')[-1]
        time = int(time[:2]) * 6 + int(time[2:]) // 10
        data_json['time'] = float(time)
        for i in range(1, len(row)):
            if row[i] in locations:
                data_json[fields[i]] = locations[row[i]]
            else:
                data_json[fields[i]] = float(row[i]) if row[i] else 0.
        for option in ["", "_W", "_E"]:
            if row[0] + option in os.listdir("./resources"):
                with open("./resources/{dir}/station_data.json".format(dir=row[0] + option), 'w') as data_file:
                    # json.dump(data_json, data_file, indent=4, separators=(',', ': '))


def metrics(predictions, actuals):


    return accuarcy, average_diff