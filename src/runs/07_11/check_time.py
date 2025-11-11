import json
import sys
import os
import datetime


def main():
    directory = sys.argv[1]

    total, n_files = 0, 0
    for fname in os.listdir(directory):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(directory, fname)) as fp:
            data = json.load(fp)
        # data = json.load(os.path.join(directory, fname))
        hms = data['training_time']['training'].split(':')

        (h, m, s) = hms if len(hms) == 3 else (0, *hms)
        elapsed_train = datetime.timedelta(hours=float(h), minutes=float(m), seconds=float(s))

        hms_eval = data['training_time']['evaluation'].split(':')
        (h, m, s) = hms if len(hms) == 3 else (0, *hms)
        elapsed_eval = datetime.timedelta(hours=float(h), minutes=float(m), seconds=float(s))
        # ela
        # elapsed_train = datetime.datetime.strptime(data['training_time']['training'], '%H:%M:%S')
        # elapsed_eval = datetime.datetime.strptime(data['training_time']['evaluation'], '%H:%M:%S')
        # elapsed = elapsed_train + elapsed_eval
        # print(elapsed_train, type(elapsed_train))
        total += elapsed_train.total_seconds() #+ elapsed_eval.total_seconds()
        n_files += 1

    total = datetime.timedelta(seconds=total)
    average = total / n_files
    # total_str = str(total).split('.', maxsplit=1)[0]
    average_str = str(average).split('.', maxsplit=1)[0]
    
    print('total hours:', total.total_seconds() / 3600)
    print('average:', average_str)
    print('n files:', n_files)





if __name__ == '__main__':
    main()
