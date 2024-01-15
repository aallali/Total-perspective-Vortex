from parse_args import load_config
from predict import predict
from train import train_data, save_pipeline
from process import pre_process_data
from ENUMS import EXPERIMENTS
import numpy as np

import concurrent.futures
import json
import hashlib
import time
def hash_list_secure(my_list):
    # Sort the list and convert it to a tuple
    sorted_tuple = tuple(sorted(my_list))
    # Use SHA-256 for a secure hash
    hash_object = hashlib.sha256(str(sorted_tuple).encode())
    return hash_object.hexdigest()


def process_subject(subjectID, args, isSingleSubject=False):
    # Record the start time
    start_time_inner = time.time()
    [X, y, epochs] = pre_process_data(subjectID, EXPERIMENTS[args['EXPERIMENT']])

    result_inner = [0, 0]
    output = []
    output.append(f"----------------------------------------------[Subject {subjectID}]")
    stats = {
        'subject_id': subjectID,
        'pipelines': [],
        'cross_val_score': 0,
        'accuracy': 0,
        'time_unit': "seconds"
    }
    if args['MODE'] == "train" or args['MODE'] == "all":
        pipelines = train_data(X=X, y=y, transformer=args['TRANSFORMER'], run_all_pipelines=False)
        best_pipeline = {'cross_val_score': -1}

        for pipel in pipelines:
            cross_val_score = pipel[2].mean()
            pipeline_name = pipel[0]
            pipeline = pipel[1]
            output.append(f":--- [S{subjectID}] {pipeline_name} cross_val_score : {cross_val_score.round(2)}")

            if cross_val_score > best_pipeline['cross_val_score']:
                best_pipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
            stats['pipelines'].append((pipeline_name, cross_val_score))
        save_pipeline(best_pipeline['pipeline'], X, y, subjectID, args['EXPERIMENT'])
        result_inner[0] = best_pipeline['cross_val_score']
        stats['cross_val_score'] = result_inner[0]

    if args['MODE'] == "predict" or args['MODE'] == "all":
        prediction_result = predict(X, y, subjectID, args['EXPERIMENT'], isSingleSubject)
        output.append(
            f":--- [S{subjectID}] Prediction accurracy: {'{:.2%}'.format(prediction_result).rstrip('0').rstrip('.')}")
        result_inner[1] = prediction_result
        stats['accuracy'] = result_inner[1]

    # Record the end time
    end_time_inner = time.time()

    # Calculate the time cost
    time_cost_inner = end_time_inner - start_time_inner
    stats['time_cost'] = time_cost_inner
    print(*output, sep="\n")
    print(f":--- [S{subjectID}] time cost: {stats['time_cost']} seconds")
    return stats

def main():
    # Record the start time
    start_time = time.time()
    args = load_config("config.yaml")
    print(args)

    print(
        f"Experiment in study: ({EXPERIMENTS[args['EXPERIMENT']][0]}) <--VS--> ({EXPERIMENTS[args['EXPERIMENT']][1]})")
    CALC_MEAN_FOR_ALL = True if len(args['SUBJECTS']) > 1 else False

    cross_val_scores = []
    accuracy_scores = []
    final_stats = {
        'subjects_hash': "all" if len(args['SUBJECTS']) == 109 else ''.join(map(str, args['SUBJECTS'])),
        'config': args,
        'events': EXPERIMENTS[args['EXPERIMENT']],
        'subjects': []
    }
    print(final_stats)
    max_workers = 3  # Adjust this value based on your CPU and system resources
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process each subject in parallel with a maximum of max_workers threads
        results = executor.map(lambda subjectID: process_subject(subjectID, args, isSingleSubject=not CALC_MEAN_FOR_ALL),
                               args['SUBJECTS'], )

    # Collect results from parallel processing

    for result in results:
        if args['MODE'] == "train" or args['MODE'] == "all":
            cross_val_scores.append(result['cross_val_score'])
        if args['MODE'] == "predict" or args['MODE'] == "all":
            accuracy_scores.append(result['accuracy'])

        final_stats['subjects'].append(result)
    if CALC_MEAN_FOR_ALL:
        print("\n----------------------------[Mean Scores for all subjects]----------------------------")
        if len(cross_val_scores) > 1:
            print(f":--- Mean cross_val : {np.mean(cross_val_scores).round(2)}")
            final_stats['mean_cross_val_score'] = np.mean(cross_val_scores)
        if len(accuracy_scores) > 1:
            print(f":--- Mean accuracy  : {np.mean(accuracy_scores).round(2)}")
            final_stats['mean_accuracy'] = np.mean(accuracy_scores)


    # Record the end time
    end_time = time.time()

    # Calculate the time cost
    time_cost = end_time - start_time
    final_stats['time_cost'] = time_cost
    if args['MODE'] == "all":
        results_filename = \
            f"../data/results/results-{args['EXPERIMENT']}-{time.time()}-{args['TRANSFORMER']}-{final_stats['subjects_hash']}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(final_stats, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()