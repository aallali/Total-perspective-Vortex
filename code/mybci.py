from parse_args import load_config
from predict import predict
from train import train_data, save_pipeline
from process import pre_process_data
from ENUMS import EXPERIMENTS
import numpy as np

if __name__ == "__main__":
    args = load_config("config.yaml")
    print(args)

    print(f"Experiment in study: ({EXPERIMENTS[args['EXPERIMENT']][0]}) <--VS--> ({EXPERIMENTS[args['EXPERIMENT']][1]})")
    CALC_MEAN_FOR_ALL = True if len(args['SUBJECTS']) > 1 else False

    cross_val_scores = []
    accuracy_scores = []

    for subjectID in args['SUBJECTS']:
        print(f"----------------------------------------------[Subject {subjectID}]")
      
        [X, y, epochs] = pre_process_data(subjectID, EXPERIMENTS[args['EXPERIMENT']])
        # print(f"X shape: {X.shape} | Y shape: {y.shape}")

        if args['MODE'] == "train" or args['MODE'] == "all":
            pipelines = train_data(X=X, y=y, transformer=args['TRANSFORMER'], run_all_pipelines=False)
            bestPipeline = {'cross_val_score': -1}
            for pipel in pipelines:
                cross_val_score = pipel[2].mean()
                pipeline_name = pipel[0]
                pipeline = pipel[1]
                print(f":--- {pipeline_name} cross_val_score : {cross_val_score.round(2)}")
                if cross_val_score > bestPipeline['cross_val_score']:
                    bestPipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
                # if pipeline_name == "LDA ":
                #     bestPipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
            # print((f":-- Best Pipeline is '{bestPipeline['name']}' : accuracy {bestPipeline['cross_val_score'].round(2)}"))
            if CALC_MEAN_FOR_ALL:
                cross_val_scores.append(cross_val_score)
            save_pipeline(bestPipeline['pipeline'], X, y, subjectID, args['EXPERIMENT'])

        if args['MODE'] == "predict" or args['MODE'] == "all":
            prediction_result = predict(X, y , subjectID, args['EXPERIMENT'], not CALC_MEAN_FOR_ALL)
            print(f":--- Prediction accurracy: {'{:.2%}'.format(prediction_result).rstrip('0').rstrip('.')}")
            if CALC_MEAN_FOR_ALL:
                accuracy_scores.append(prediction_result)

    if CALC_MEAN_FOR_ALL:
        print("----------------------------[Mean Scores for all subjects]----------------------------")
        if len(cross_val_scores) > 1:
            print(f":--- Mean cross_val : {np.mean(cross_val_scores).round(2)}")
        if len(accuracy_scores) > 1:
            print(f":--- Mean accuracy  : {np.mean(accuracy_scores).round(2)}")

