from parse_args import load_config
from predict import predict
from train import train_data, save_pipeline
from process import pre_process_data
from ENUMS import EXPERIMENTS
if __name__ == "__main__":
    args = load_config("config.yaml")
    print(args)
    print(EXPERIMENTS[args['EXPERIMENT']])
    # args = {
    #     'subjects':  subjects,
    #     'predict': True,
    #     'train': True,
    #     'transformer': "CSP",  # or "SPoC",
    #     'experiment': 'do/hands'
    # }
    for subjectID in args['SUBJECTS']:
        print(f"-----------------------[Subject {subjectID}]------------------------")
      
        [X, y, epochs] = pre_process_data(subjectID, EXPERIMENTS[args['EXPERIMENT']])
        print(f"X shape: {X.shape} | Y shape: {y.shape}")

        if args['MODE'] == "train" or args['MODE'] == "all":
            pipelines = train_data(X, y, args['TRANSFORMER'])
            bestPipeline = {'cross_val_score': -1}
            for pipel in pipelines:
                cross_val_score = pipel[2].mean()
                pipeline_name = pipel[0]
                pipeline = pipel[1]
                print(f":--- {pipeline_name} : accuracy {cross_val_score.round(2)}")
                # if cross_val_score > bestPipeline['cross_val_score']:
                #     bestPipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
                if pipeline_name == "LDA ":
                    bestPipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
            # print((f":-- Best Pipeline is '{bestPipeline['name']}' : accuracy {bestPipeline['cross_val_score'].round(2)}"))
            save_pipeline(bestPipeline['pipeline'],X, y, subjectID)

        if args['MODE'] == "predict" or args['MODE'] == "all":
            prediction_result = predict(X, y , subjectID)
            print(f":-- Prediction accurracy: {prediction_result}")
        print("\n\n")