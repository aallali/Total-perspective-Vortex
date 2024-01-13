
from predict import predict
from train import train_data, save_pipeline
from process import pre_process_data

if __name__ == "__main__":
    subjects = range(1, 2)
    experiments = [ 'do/hands', 'do/feets', 'imagine/hands', 'imagine/feet' ]
    args = {
        'subjects':  subjects,
        'predict': True, 
        'train': True, 
        'transformer': "CSP",  # or "SPoC",
        'experiment': 'do/hands'
    }
    for subjectID in args['subjects']:
        print(f"-----------------------[Subject {subjectID}]------------------------")
      
        [X, y] = pre_process_data(subjectID, args['experiment'])
        print(f"X shape: {X.shape} | Y shape: {y.shape}")

        if args['train']:
            pipelines = train_data(X, y, subjectID, args['transformer'])
            bestPipeline = {'cross_val_score': -1}
            for pipel in pipelines:
                cross_val_score = pipel[2].mean()
                pipeline_name = pipel[0]
                pipeline = pipel[1]
                print(f":--- {pipeline_name} : accuracy {cross_val_score.round(2)}")
                if cross_val_score > bestPipeline['cross_val_score']:
                    bestPipeline = {'name': pipeline_name, 'cross_val_score': cross_val_score, 'pipeline': pipeline}
            
            print((f":-- Best Pipeline is '{bestPipeline['name']}' : accuracy {bestPipeline['cross_val_score'].round(2)}"))
            save_pipeline(bestPipeline['pipeline'],X, y, subjectID)

        if args['predict']:
            prediction_result = predict(X, y , subjectID)
            print(f":-- Prediction accurracy: {prediction_result}")
        print("\n\n")