import numpy as np
import pandas as pd
from sklearn.externals import joblib
import sys

class EmployeeRetentionModel:
    
    def __init__(self, model_location, train_mean, train_std):
        '''
           model_location -> will be the file location of the saved final model.
           train_mean -> location of mean of the features in training data
           train_std -> location of standard deviation of the features in training data
        '''
        # Load the model
        self.model = joblib.load(model_location)
        
        # Load the train mean and train std dev
        self.train_mean = pd.read_pickle(train_mean)
        self.train_std = pd.read_pickle(train_std)
    
    def predict_proba(self, X_new, clean=True, augment=True):
        if clean:
            X_new = self.clean_data(X_new)
        
        if augment:
            X_new = self.engineer_features(X_new)
        
        ## Standardizing the data
        X_new = (X_new - self.train_mean)/ self.train_std
        
        return X_new, self.model.predict_proba(X_new)[:, 1]
    
    # Add functions here
    def clean_data(self, df):
        # Drop duplicates
        df = df.drop_duplicates()

        # Drop temporary workers
        df = df[df.department != 'temp']

        # Missing filed_complaint values should be 0
        df['filed_complaint'] = df.filed_complaint.fillna(0)

        # Missing recently_promoted values should be 0
        df['recently_promoted'] = df.recently_promoted.fillna(0)

        # 'information_technology' should be 'IT'
        df.department.replace('information_technology', 'IT', inplace=True)

        # Fill missing values in department with 'Missing'
        df['department'].fillna('Missing', inplace=True)

        # Indicator variable for missing last_evaluation
        df['last_evaluation_missing'] = df.last_evaluation.isnull().astype(int)

        # Fill missing values in last_evaluation with 0
        df.last_evaluation.fillna(0, inplace=True)

        # Return cleaned dataframe
        return df
    
    def engineer_features(self, df):
        # Create indicator features
        df['underperformer'] = ((df.last_evaluation < 0.6) & 
                                (df.last_evaluation_missing == 0)).astype(int)

        df['unhappy'] = (df.satisfaction < 0.2).astype(int)

        df['overachiever'] = ((df.last_evaluation > 0.8) & (df.satisfaction > 0.7)).astype(int)

        # Create new dataframe with dummy features
        df = pd.get_dummies(df, columns=['department', 'salary'])

        # Return augmented DataFrame
        return df


def main(data_location, output_location, model_location, train_mean, train_std, clean=True, augment=True):
    # Read dataset
    df = pd.read_csv(data_location)

    # Initialize model
    retention_model = EmployeeRetentionModel(model_location, train_mean, train_std,)

    # Make prediction
    df, y_pred_proba = retention_model.predict_proba(df)

    # Add prediction to dataset
    df['prediction'] = y_pred_proba

    # Save dataset after making predictions
    df.to_csv(output_location, index=None)

if __name__ == '__main__':
    main( *sys.argv[1:] )  ## argument vector