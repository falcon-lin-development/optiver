import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import sys

def _test_amount_of_data(test_df):
    counts = test_df.groupby(["date_id", "seconds_in_bucket"]).size().reset_index(name="count")
    return counts.shape[0]

def test_generator(test_df):
    # print("new")
    """Yields batches of rows from the dataframe."""
    # for time_id in test_df["time_id"].unique():
    for (date_id, seconds_in_bucket), group in test_df.groupby([ "date_id", "seconds_in_bucket"]): 
        # _test = test_df.loc[test_df["time_id"] == time_id]
        # _test = _test.drop(["time_id"], axis=1)
        _test = group
        _sample_submission = _test[ ["row_id"] ]
        _sample_submission.loc[:, ["target"]] = 1
        yield _test, None, _sample_submission



def mock_inference(test_df, inferencer, fast=False):
    if fast:
        _sample_prediction = pd.DataFrame({
            'row_id': test_df['row_id'],
            'target': 1
        })
        prediction = inferencer.predict(test_df, None, _sample_prediction)
        return prediction

    submission = None
    def add_submission(submission_df):
        nonlocal submission
        if submission is None:
            submission = [submission_df]
        else:
            submission.append(submission_df)

    counter = 0
    total = _test_amount_of_data(test_df)
    for (_test, _, _sample_prediction) in test_generator(test_df):    
        counter += 1
        sys.stdout.write(f"\r{counter}/{total}")
        prediction = inferencer.predict(_test, _, _sample_prediction)
        add_submission(prediction)
    
    print()
    return pd.concat(
                [
                    *submission
                ]
                , ignore_index=True
            )

