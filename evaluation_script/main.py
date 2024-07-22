import pandas as pd
import json

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    """
    Evaluates the submission for a particular challenge phase and returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    # Load ground truth and user submission
    ground_truth = pd.read_csv(test_annotation_file)
    submission = pd.read_csv(user_submission_file)

    # Ensure both files have the same length
    if len(ground_truth) != len(submission):
        raise ValueError("Mismatch in number of records between ground truth and submission")

    # Calculate accuracy
    accuracy = (ground_truth['class3'] == submission['class3']).mean()

    # Prepare the result dictionary
    output = {
        "result": [
            {
                "split": "dev_split" if phase_codename == "dev_phase" else "test_split",
                "show_to_participant": True,
                "accuracies": {"Accuracy": accuracy * 100},
            }
        ],
        "submission_result": {"Accuracy": accuracy * 100},
    }

    print("Completed evaluation")
    return output
