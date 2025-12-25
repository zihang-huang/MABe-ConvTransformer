"""
Kaggle-compatible F Beta metric for MABe behavior recognition.

This module implements the exact same evaluation metric used by Kaggle
for the MABe Mouse Behavior Detection competition. Use this to get
accurate local scores before submitting to Kaggle.
"""

import json
from collections import defaultdict
from typing import Optional

import pandas as pd
import polars as pl


class HostVisibleError(Exception):
    """Error visible to competition host."""
    pass


def single_lab_f1(
    lab_solution: pl.DataFrame,
    lab_submission: pl.DataFrame,
    beta: float = 1
) -> float:
    """
    Compute F-beta score for a single lab's videos.

    This function computes segment-level F-beta by matching prediction frames
    to ground truth frames for each (agent_id, target_id, action) combination.

    Args:
        lab_solution: Ground truth annotations for one lab
        lab_submission: Predictions for the same lab's videos
        beta: Beta parameter for F-beta score (default=1 for F1)

    Returns:
        Macro-averaged F-beta score across all actions
    """
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    # Build ground truth frame sets for each (video, agent, target, action) key
    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    # Process predictions, respecting the behaviors_labeled filter
    for video in lab_solution['video_id'].unique():
        active_labels_str: str = lab_solution.filter(
            pl.col('video_id') == video
        )['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels_str))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            # Only evaluate predictions that are in the active labels for this video
            pred_key = ','.join([str(row['agent_id']), str(row['target_id']), row['action']])
            if pred_key not in active_labels:
                continue

            new_frames = set(range(row['start_frame'], row['stop_frame']))
            # Ignore truly redundant predictions
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])

            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                # A single agent can have multiple targets per frame but only one action per target
                raise HostVisibleError(
                    'Multiple predictions for the same frame from one agent/target pair'
                )

            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    # Compute TP, FN, FP for each action
    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)

    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    # Count FN for labels with no predictions
    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    # Compute per-action F-beta scores
    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            numerator = (1 + beta**2) * tps[action]
            denominator = (1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]
            action_f1s.append(numerator / denominator)

    return sum(action_f1s) / len(action_f1s) if action_f1s else 0.0


def mouse_fbeta(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    beta: float = 1
) -> float:
    """
    Compute the official MABe competition F-beta metric.

    This is the exact metric used by Kaggle for scoring submissions.
    It computes segment-level F-beta by matching prediction frames to
    ground truth frames for each (video_id, agent_id, target_id, action) key,
    then macro-averages across labs.

    Args:
        solution: Ground truth DataFrame with columns:
            - video_id, agent_id, target_id, action, start_frame, stop_frame
            - lab_id (for grouping)
            - behaviors_labeled (JSON list of valid agent,target,action combinations)
        submission: Prediction DataFrame with columns:
            - video_id, agent_id, target_id, action, start_frame, stop_frame
        beta: Beta parameter for F-beta score (default=1 for F1)

    Returns:
        Macro-averaged F-beta score across all labs

    Raises:
        ValueError: If solution or submission is empty or missing required columns
        HostVisibleError: If multiple predictions exist for same agent/target/frame
    """
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    # Convert to Polars for efficient processing
    solution_pl: pl.DataFrame = pl.DataFrame(solution)
    submission_pl: pl.DataFrame = pl.DataFrame(submission)

    # Validate frame ordering
    assert (solution_pl['start_frame'] <= solution_pl['stop_frame']).all()
    assert (submission_pl['start_frame'] <= submission_pl['stop_frame']).all()

    # Filter submission to only include videos in solution
    solution_videos = set(solution_pl['video_id'].unique())
    submission_pl = submission_pl.filter(pl.col('video_id').is_in(solution_videos))

    # Create composite keys for matching
    solution_pl = solution_pl.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission_pl = submission_pl.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    # Compute per-lab scores and macro-average
    lab_scores = []
    for lab in solution_pl['lab_id'].unique():
        lab_solution = solution_pl.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission_pl.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores) if lab_scores else 0.0


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    beta: float = 1
) -> float:
    """
    Compute F-beta score for MABe Challenge (Kaggle API format).

    This is the entry point used by Kaggle's evaluation system.

    Args:
        solution: Ground truth DataFrame
        submission: Prediction DataFrame
        row_id_column_name: Name of the row ID column to drop
        beta: Beta parameter for F-beta score

    Returns:
        F-beta score
    """
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)


def evaluate_submission(
    submission_df: pd.DataFrame,
    solution_df: pd.DataFrame,
    beta: float = 1,
    verbose: bool = True
) -> dict:
    """
    Evaluate a submission against the solution and return detailed metrics.

    This is a convenience wrapper that provides more detailed output
    for local evaluation and debugging.

    Args:
        submission_df: Prediction DataFrame with required columns
        solution_df: Ground truth DataFrame with required columns including lab_id
        beta: Beta parameter for F-beta score
        verbose: Whether to print progress information

    Returns:
        Dictionary containing:
            - 'score': Overall F-beta score
            - 'per_lab': Per-lab scores
            - 'num_predictions': Number of predictions made
            - 'num_ground_truth': Number of ground truth segments
    """
    if verbose:
        print(f"Evaluating submission with {len(submission_df)} predictions...")
        print(f"Ground truth has {len(solution_df)} segments across "
              f"{solution_df['lab_id'].nunique()} labs")

    try:
        overall_score = mouse_fbeta(solution_df, submission_df, beta=beta)
    except Exception as e:
        if verbose:
            print(f"Evaluation failed: {e}")
        return {
            'score': 0.0,
            'error': str(e),
            'num_predictions': len(submission_df),
            'num_ground_truth': len(solution_df)
        }

    # Compute per-lab scores
    per_lab = {}
    solution_pl = pl.DataFrame(solution_df)
    submission_pl = pl.DataFrame(submission_df)

    # Add keys
    solution_pl = solution_pl.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission_pl = submission_pl.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    for lab in solution_pl['lab_id'].unique():
        lab_solution = solution_pl.filter(pl.col('lab_id') == lab)
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission_pl.filter(pl.col('video_id').is_in(lab_videos))

        try:
            lab_score = single_lab_f1(lab_solution, lab_submission, beta=beta)
        except Exception:
            lab_score = 0.0

        per_lab[str(lab)] = lab_score

    if verbose:
        print(f"\nOverall F{beta} score: {overall_score:.4f}")
        print(f"Per-lab scores:")
        for lab, score in sorted(per_lab.items()):
            print(f"  {lab}: {score:.4f}")

    return {
        'score': overall_score,
        'per_lab': per_lab,
        'num_predictions': len(submission_df),
        'num_ground_truth': len(solution_df)
    }
