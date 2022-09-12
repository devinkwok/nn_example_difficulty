"""Metrics from
    Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and
    Gordon, G. J. (2019). An empirical study of example forgetting during deep
    neural network learning. In International Conference on Learning Representations.
as well as additional similar metrics.

Following functions assume dimensions (..., T, N) where T is steps (iterations)
"""
import numpy as np


def _forgetting_events(zero_one_accuracy: np.ndarray, is_forward: bool) -> np.ndarray:
    single_iteration = zero_one_accuracy[..., 0:1, :]
    zeros = np.zeros_like(single_iteration)
    ones = np.ones_like(single_iteration)
    start = zeros if is_forward else ones
    end = ones if is_forward else zeros
    accuracies = np.concatenate([start, zero_one_accuracy, end], axis=-2)
    accuracies = np.array(accuracies, dtype=int)
    diffs = accuracies[..., 1:, :] - accuracies[..., :-1, :]
    return diffs


def count_events_over_steps(events: np.ndarray, match) -> np.ndarray:
    return np.sum(events == match, axis=-2)


def _first_event_over_steps(events: np.ndarray, match) -> np.ndarray:
    matches = (events == match)
    # force match at T if there are no matches
    guaranteed_match = np.full_like(matches[..., 0:1, :], match)
    matches = np.concatenate([matches, guaranteed_match], axis=-2)
    # from np.argmax: In case of multiple occurrences of the maximum values
    # the indices corresponding to the first occurrence are returned.
    return np.argmax(matches, axis=-2)


def _last_event_over_steps(events: np.ndarray, match) -> np.ndarray:
    reversed = np.flip(events, axis=-2)
    backward_idx = _first_event_over_steps(reversed, match)
    return events.shape[-2] - 1 - backward_idx


def forgetting_events(zero_one_accuracy: np.ndarray) -> np.ndarray:
    return _forgetting_events(zero_one_accuracy, is_forward=True)


def count_forgetting(forgetting_events: np.ndarray) -> np.ndarray:
    """
    Args:
        forgetting_events (np.ndarray): output of forgetting_events() of shape (...,T+1, N)
            where T is the step

    Returns:
        np.ndarray: number of forgetting events
    """
    return count_events_over_steps(forgetting_events, -1)


def first_learn(forgetting_events: np.ndarray) -> np.ndarray:
    """
    Args:
        forgetting_events (np.ndarray): output of forgetting_events() of shape (...,T+1, N)
            where T is the step

    Returns:
        np.ndarray: iteration at which an example is first learned, from 0 to T inclusive
            0 means always learned, T means never learned
    """
    return _first_event_over_steps(forgetting_events, 1)


def is_unforgettable(forgetting_events: np.ndarray) -> np.ndarray:
    """
    Args:
        first_learn (np.ndarray): output of first_learn() of shape (...,T+1, N)
            where T is the step

    Returns:
        np.ndarray: 0 if example is forgotten, 1 if example is not forgotten (..., N)
    """
    t = forgetting_events.shape[-2] - 1
    learn = first_learn(forgetting_events)
    forget = count_forgetting(forgetting_events)
    return np.logical_and(learn < t, forget == 0)


def first_unforgettable(forgetting_events: np.ndarray) -> np.ndarray:
    """Also called iteration learned in
    Baldock, R., Maennel, H., and Neyshabur, B. (2021).
    Deep learning through the lens of example difficulty.
    Advances In Neural Information Processing Systems, 34.

    Returns:
        np.ndarray: step after which classification is always correct
    """
    return _last_event_over_steps(forgetting_events, 1)


def perturb_forgetting_events(zero_one_accuracy: np.ndarray) -> np.ndarray:
    return _forgetting_events(zero_one_accuracy, is_forward=False)


def perturb_first_forget(perturb_forgetting_events: np.ndarray) -> np.ndarray:
    return _first_event_over_steps(perturb_forgetting_events, -1)
