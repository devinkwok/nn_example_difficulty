"""Metrics from
    Toneva, M., Sordoni, A., des Combes, R. T., Trischler, A., Bengio, Y., and
    Gordon, G. J. (2019). An empirical study of example forgetting during deep
    neural network learning. In International Conference on Learning Representations.
as well as additional similar metrics.

Following functions assume dimensions (..., T, N) where T is steps (iterations)
"""
import torch


def _change_events(zero_one_accuracy: torch.Tensor, is_forward: bool, is_falling: bool) -> torch.Tensor:
    single_iteration = zero_one_accuracy[..., 0:1, :]
    zeros = torch.zeros_like(single_iteration)
    ones = torch.ones_like(single_iteration)
    start = zeros if is_forward else ones
    end = ones if is_forward else zeros
    accuracies = torch.concatenate([start, zero_one_accuracy, end], dim=-2)
    prev = accuracies[..., :-1, :]
    next = accuracies[..., 1:, :]
    if is_falling:  # events transitioning from 1 to 0
        return torch.logical_and(prev, torch.logical_not(next))
    else:  # 0 to 1 (always guaranteed to occur at least once)
        return torch.logical_and(torch.logical_not(prev), next)


def forgetting_events(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    return _change_events(zero_one_accuracy, is_forward=True, is_falling=True)


def learning_events(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    return _change_events(zero_one_accuracy, is_forward=True, is_falling=False)


def perturb_forgetting_events(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    return _change_events(zero_one_accuracy, is_forward=False, is_falling=True)


def perturb_learning_events(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    return _change_events(zero_one_accuracy, is_forward=False, is_falling=False)


def count_events(events: torch.Tensor) -> torch.Tensor:
    return torch.count_nonzero(events, dim=-2)


def first_event_time(events: torch.Tensor) -> torch.Tensor:
    # force match at T if there are no matches
    guaranteed_event = torch.ones_like(events[..., 0:1, :])
    # make numeric to allow use of argmax
    events_plus_extra = 1*torch.concatenate([events, guaranteed_event], dim=-2)
    # from torch.argmax: In case of multiple occurrences of the maximum values
    # the indices corresponding to the first occurrence are returned.
    first_event = torch.argmax(events_plus_extra, dim=-2)
    return first_event


def last_event_time(events: torch.Tensor) -> torch.Tensor:
    reversed = torch.flip(events, dims=[-2])
    backward_idx = first_event_time(reversed)
    last_event = events.shape[-2] - 1 - backward_idx
    return last_event


def first_learn(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    """Equivalent to first_event_time(learning_events(zero_one_accuracy))

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T+1, N) where T is the step

    Returns:
        torch.Tensor: iteration at which an example is first learned, from 0 to T inclusive
            0 means always learned, T means never learned
    """
    events = learning_events(zero_one_accuracy)
    return first_event_time(events)


def is_unforgettable(learning_events: torch.Tensor, forgetting_events: torch.Tensor) -> torch.Tensor:
    """
    Args:
        learning_events (torch.Tensor): output of learning_events() or perturb_learning_events()
            with shape (...,T+1, N) where T is the step
        forgetting_events (torch.Tensor): output of forgetting_events() or perturb_forgetting_events()
            with shape (...,T+1, N) where T is the step

    Returns:
        torch.Tensor: 0 if example is forgotten, 1 if example is not forgotten (..., N)
    """
    t = learning_events.shape[-2] - 1
    learn = first_learn(learning_events)
    forget = count_forgetting(forgetting_events)
    unforgettable = torch.logical_and(learn < t, forget == 0)
    return unforgettable


def count_forgetting(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    """Equivalent to count_events(forgetting_events(zero_one_accuracy))

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T+1, N) where T is the step

    Returns:
        torch.Tensor: number of forgetting events
    """
    events = forgetting_events(zero_one_accuracy)
    return count_events(events)


def first_unforgettable(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    """Equivalent to last_event_time(learning_events(zero_one_accuracy))

    Also called iteration learned in
    Baldock, R., Maennel, H., and Neyshabur, B. (2021).
    Deep learning through the lens of example difficulty.
    Advances In Neural Information Processing Systems, 34.

    Args:
        zero_one_accuracy (torch.Tensor): output of pointwise.zero_one_accuracy(),
            bool tensor of shape (...,T+1, N) where T is the step

    Returns:
        torch.Tensor: step after which classification is always correct
    """
    events = learning_events(zero_one_accuracy)
    return last_event_time(events)


def perturb_first_forget(zero_one_accuracy: torch.Tensor) -> torch.Tensor:
    events = perturb_forgetting_events(zero_one_accuracy)
    return first_event_time(events)
