from tqdm import tqdm
import torch
from utils import get_parameter_groups, get_lr_scheduler_with_warmup


def prepare_for_training(model, args):
    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    gradscaler = torch.amp.GradScaler(enabled=args.amp)
    scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                             num_steps=args.max_steps, warmup_proportion=args.warmup_proportion)

    return optimizer, gradscaler, scheduler


def train_step_full_graph_transductive(model, dataset, optimizer, scheduler, gradscaler, amp=False):
    model.train()

    with torch.autocast(enabled=amp, device_type=dataset.graph.device.type):
        preds = model(graph=dataset.graph, x=dataset.features)
        loss = dataset.loss_fn(input=preds[dataset.train_mask], target=dataset.targets[dataset.train_mask])

    gradscaler.scale(loss).backward()
    gradscaler.step(optimizer)
    gradscaler.update()
    optimizer.zero_grad()
    scheduler.step()


def train_step_full_graph_inductive(model, dataset, optimizer, scheduler, gradscaler, amp=False):
    model.train()

    with torch.autocast(enabled=amp, device_type=dataset.train_graph.device.type):
        preds = model(graph=dataset.train_graph, x=dataset.train_features)
        loss = dataset.loss_fn(input=preds[dataset.train_mask], target=dataset.train_targets[dataset.train_mask])

    gradscaler.scale(loss).backward()
    gradscaler.step(optimizer)
    gradscaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate_full_graph_transductive(model, dataset, amp=False):
    model.eval()

    with torch.autocast(enabled=amp, device_type=dataset.graph.device.type):
        preds = model(graph=dataset.graph, x=dataset.features)

    metrics = dataset.compute_metrics_transductive(preds)

    return metrics


@torch.no_grad()
def evaluate_full_graph_inductive(model, dataset, best_prev_val_metric, amp=False):
    model.eval()
    metrics = {}

    with torch.autocast(enabled=amp, device_type=dataset.val_graph.device.type):
        preds = model(graph=dataset.val_graph, x=dataset.val_features)

    metrics[f'val {dataset.metric_name}'] = dataset.compute_val_metric_inductive(preds)

    if metrics[f'val {dataset.metric_name}'] > best_prev_val_metric:
        with torch.autocast(enabled=amp, device_type=dataset.test_graph.device.type):
            preds = model(graph=dataset.test_graph, x=dataset.test_features)

        metrics[f'test {dataset.metric_name}'] = dataset.compute_test_metric_inductive(preds)

    return metrics


def update_results(results, new_metrics, metric_name, step):
    if new_metrics[f'val {metric_name}'] > results[f'val {metric_name}']:
        results[f'val {metric_name}'] = new_metrics[f'val {metric_name}']
        results[f'test {metric_name}'] = new_metrics[f'test {metric_name}']
        results['step'] = step

    return results


def train_full_graph_transductive(model, dataset, args, run_id):
    optimizer, gradscaler, scheduler = prepare_for_training(model=model, args=args)
    results = {f'val {dataset.metric_name}': 0, f'test {dataset.metric_name}': 0, 'step': None}
    with tqdm(total=args.max_steps, desc=f'Run {run_id}') as progress_bar:
        for step in range(1, args.max_steps + 1):
            train_step_full_graph_transductive(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                                               gradscaler=gradscaler, amp=args.amp)
            metrics = evaluate_full_graph_transductive(model=model, dataset=dataset, amp=args.amp)
            results = update_results(results=results, new_metrics=metrics, metric_name=dataset.metric_name, step=step)

            progress_bar.update()
            progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})

    model.cpu()
    del model

    return results


def train_full_graph_inductive(model, dataset, args, run_id):
    optimizer, gradscaler, scheduler = prepare_for_training(model=model, args=args)
    results = {f'val {dataset.metric_name}': 0, f'test {dataset.metric_name}': 0, 'step': None}
    with tqdm(total=args.max_steps, desc=f'Run {run_id}') as progress_bar:
        for step in range(1, args.max_steps + 1):
            train_step_full_graph_inductive(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                                            gradscaler=gradscaler, amp=args.amp)
            metrics = evaluate_full_graph_inductive(model=model, dataset=dataset,
                                                    best_prev_val_metric=results[f'val {dataset.metric_name}'],
                                                    amp=args.amp)
            results = update_results(results=results, new_metrics=metrics, metric_name=dataset.metric_name, step=step)

            progress_bar.update()
            progress_bar.set_postfix({f'val {dataset.metric_name}': f'{metrics[f"val {dataset.metric_name}"]:.2f}'})

    model.cpu()
    del model

    return results


def train_minibatch(model, dataset, args, run_id):
    optimizer, gradscaler, scheduler = prepare_for_training(model=model, args=args)
    return results


def get_train_fn(train_regime, transductive):
    if train_regime == 'full-graph':
        if transductive:
            return train_full_graph_transductive
        else:
            return train_full_graph_inductive

    elif train_regime == 'minibatch':
        return train_minibatch

    else:
        raise ValueError(f'Unknown train_regime: {train_regime}. Supported values are: "full-graph", "minibatch".')
