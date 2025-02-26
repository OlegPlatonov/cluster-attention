import torch
from model import get_model
from logger import Logger
from utils import update_hparams, generate_grid_search_hparam_combs


def run_main_experiment(dataset, train_fn, args, logger):
    logger.start_main_experiment()
    for run_id in range(1, args.num_runs_with_best_hparams_comb + 1):
        logger.start_main_run(run_id=run_id)
        torch.manual_seed(run_id)
        model = get_model(args=args, dataset=dataset)
        run_results = train_fn(model=model, dataset=dataset, args=args, run_id=run_id)
        logger.finish_main_run(run_results=run_results)

    logger.finish_main_experiment()


def run_hparam_search_experiments(dataset, train_fn, args, logger, hparam_combs_generator):
    logger.start_search_phase()
    for hparam_comb in hparam_combs_generator:
        args = update_hparams(args=args, hparam_comb=hparam_comb)

        logger.start_search_experiment(hparams=hparam_comb)
        for run_id in range(1, args.num_runs_with_each_hparams_comb + 1):
            logger.start_search_run(run_id=run_id)
            torch.manual_seed(run_id)
            model = get_model(args=args, dataset=dataset)
            run_results = train_fn(model=model, dataset=dataset, args=args, run_id=run_id)
            logger.finish_search_run(run_results=run_results)

        logger.finish_search_experiment()

    logger.finish_search_phase()


def run_fixed_hparams_experiment(dataset, train_fn, args):
    logger = Logger(args=args, metric_name=dataset.metric_name)
    run_main_experiment(dataset=dataset, train_fn=train_fn, args=args, logger=logger)


def run_grid_search_experiments(dataset, train_fn, args):
    logger = Logger(args=args, metric_name=dataset.metric_name)

    hparam_combs_generator = generate_grid_search_hparam_combs(args)
    run_hparam_search_experiments(dataset=dataset, train_fn=train_fn, args=args, logger=logger,
                                  hparam_combs_generator=hparam_combs_generator)

    best_hparam_comb = logger.get_best_hparams()
    args = update_hparams(args=args, hparam_comb=best_hparam_comb)

    run_main_experiment(dataset=dataset, train_fn=train_fn, args=args, logger=logger)


def run_optuna_experiments(dataset, train_fn, args):
    pass


def get_experiments_fn(hparam_search_strategy):
    if hparam_search_strategy == 'fixed':
        return run_fixed_hparams_experiment
    elif hparam_search_strategy == 'grid-search':
        return run_grid_search_experiments
    elif hparam_search_strategy == 'optuna':
        return run_optuna_experiments
    else:
        raise ValueError(f'Unknown hparam_search_strategy: {hparam_search_strategy}. '
                         f'Supported values are: "fixed", "grid-search", "optuna".')
