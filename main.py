import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
import random
from time import time
from sklearn.metrics import balanced_accuracy_score

def random_seed_setup(seed:int=None):
    torch.backends.cudnn.enabled = True
    if seed:
        print('Set random seed as',seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    else:
        torch.backends.cudnn.benchmark = True

def evaluate_with_rejection(config, model, data_loader, device):
    """
    Simple rejection-based evaluation using original TLC model uncertainty.
    """
    logger = config.get_logger('test')
    logger.info("--- Starting Final Evaluation with TLC Uncertainty Rejection ---")
    
    model.eval()
    all_targets = []
    all_uncertainties = []
    all_predictions = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            all_targets.append(target.cpu())
            
            # Use original TLC model
            output = model(data)
            
            # Get uncertainty from TLC model (from the last expert's weight)
            if hasattr(model.backbone, 'w') and len(model.backbone.w) > 0:
                uncertainty = model.backbone.w[-1]  # Use last expert's uncertainty
            else:
                # Fallback: calculate uncertainty from output logits
                probs = torch.softmax(output, dim=1)
                max_probs = torch.max(probs, dim=1)[0]
                uncertainty = 1 - max_probs  # Simple uncertainty measure
            
            _, predictions = torch.max(output, 1)
            
            all_uncertainties.append(uncertainty.cpu())
            all_predictions.append(predictions.cpu())

    all_targets = torch.cat(all_targets)
    all_uncertainties = torch.cat(all_uncertainties)
    all_predictions = torch.cat(all_predictions)

    # --- Generate Risk-Coverage Curve Data ---
    logger.info("Threshold | Coverage  | Balanced Error | Worst Error")
    logger.info("---------------------------------------------------------")
    results = []
    
    # Test 21 different thresholds from 0 to 1
    for threshold in np.linspace(0, 1, 21):
        is_rejected = all_uncertainties > threshold
        accepted_mask = ~is_rejected
        
        num_total = len(all_targets)
        num_rejected = torch.sum(is_rejected).item()
        num_accepted = num_total - num_rejected
        
        if num_accepted == 0:
            coverage = 0
            balanced_acc = 0 
        else:
            coverage = num_accepted / num_total
            
            accepted_preds = all_predictions[accepted_mask].numpy()
            accepted_targets = all_targets[accepted_mask].numpy()
            
            balanced_acc = balanced_accuracy_score(accepted_targets, accepted_preds)

        balanced_error = 1 - balanced_acc
        worst_error = 1 - balanced_acc
        results.append({'threshold': threshold, 'coverage': coverage, 'balanced_error': balanced_error, 'worst_error': worst_error})
        logger.info(f"{threshold:9.2f} | {coverage:9.3f} | {balanced_error:12.4f} | {worst_error:13.4f}")

    return results

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances (ORIGINAL)
    data_loader = config.init_obj('data_loader',module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console (ORIGINAL)
    model = config.init_obj('arch',module_arch)

    # get loss (ORIGINAL)
    loss_class = getattr(module_loss, config["loss"]["type"])
    criterion = config.init_obj('loss',module_loss, cls_num_list=data_loader.cls_num_list)

    # build optimizer, learning rate scheduler (ORIGINAL)
    optimizer = config.init_obj('optimizer',torch.optim,model.parameters())

    if "type" in config._config["lr_scheduler"]:
        lr_scheduler_args = config["lr_scheduler"]["args"]
        gamma = lr_scheduler_args["gamma"] if "gamma" in lr_scheduler_args else 0.1
        print("step1, step2, warmup_epoch, gamma:",(lr_scheduler_args["step1"],lr_scheduler_args["step2"],lr_scheduler_args["warmup_epoch"],gamma))

        def lr_lambda(epoch):
            if epoch >= lr_scheduler_args["step2"]:
                lr = gamma*gamma
            elif epoch >= lr_scheduler_args["step1"]:
                lr = gamma
            else:
                lr = 1
            warmup_epoch = lr_scheduler_args["warmup_epoch"]
            if epoch < warmup_epoch:
                lr = lr*float(1+epoch)/warmup_epoch
            return lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)
    else:
        lr_scheduler = None

    trainer = Trainer(
        model                                   ,
        criterion                               ,
        optimizer                               ,
        config              = config            ,
        data_loader         = data_loader       ,
        valid_data_loader   = valid_data_loader ,
        lr_scheduler        = lr_scheduler
    )
    random_seed_setup()
    trainer.train()
    
    # --- ADD REJECTION EVALUATION ---
    logger.info("--- Training finished. Starting rejection evaluation. ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluate_with_rejection(config, model, valid_data_loader, device)

if __name__=='__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c','--config',default=None,type=str,help='config file path (default: None)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs','flags type target')
    options = [
        CustomArgs(['--name'],type=str,target='name'),
        CustomArgs(['--save_period'],type=int,target='trainer;save_period'),
        CustomArgs(['--distribution_aware_diversity_factor'],type=float,target='loss;args;additional_diversity_factor'),
        CustomArgs(['--pos_weight'],type=float,target='arch;args;pos_weight'),
        CustomArgs(['--collaborative_loss'],type=int,target='loss;args;collaborative_loss'),
    ]
    config = ConfigParser.from_args(args,options)

    # Training
    start = time()
    main(config)
    end = time()

    # Show used time
    minute = (end-start)/60
    hour = minute/60
    if minute<60:
        print('Training finished in %.1f min'%minute)
    else:
        print('Training finished in %.1f h'%hour)
