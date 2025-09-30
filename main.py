import argparse
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss # Used as a reference, we define our own
import model.model as module_arch # The backbone model architecture
from parse_config import ConfigParser
# The original Trainer class is replaced by our custom GatedTrainer
from trainer import Trainer
import random
from time import time
from sklearn.metrics import balanced_accuracy_score

# --- 1. New Components for Gated Ensemble Rejection Method ---

# --- 1a. Basic ResNet Components ---
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = lambda x: x
        if stride != 1 or in_planes != planes:
            self.planes = planes
            self.in_planes = in_planes
            self.shortcut = lambda x: F.pad(x[:,:,::2,::2],(0,0,0,0,(planes-in_planes)//2,(planes-in_planes)//2),"constant",0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        out = F.relu(out)
        return out

# --- 1b. Gated Ensemble Model ---
# This class wraps a backbone model and adds multiple "expert" heads.
# Each head is trained to output "evidence" for a Dirichlet distribution,
# enabling us to quantify uncertainty more reliably than with softmax.
class GatedEnsembleModel(nn.Module):
    def __init__(self, backbone_model, num_experts, num_classes):
        super().__init__()
        # Use the original TLC backbone but modify it for our gated approach
        self.backbone = backbone_model.backbone  # Use the original ResNet_s
        self.num_experts = num_experts
        self.num_classes = num_classes
        
        # Override the original linears with our evidence-based experts
        self.experts = nn.ModuleList([
            nn.Linear(64, num_classes) for _ in range(num_experts)
        ])
        
        # Store original linears for reference but don't use them
        self.original_linears = self.backbone.linears
        
        print(f"Initialized GatedEnsembleModel with {num_experts} experts and {num_classes} classes.")

    def forward(self, x):
        """
        Forward pass using the original TLC backbone but with our evidence-based experts.
        """
        # Use the original TLC forward pass but replace the final linear layers
        x = F.relu(self.backbone.bn1(self.backbone.conv1(x)))
        
        # Get features from each expert branch (like original TLC)
        evidences = []
        for i in range(self.num_experts):
            xi = self.backbone.layer1s[i](x)
            xi = self.backbone.layer2s[i](xi)
            xi = self.backbone.layer3s[i](xi)
            xi = F.avg_pool2d(xi, xi.shape[3])
            xi = xi.flatten(1)
            
            # Use our evidence-based expert instead of original linear
            evidence = F.softplus(self.experts[i](xi))
            evidences.append(evidence)
        
        return evidences

    def _hook_before_iter(self):
        """Hook method required by original TLC training process."""
        # Call the backbone's hook method if it exists
        if hasattr(self.backbone, '_hook_before_iter'):
            self.backbone._hook_before_iter()

    def fuse_evidences_dst(self, alpha1, alpha2):
        """
        Fuses two sets of Dirichlet parameters (alpha) using the Dempster-Shafer combination rule.
        This allows us to aggregate knowledge from multiple experts.
        """
        S1 = torch.sum(alpha1, dim=1, keepdim=True)
        S2 = torch.sum(alpha2, dim=1, keepdim=True)
        
        b1 = (alpha1 - 1) / S1
        b2 = (alpha2 - 1) / S2
        
        # The combination rule for belief masses
        b_fused = 0.5 * (b1 * S2 + b2 * S1) / (S1 + S2)
        
        # Convert fused belief back to alpha
        alpha_fused = b_fused * (S1 + S2) + 1
        return alpha_fused

# --- 1b. New Loss Function (inspired by TLC paper) ---
def TLC_Loss(evidences, y, current_epoch, total_epochs, lambda_kl=0.1, lambda_div=1.0):
    """
    Restored TLC loss function with proper evidence-based learning.
    """
    y_one_hot = F.one_hot(y, num_classes=evidences[0].shape[1]).float()
    total_loss = 0
    
    # Annealing factor for the KL divergence term
    annealing_factor = min(1.0, current_epoch / (total_epochs / 2)) * lambda_kl

    for e in evidences:
        alpha = e + 1
        S = torch.sum(alpha, dim=1, keepdim=True)

        # Type II Maximum Likelihood Loss (Eq. 9 in TLC)
        log_likelihood = torch.sum(y_one_hot * (torch.log(S) - torch.log(alpha)), dim=1)
        loss_ml = torch.mean(log_likelihood)

        # KL Divergence Loss for uncertainty calibration (Eq. 10 in TLC)
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        kl_div = torch.sum(
            (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1
        )
        loss_kl = torch.mean(kl_div)

        total_loss += (loss_ml + annealing_factor * loss_kl)

    # Diversity Loss (Eq. 12 in TLC) - encourages experts to learn different features
    alphas = [e + 1 for e in evidences]
    mean_alpha = torch.mean(torch.stack(alphas), dim=0)
    
    loss_div = 0
    if len(alphas) > 1:
        for alpha in alphas:
            loss_div += torch.mean(
                torch.sum(F.kl_div(torch.log(alpha), mean_alpha, reduction='none'), dim=1)
            )
        total_loss += lambda_div * loss_div

    return total_loss / len(evidences)

# --- 1c. New Trainer with Gating Logic ---
# This custom trainer class overrides the validation method to test our rejection mechanism.
class GatedTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        # Don't call super().__init__ because it tries to move criterion to device
        # Instead, initialize manually
        self.device = torch.device('cuda:0')
        self.config = config
        self.model = model.to(self.device)
        self.criterion = criterion  # Keep as function, don't move to device
        self.opt = optimizer
        self.epochs = config['trainer']['epochs']
        
        # Initialize the rest like the original Trainer
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader)
        self.valid_data_loader = valid_data_loader
        if valid_data_loader is not None:
            self.val_targets = torch.tensor(valid_data_loader.dataset.targets,device=self.device).long()
            self.num_class = self.val_targets.max().item()+1
        else:
            self.val_targets = None
            self.num_class = None
        self.lr_scheduler = lr_scheduler
        
        # Add our custom attributes
        self.num_classes = self.model.num_classes
        self.total_epochs = self.epochs
        self.log_step = 10  # Log every 10 batches
        self.logger = config.get_logger('train')

    def train(self):
        """Training loop"""
        for epoch in range(1, self.epochs + 1):
            result = self._train_epoch(epoch)

    def _train_epoch(self, epoch):
        """
        Custom training loop that works with the original TLC loss function.
        """
        self.model.train()
        self.model._hook_before_iter()
        self.criterion._hook_before_epoch(epoch)
        
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.opt.zero_grad()
            evidences = self.model(data)
            
            # Convert evidences to the format expected by original TLC loss
            # The original TLC expects a single output, but we have evidences
            # We'll use the first expert's output for the main loss
            main_output = evidences[0]  # Use first expert as main output
            
            # Create extra_info for the original TLC loss
            extra_info = {
                "num_expert": len(evidences),
                "logits": evidences,
                'w': [torch.ones(len(data), device=data.device) for _ in range(len(evidences))]  # Dummy weights
            }
            
            # Use original TLC loss
            loss = self.criterion(x=main_output, y=target, epoch=epoch, extra_info=extra_info)
            
            loss.backward()
            self.opt.step()

            total_loss += loss.item()

            if batch_idx % self.log_step == 0:
                self.logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.data_loader.dataset)} '
                                 f'({100.0 * batch_idx / len(self.data_loader):.0f}%)] Loss: {loss.item():.6f}')
        
        log = {'loss': total_loss / len(self.data_loader)}
        
        # Call validation epoch
        val_log = self._valid_epoch(epoch)
        log.update(val_log)
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return log
        
    def _valid_epoch(self, epoch):
        """
        The validation loop is kept simple here. The detailed evaluation with rejection
        is performed *after* training is complete in the `evaluate_with_rejection` function.
        """
        if self.valid_data_loader is None:
            return {'val_loss': 0, 'val_balanced_accuracy': 0}
            
        self.model.eval()
        total_val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                evidences = self.model(data)
                
                # Use first expert for validation loss calculation
                main_output = evidences[0]
                extra_info = {
                    "num_expert": len(evidences),
                    "logits": evidences,
                    'w': [torch.ones(len(data), device=data.device) for _ in range(len(evidences))]
                }
                loss = self.criterion(x=main_output, y=target, epoch=epoch, extra_info=extra_info)
                total_val_loss += loss.item()

                # Use the first expert's prediction for standard validation accuracy
                _, pred = torch.max(evidences[0], 1)
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        # Calculate classification accuracies like the original TLC
        accuracy = (all_preds == all_targets).float().mean()
        balanced_acc = balanced_accuracy_score(all_targets.numpy(), all_preds.numpy())
        
        # Calculate region-based accuracies (head, medium, tail)
        num_classes = self.num_class
        region_len = num_classes / 3
        
        # Calculate region accuracies
        region_correct = (all_preds / region_len).long() == (all_targets / region_len).long()
        region_acc = region_correct.float().mean()
        
        # Calculate head, medium, tail accuracies
        split_acc = [0, 0, 0]
        region_idx = (torch.arange(num_classes) / region_len).long()
        region_vol = [
            num_classes - torch.count_nonzero(region_idx).item(),
            torch.where(region_idx == 1, True, False).sum().item(),
            torch.where(region_idx == 2, True, False).sum().item()
        ]
        target_count = all_targets.bincount().cpu().numpy()
        region_vol = [
            target_count[:region_vol[0]].sum(), 
            target_count[region_vol[0]:(region_vol[0] + region_vol[1])].sum(),
            target_count[-region_vol[2]:].sum()
        ]
        
        for i in range(len(all_targets)):
            split_acc[region_idx[all_targets[i].item()]] += (all_preds[i] == all_targets[i]).item()
        split_acc = [split_acc[i] / region_vol[i] if region_vol[i] > 0 else 0 for i in range(3)]
        
        # Print classification accuracies like the original TLC
        print(f'================ Epoch: {epoch:03d} ================')
        print('Classification ACC:')
        print(f'\t all \t = {accuracy:.4f}')
        print(f'\t region  = {region_acc:.4f}')
        print(f'\t head \t = {split_acc[0]:.4f}')
        print(f'\t med \t = {split_acc[1]:.4f}')
        print(f'\t tail \t = {split_acc[2]:.4f}')
        
        self.logger.info(f"Validation Epoch: {epoch}, Accuracy: {accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}")

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_balanced_accuracy': balanced_acc
        }

# --- 2. Post-Training Evaluation Function ---
def evaluate_with_rejection(config, model, data_loader, device):
    """
    This is the core evaluation function. After training, it runs the model
    on the test set and calculates the trade-off between balanced error (risk)
    and coverage for various rejection thresholds.
    """
    logger = config.get_logger('test')
    logger.info("--- Starting Final Evaluation with Gated Rejection ---")
    
    model.eval()
    all_targets = []
    all_final_uncertainties = []
    all_final_predictions = []
    
    num_classes = model.num_classes
    num_experts = model.num_experts

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            all_targets.append(target.cpu())
            
            evidences = model(data)
            
            # Fuse all experts to get final uncertainty and prediction
            current_alpha = evidences[0] + 1
            if num_experts > 1:
                for i in range(1, num_experts):
                    next_alpha = evidences[i] + 1
                    current_alpha = model.fuse_evidences_dst(current_alpha, next_alpha)

            final_uncertainty = num_classes / torch.sum(current_alpha, dim=1)
            _, final_preds = torch.max(current_alpha, 1)
            
            all_final_uncertainties.append(final_uncertainty.cpu())
            all_final_predictions.append(final_preds.cpu())

    all_targets = torch.cat(all_targets)
    all_final_uncertainties = torch.cat(all_final_uncertainties)
    all_final_predictions = torch.cat(all_final_predictions)

    # --- Generate Risk-Coverage Curve Data ---
    logger.info("Threshold | Coverage  | Balanced Error | Worst Error")
    logger.info("---------------------------------------------------------")
    results = []
    # Test 21 different thresholds from 0 to 1
    for threshold in np.linspace(0, 1, 21):
        is_rejected = all_final_uncertainties > threshold
        accepted_mask = ~is_rejected
        
        num_total = len(all_targets)
        num_rejected = torch.sum(is_rejected).item()
        num_accepted = num_total - num_rejected
        
        if num_accepted == 0:
            coverage = 0
            balanced_acc = 0 
        else:
            coverage = num_accepted / num_total
            
            accepted_preds = all_final_predictions[accepted_mask].numpy()
            accepted_targets = all_targets[accepted_mask].numpy()
            
            balanced_acc = balanced_accuracy_score(accepted_targets, accepted_preds)

        balanced_error = 1 - balanced_acc
        worst_error = 1 - balanced_acc  # For now, using same as balanced error
        results.append({'threshold': threshold, 'coverage': coverage, 'balanced_error': balanced_error, 'worst_error': worst_error})
        logger.info(f"{threshold:9.2f} | {coverage:9.3f} | {balanced_error:12.4f} | {worst_error:13.4f}")

    return results


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

# --- 3. Modified `main` function to use our new components ---
def main(config):
    logger = config.get_logger('train')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    num_classes = len(data_loader.cls_num_list)
    
    # Build a standard backbone model from the config
    backbone_model = config.init_obj('arch', module_arch)
    
    # --- OUR CHANGE: Wrap it in our GatedEnsembleModel ---
    num_experts = config['arch']['args'].get('num_experts', 3) # Add this to your config
    model = GatedEnsembleModel(backbone_model, num_experts=num_experts, num_classes=num_classes)
    logger.info(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Use the original TLC loss function
    loss_class = getattr(module_loss, config["loss"]["type"])
    criterion = config.init_obj('loss', module_loss, cls_num_list=data_loader.cls_num_list)
    criterion = criterion.to(device)  # Move loss function to the same device as model

    # Build optimizer and learning rate scheduler (same as original)
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = None
    if "type" in config.config.get("lr_scheduler", {}):
        lr_scheduler_args = config["lr_scheduler"]["args"]
        gamma = lr_scheduler_args.get("gamma", 0.1)
        def lr_lambda(epoch):
            if epoch >= lr_scheduler_args["step2"]: lr = gamma*gamma
            elif epoch >= lr_scheduler_args["step1"]: lr = gamma
            else: lr = 1
            warmup_epoch = lr_scheduler_args.get("warmup_epoch", 0)
            if epoch < warmup_epoch: lr = lr * float(1 + epoch) / warmup_epoch
            return lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- OUR CHANGE: Use our GatedTrainer ---
    trainer = GatedTrainer(
        model, criterion, optimizer, config, data_loader,
        valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler
    )
    
    random_seed_setup(None)  # Use default random seed behavior
    trainer.train()

    # --- Final Evaluation after Training ---
    logger.info("--- Training finished. Starting final evaluation. ---")
    # The model is already trained and in memory, no need to load from checkpoint
    
    # Evaluate on the validation/test set to generate the risk-coverage curve
    evaluate_with_rejection(config, model, valid_data_loader, device)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Gated Rejection Learning')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    # Custom CLI options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        # You can add new arguments here to control your model from the command line
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
    ]
    config = ConfigParser.from_args(args, options)

    # Training
    start = time()
    main(config)
    end = time()

    # Show total time used
    minute = (end - start) / 60
    hour = minute / 60
    print(f'Total time: {hour:.2f} hours' if hour > 1 else f'Total time: {minute:.2f} minutes')
