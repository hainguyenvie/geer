import argparse
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import data_loader.data_loaders as module_data
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from time import time
from sklearn.metrics import balanced_accuracy_score
import random

# --- 1. Gated Ensemble with Evidential Rejection (GEER) Components ---

class GEERModel(nn.Module):
    """
    Gated Ensemble with Evidential Rejection Model.
    Wraps a backbone model and adds multiple expert heads. It features a sequential
    gating mechanism for efficient and dynamic inference.
    """
    def __init__(self, backbone_model, num_experts, num_classes, tau_thresholds=None):
        super().__init__()
        # The shared feature extractor
        self.backbone = backbone_model.backbone
        self.num_experts = num_experts
        self.num_classes = num_classes
        
        # Sequential gating thresholds for experts 2, 3, ...
        if tau_thresholds is None:
            # Default decreasing thresholds: later experts are harder to engage
            self.tau_thresholds = [0.8, 0.6, 0.4][:num_experts - 1]
        else:
            self.tau_thresholds = tau_thresholds
        
        # Get feature dimension from the original model's final layer
        feature_dim = backbone_model.linear.in_features
        
        # Create 'M' expert heads. Each is a linear layer.
        self.experts = nn.ModuleList([
            nn.Linear(feature_dim, num_classes) for _ in range(num_experts)
        ])
        
        print(f"Initialized GEERModel with {num_experts} experts, {num_classes} classes.")
        print(f"Inference Gating Thresholds: {self.tau_thresholds}")

    def forward(self, x, training=True):
        """
        Main forward pass. Switches between training and inference modes.
        - Training: Engages all experts to ensure full supervision.
        - Inference: Uses the sequential gating mechanism for efficiency.
        """
        features = self.backbone(x)
        if training:
            return self._forward_all_experts(features)
        else:
            return self._forward_sequential_gating(features)
    
    def _forward_all_experts(self, features):
        """Forward pass for training. All experts are used."""
        evidences = [F.softplus(expert(features)) for expert in self.experts]
        return evidences
    
    def _forward_sequential_gating(self, features):
        """Forward pass for inference with the sequential gating logic."""
        # The first expert is always engaged
        evidence = F.softplus(self.experts[0](features))
        alpha_joint = evidence + 1
        
        # Sequentially decide to engage subsequent experts
        for m in range(1, self.num_experts):
            S_joint = torch.sum(alpha_joint, dim=1)
            u_joint = self.num_classes / S_joint
            
            # Gating decision: engage expert 'm' if joint uncertainty is high
            engage_mask = u_joint > self.tau_thresholds[m-1]
            
            if torch.any(engage_mask):
                # Only compute for samples that need this expert
                evidence_m = F.softplus(self.experts[m](features[engage_mask]))
                alpha_m = evidence_m + 1
                
                # Fuse the new evidence into the joint alpha for the engaged samples
                alpha_joint[engage_mask] = self.fuse_evidences_dst(
                    alpha_joint[engage_mask], alpha_m
                )
        
        return alpha_joint

    def fuse_evidences_dst(self, alpha1, alpha2):
        """
        Fuses two sets of Dirichlet parameters using the Dempster-Shafer combination rule.
        This is theoretically more robust for combining expert opinions.
        """
        S1 = torch.sum(alpha1, dim=1, keepdim=True)
        S2 = torch.sum(alpha2, dim=1, keepdim=True)
        
        b1 = (alpha1 - 1) / S1
        b2 = (alpha2 - 1) / S2
        
        b_fused_num = b1 * S2 + b2 * S1
        S_fused = S1 + S2 - torch.sum(b1 * S2, dim=1, keepdim=True) # Approximation of Dempster's rule
        
        b_fused = b_fused_num / (S_fused + 1e-8)
        alpha_fused = b_fused * S_fused + 1
        return alpha_fused


class BEL2RLoss(nn.Module):
    """
    Balanced Evidential Learning to Reject Loss (BEL2R).
    This loss function has three main components:
    1. L_B-EDL: A balanced version of the Evidential Deep Learning loss to handle long-tail data.
    2. L_rej: An uncertainty-based regularizer that encourages the model to be uncertain on hard examples.
    3. L_div: A diversity regularizer that encourages experts to learn different features.
    """
    def __init__(self, cls_num_list, lambda_rej=0.2, lambda_div=0.3, rejection_cost=0.5):
        super().__init__()
        self.lambda_rej = lambda_rej
        self.lambda_div = lambda_div
        self.rejection_cost = rejection_cost
        
        # Calculate class weights for balanced learning
        cls_num_list = np.array(cls_num_list)
        class_weights = 1.0 / cls_num_list
        self.class_weights = torch.tensor(class_weights / np.sum(class_weights) * len(cls_num_list), dtype=torch.float32)
        
    def to(self, device):
        self.class_weights = self.class_weights.to(device)
        return self
    
    def forward(self, evidences, targets, epoch, total_epochs):
        num_classes = evidences[0].shape[1]
        y_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        class_weights_batch = self.class_weights[targets]
        
        total_edl_loss = 0
        total_uncertainty = 0
        
        for evidence in evidences:
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1)
            
            # Component 1.1: Balanced Type-II Maximum Likelihood Loss
            log_likelihood = torch.lgamma(S) - torch.lgamma(torch.tensor(num_classes)) - torch.sum(torch.lgamma(alpha), dim=1) + torch.sum((alpha - 1) * y_one_hot, dim=1)
            loss_ml = torch.mean(class_weights_batch * -log_likelihood)
            
            # Component 1.2: Balanced KL Divergence to regularize incorrect evidence
            annealing_factor = min(1.0, epoch / total_epochs)
            kl_div_term = torch.sum((alpha - 1) * (1 - y_one_hot), dim=1)
            loss_kl = torch.mean(class_weights_batch * kl_div_term)
            
            total_edl_loss += loss_ml + annealing_factor * loss_kl
            total_uncertainty += num_classes / S

        loss_b_edl = total_edl_loss / len(evidences)
        
        # Component 2: Rejection Cost Regularizer (L_rej)
        mean_uncertainty = total_uncertainty / len(evidences)
        loss_rej = self.rejection_cost * torch.mean(mean_uncertainty)
        
        # Component 3: Expert Diversity Loss (L_div)
        loss_div = 0
        if len(evidences) > 1:
            probs = [F.softmax(e, dim=1) for e in evidences]
            mean_prob = torch.mean(torch.stack(probs), dim=0)
            for prob in probs:
                loss_div += F.kl_div(torch.log(prob + 1e-8), mean_prob, reduction='batchmean')
            loss_div = -loss_div / len(evidences) # We want to MAXIMIZE divergence, so we minimize its negative
        
        total_loss = loss_b_edl + self.lambda_rej * loss_rej + self.lambda_div * loss_div
        
        return total_loss, {
            'loss_b_edl': loss_b_edl.item(),
            'loss_rej': loss_rej.item(), 
            'loss_div': loss_div.item() if isinstance(loss_div, torch.Tensor) else loss_div,
            'mean_uncertainty': torch.mean(mean_uncertainty).item()
        }


class GatedTrainer(Trainer):
    """
    Custom Trainer to handle the GEERModel and BEL2RLoss.
    """
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loss_components = collections.defaultdict(float)
        
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            
            # Use training forward pass to get all expert evidences
            evidences = self.model(data, training=True)
            
            loss, components = self.criterion(evidences, target, epoch, self.epochs)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected at epoch {epoch}, batch {batch_idx}. Skipping update.")
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            for key, value in components.items():
                loss_components[key] += value

            if batch_idx % 10 == 0:
                self.logger.info(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.data_loader.dataset)}] '
                    f'Loss: {loss.item():.4f} (EDL: {components["loss_b_edl"]:.2f}, '
                    f'Rej: {components["loss_rej"]:.2f}, Div: {components["loss_div"]:.2f})'
                )

        log = {'loss': total_loss / len(self.data_loader)}
        for key in loss_components:
            log[key] = loss_components[key] / len(self.data_loader)
        
        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return log
        
    def _valid_epoch(self, epoch):
        """Validation for checkpointing purposes. Uses a simple fusion of all experts."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, target in self.valid_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Use all experts for validation to get a robust accuracy measure
                evidences = self.model(data, training=True)
                
                # Fuse evidences by summing (a simple but effective fusion)
                alpha_final = evidences[0] + 1
                for i in range(1, len(evidences)):
                    alpha_final += evidences[i] 
                
                _, pred = torch.max(alpha_final, 1)
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        balanced_acc = balanced_accuracy_score(all_targets, all_preds)
        self.logger.info(f"Validation Epoch: {epoch}, Balanced Accuracy: {balanced_acc:.4f}")

        return {'balanced_accuracy': balanced_acc}


def evaluate_with_rejection(config, model, data_loader, device):
    """
    Final evaluation function. This is the crucial part that tests the rejection mechanism.
    It simulates the risk-coverage trade-off by varying the rejection threshold.
    """
    logger = config.get_logger('test')
    logger.info("--- Starting Final Evaluation with Gated Rejection Mechanism ---")
    
    model.eval()
    all_targets = []
    all_final_uncertainties = []
    all_final_predictions = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            all_targets.append(target.cpu())
            
            # CRITICAL: Use the model's sequential gating inference path
            alpha_joint = model(data, training=False)
            
            # Calculate final uncertainty from the joint (fused) alpha
            final_uncertainty = model.num_classes / torch.sum(alpha_joint, dim=1)
            
            # Final prediction is based on the highest evidence in the joint alpha
            _, final_preds = torch.max(alpha_joint, 1)
            
            all_final_uncertainties.append(final_uncertainty.cpu())
            all_final_predictions.append(final_preds.cpu())

    all_targets = torch.cat(all_targets)
    all_final_uncertainties = torch.cat(all_final_uncertainties)
    all_final_predictions = torch.cat(all_final_predictions)

    # --- Generate Risk-Coverage Curve Data ---
    logger.info("\n--- Risk-Coverage Curve Results ---")
    logger.info("Rej. Thr | Coverage  | Balanced Acc | Risk (Error)")
    logger.info("----------------------------------------------------")
    
    # Test multiple rejection thresholds based on uncertainty percentiles
    for percentile in np.linspace(0, 100, 21):
        if percentile == 100:
            threshold = all_final_uncertainties.max() + 1
        else:
            threshold = np.percentile(all_final_uncertainties.numpy(), 100 - percentile)
        
        # Accept if uncertainty is LOWER than the threshold
        accepted_mask = all_final_uncertainties <= threshold
        
        num_total = len(all_targets)
        num_accepted = torch.sum(accepted_mask).item()
        
        if num_accepted == 0:
            coverage = 0.0
            balanced_acc = 0.0
        else:
            coverage = num_accepted / num_total
            accepted_preds = all_final_predictions[accepted_mask].numpy()
            accepted_targets = all_targets[accepted_mask].numpy()
            balanced_acc = balanced_accuracy_score(accepted_targets, accepted_preds)

        risk = 1 - balanced_acc
        logger.info(f"{threshold:10.4f} | {coverage:9.3f} | {balanced_acc:12.4f} | {risk:12.4f}")


def main(config):
    logger = config.get_logger('train')

    # Setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    num_classes = len(data_loader.cls_num_list)
    
    # Build the backbone model from config
    backbone_model = config.init_obj('arch', module_arch)
    
    # Wrap the backbone in our GEERModel
    num_experts = config['arch']['args'].get('num_experts', 3)
    model = GEERModel(backbone_model, num_experts=num_experts, num_classes=num_classes)
    logger.info(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Instantiate our custom loss function
    loss_args = config.config.get('loss', {}).get('args', {})
    criterion = BEL2RLoss(
        cls_num_list=data_loader.cls_num_list,
        lambda_rej=loss_args.get('lambda_rej', 0.2),
        lambda_div=loss_args.get('lambda_div', 0.3),
        rejection_cost=loss_args.get('rejection_cost', 0.5)
    ).to(device)

    # Build optimizer and learning rate scheduler
    optimizer = config.init_obj('optimizer', torch.optim, model.parameters())
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    # Use our custom GatedTrainer
    trainer = GatedTrainer(
        model, criterion, optimizer, config, data_loader,
        valid_data_loader=valid_data_loader, lr_scheduler=lr_scheduler
    )
    
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    trainer.train()

    # --- Final Evaluation after Training ---
    logger.info("--- Training finished. Starting final evaluation. ---")
    best_model_path = str(config.save_dir / 'model_best.pth')
    
    # Load the best model for final evaluation
    try:
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded best model from: {best_model_path}")
    except FileNotFoundError:
        logger.warning("Could not find best model checkpoint. Evaluating with the last model state.")
    
    evaluate_with_rejection(config, model, valid_data_loader, device)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Gated Rejection Learning')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--num_experts'], type=int, target='arch;args;num_experts'),
    ]
    config = ConfigParser.from_args(args, options)
    
    start = time()
    main(config)
    end = time()
    minute = (end - start) / 60
    hour = minute / 60
    print(f'Total time: {hour:.2f} hours' if hour > 1 else f'Total time: {minute:.2f} minutes')
