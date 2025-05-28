import os
import torch
import warnings
import pandas as pd
import torch.distributed as dist
from models import get_model
from datasets import AIECPyramidDataset, experts_train_transforms, experts_test_transforms
from .metrics import compute_cls_metrics, compute_surv_metrics
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, StratifiedKFold
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.nn.parallel import DistributedDataParallel as DDP
from .losses import CrossEntropySurvLoss, NLLSurvLoss, CoxSurvLoss, CrossEntropyClsLoss, CrossViewConsistency, MultiviewCrossEntropyClsLoss


class MetricLogger:
    def __init__(self, n_folds):
        self.fold = 0
        self.n_folds = n_folds
        self.fold_metrics = [{} for _ in range(n_folds)] # save final metrics for each fold
    
    @property
    def metrics(self):
        return list(self.fold_metrics[self.fold].keys())
    
    def _set_fold(self, fold):
        self.fold = fold
    
    def _empty_dict(self):
        return {key: 0.0 for key in self.metrics}

    def update(self, metric_dict):
        for key in metric_dict:
            self.fold_metrics[self.fold][key] = metric_dict[key]
    
    def _fold_average(self):
        if self.fold < self.n_folds - 1:
            raise Warning("Not all folds have been completed.")
        avg_metrics = self._empty_dict()
        for metric in avg_metrics:
            for fold in self.fold_metrics:
                avg_metrics[metric] += fold[metric]
            avg_metrics[metric] /= self.n_folds
        
        return avg_metrics


class Trainer:
    def __init__(self, args, wb_logger=None, verbose=True, val_steps=50):
        self.verbose = verbose
        self.val_steps = val_steps
        self.wsi_df = pd.read_excel(args.wsi_csv_path)
        self.args = args
        self.wb_logger = wb_logger
        self.m_logger = MetricLogger(n_folds=args.kfold)
        self.surv2lossfunc = {'nll': NLLSurvLoss, 'cox': CoxSurvLoss, 'ce': CrossEntropySurvLoss}
    
    def _dataset_split(self, train_csv, test_csv, args):

        train_transforms = experts_train_transforms(n_experts=args.n_views, num_levels=args.num_levels, 
                                                    downsample_factor=args.downsample_factor, lowest_level=args.lowest_level, 
                                                    dropout=args.tree_dropout, visible_levels=args.visible_levels, 
                                                    fix_agent=args.fix_agent, random_layer=args.random_layer)
        test_transforms = experts_test_transforms(n_experts=args.n_views, num_levels=args.num_levels, 
                                                  downsample_factor=args.downsample_factor, lowest_level=args.lowest_level, 
                                                  visible_levels=args.visible_levels)

        self.train_dataset = AIECPyramidDataset(args.data_root, train_csv, task=args.task, transforms=train_transforms)
        if args.world_size > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
            )
        else:
            train_sampler = None

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                       drop_last=True, num_workers=args.workers, sampler=train_sampler, pin_memory=True,
                                       collate_fn=AIECPyramidDataset.collate_fn)
        
        if args.rank == 0:
            self.test_dataset = AIECPyramidDataset(args.data_root, test_csv, task=args.task, transforms=test_transforms)

            self.test_loader = DataLoader(self.test_dataset, batch_size=args.batch_size, shuffle=False,
                                        drop_last=False, num_workers=args.workers, pin_memory=True,
                                        collate_fn=AIECPyramidDataset.collate_fn)
            
            print(f"Train dataset size: {len(self.train_dataset)}, Test dataset size: {len(self.test_dataset)}")
        else:
            self.test_loader = None
            self.test_dataset = None
        
        args.n_classes = self.train_dataset.n_classes
        step_per_epoch = len(self.train_dataset) // (args.batch_size * args.world_size)

        self.model = get_model(args).cuda()
        self.optimizer = getattr(torch.optim, args.optimizer)(self.model.parameters(), lr=args.lr,
                                                              weight_decay=args.weight_decay, betas=(0.9, 0.99))
        if self.args.task == 'grading' or self.args.task == 'subtyping':
            self.criterion = CrossEntropyClsLoss().cuda()
        else:
            self.criterion = self.surv2lossfunc[self.args.surv_loss.lower()]().cuda()
        
        self.xview_criterion = CrossViewConsistency(args.batch_size, args.world_size).cuda()
        self.multiview_criterion = MultiviewCrossEntropyClsLoss().cuda()
        
        if args.scheduler:
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, args.warmup_epochs * step_per_epoch, 
                                                             args.epochs * step_per_epoch)
        else:
            self.scheduler = None
        
        if args.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[args.rank], static_graph=True)


    def run(self, args):
        train_csv = pd.read_excel(args.train_csv_path)
        test_csv = pd.read_excel(args.test_csv_path)
        self._dataset_split(train_csv, test_csv, args)
        self.fold = 0
        self.train(args)
        if args.rank == 0:
            metric_dict = self.validate(args)
            print('-'*20, 'Metrics', '-'*20)
            print(metric_dict)


    def kfold_train(self, args):
        patient_df = self.wsi_df.groupby('Case.ID').first().reset_index()
        if args.task == 'grading':
            patient_df = patient_df.dropna(subset=['Tumor.Grading'])
            patient_label_list = patient_df['Tumor.Grading'].values
        elif args.task == 'subtyping':
            patient_df = patient_df.dropna(subset=['Tumor.MolecularSubtype'])
            patient_label_list = patient_df['Tumor.MolecularSubtype'].values
        elif args.task == 'survival':
            patient_df = patient_df.dropna(subset=['Overall.Survival.Interval'])
            patient_label_list = patient_df['Overall.Survival.Interval'].values
        else:
            raise ValueError(f"Unknown task: {args.task}")
        patient_list = patient_df['Case.ID'].values
        kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        
        for fold, (train_idx, test_idx) in enumerate(kfold.split(patient_list, patient_label_list)):
            if self.args.rank == 0:
                print('-'*20, f'Fold {fold}', '-'*20)
            train_pid = patient_list[train_idx]
            test_pid = patient_list[test_idx]
            train_csv = self.wsi_df[self.wsi_df['Case.ID'].isin(train_pid)]
            test_csv = self.wsi_df[self.wsi_df['Case.ID'].isin(test_pid)]
            self._dataset_split(train_csv, test_csv, args)

            self.m_logger._set_fold(fold)
            self.fold = fold

            self.train(args)
            # validate for the fold
            
            if args.rank == 0 and self.verbose:
                metric_dict = self.validate(args)
                self.m_logger.update(metric_dict)
                print('-'*20, f'Fold {fold} Metrics', '-'*20)
                print(metric_dict)

            # do univariate cox regression analysis
            # if 'surv' in self.task:
            #     self.fold_univariate_cox_regression_analysis(fold)
        
        avg_metrics = self.m_logger._fold_average()
        print('-'*20, 'Average Metrics', '-'*20)
        print(avg_metrics)
        # self._save_fold_avg_results(avg_metrics)
        # self.save_model()

    def train(self, args):
        torch.cuda.empty_cache()
        self.model.train()
        cur_iters = 0
        for i in range(self.args.epochs):
            for data in self.train_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}

                outputs = self.model(data['features'])
                xview_loss = self.xview_criterion(outputs['agents'], data['label'])
                cls_loss = self.criterion(outputs, data)
                loss = cls_loss + args.lambda_xview * xview_loss

                self.optimizer.zero_grad()
                loss.backward()

                # if dist.is_available() and dist.is_initialized():
                #     for name, p in self.model.named_parameters():
                #         if p.grad is not None:
                #             dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                #             p.grad.data /= dist.get_world_size()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                cur_iters += 1
                if self.verbose and args.rank == 0:
                    if cur_iters % self.val_steps == 0:
                        cur_lr = self.optimizer.param_groups[0]['lr']
                        metric_dict = self.validate(args)
                        print(f"Fold {self.fold} | Epoch {i} | Loss: {loss.item()} | Acc: {metric_dict['Accuracy']} | LR: {cur_lr}")
                        if self.wb_logger is not None:
                            self.wb_logger.log({f"Fold_{self.fold}": {
                                'Train': {'loss': loss.item(), 
                                          'cls_loss': cls_loss.item(),
                                          'xview_loss': xview_loss.item(),
                                          'lr': cur_lr},
                                'Test': metric_dict
                            }})

    def validate(self, args):
        training = self.model.training
        self.model.eval()

        if args.task == 'grading' or args.task == 'subtyping':
            ground_truth = torch.Tensor().cuda()
            probabilities = torch.Tensor().cuda()
            loss = 0.0
        elif args.task == 'survival':
            event_indicator = torch.Tensor().cuda() # whether the event (death) has occurred
            event_time = torch.Tensor().cuda()
            estimate = torch.Tensor().cuda()        

        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model(data['features'])

                if args.task == 'grading' or args.task == 'subtyping':
                    loss += self.criterion(outputs, data).item()
                    prob = outputs.y_prob
                    ground_truth = torch.cat((ground_truth, data['label']), dim=0)
                    probabilities = torch.cat((probabilities, prob), dim=0)
                
                elif args.task == 'survival':
                    risk = -torch.sum(outputs['surv'], dim=1)
                    event_indicator = torch.cat((event_indicator, data['dead']), dim=0)
                    event_time = torch.cat((event_time, data['event_time']), dim=0)
                    estimate = torch.cat((estimate, risk), dim=0)

            if args.task == 'grading' or args.task == 'subtyping':
                metric_dict = compute_cls_metrics(ground_truth, probabilities)
                metric_dict['Loss'] = loss / len(self.test_loader)
            elif args.task == 'survival':
                metric_dict = compute_surv_metrics(event_indicator, event_time, estimate)
        
        self.model.train(training)

        return metric_dict
    
    def save_model(self):
        model_name = f"{self.args.backbone}_{self.args.extractor}.pt"
        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints, exist_ok=True)
        save_path = os.path.join(self.args.checkpoints, model_name)
        torch.save(self.model.state_dict(), save_path)

    def _save_fold_avg_results(self, metric_dict, keep_best=True):
        # keep_best: whether save the best model (highest mcc) for each fold
        task2name = {'2_cls': 'Binary', '4_cls': '4Class', 'survival': 'Survival'}
        taskname = task2name[self.args.task]

        df_name = f"{self.args.kfold}Fold_{taskname}.xlsx"
        res_path = self.args.results
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        dataset_settings = ['Model', 'KFold', 'Feature Extractor', 'Magnification', 'Patch Size', 
                    'Patch Overlap', 'New Annotation', 'Stain Normalization', 'Augmentation', 'Epochs']
        dataset_kwargs = ['backbone', 'kfold', 'extractor', 'magnification', 'patch_size', 
                          'patch_overlap', 'calibrate', 'stain_norm', 'augmentation', 'epochs']
        task_settings = ['Metric Average Method'] if 'cls'in self.task else ['Survival Loss']
        task_kwargs = ['metric_avg'] if 'cls' in self.task else ['surv_loss']
        
        settings = dataset_settings + task_settings
        set2kwargs = {k: v for k, v in zip(settings, dataset_kwargs + task_kwargs)}

        metric_names = self.m_logger.metrics
        df_columns = settings + metric_names
        
        df_path = os.path.join(res_path, df_name)
        if not os.path.exists(df_path):
            df = pd.DataFrame(columns=df_columns)
        else:
            df = pd.read_excel(df_path)
            if df_columns != df.columns.tolist():
                warnings.warn("Columns in the existing excel file do not match the current settings.")
                df = pd.DataFrame(columns=df_columns)
        
        new_row = {k: self.args.__dict__[v] for k, v in set2kwargs.items()}
        # fine-grained modification for better presentation
        new_row['Feature Extractor'] = new_row['Feature Extractor'].upper()
        new_row['Magnification'] = f"{new_row['Magnification']}\u00D7"
        new_row['New Annotation'] = 'Yes' if new_row['New Annotation'] else 'No'
        new_row['Stain Normalization'] = 'Yes' if new_row['Stain Normalization'] else 'No'
        new_row['Augmentation'] = 'Yes' if new_row['Augmentation'] else 'No'

        if keep_best: # keep the rows with the best mcc for each fold
            reference = 'MCC' if 'cls' in self.task else 'C-index'
            exsiting_rows = df[(df[settings] == pd.Series(new_row)).all(axis=1)]
            if not exsiting_rows.empty:
                exsiting_mcc = exsiting_rows[reference].values
                if metric_dict[reference] > exsiting_mcc:
                    df = df.drop(exsiting_rows.index)
                else:
                    return

        new_row.update(metric_dict)
        df = df._append(new_row, ignore_index=True)
        df.to_excel(df_path, index=False)
        
    def fold_univariate_cox_regression_analysis(self, fold):
        training = self.model.training
        self.model.eval()

        event_indicator = torch.empty(0).cuda()
        event_time = torch.empty(0).cuda()
        risk_factor = torch.empty(0).cuda()
        slide_id = []

        df_name = f"{self.args.kfold}Fold_Cox.xlsx"
        res_path = self.args.results
        df_path = os.path.join(res_path, df_name)
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        
                
        with torch.no_grad():
            for data in self.test_loader:
                data = {k: v.cuda(non_blocking=True) if hasattr(v, 'cuda') else v for k, v in data.items()}
                outputs = self.model(data)
                risk = torch.sum(outputs['hazards'], dim=1)
                event_indicator = torch.cat((event_indicator, data['dead']), dim=0)
                event_time = torch.cat((event_time, data['event_time']), dim=0)
                risk_factor = torch.cat((risk_factor, risk), dim=0)
                slide_id.extend(data['id'])
        
        event_indicator = event_indicator.cpu().numpy()
        event_time = event_time.cpu().numpy()
        risk_factor = risk_factor.cpu().numpy()
                

        fold_df = pd.DataFrame({
            'Slide.ID': slide_id,
            'Fold': [fold] * len(slide_id),
            'event': event_indicator,
            'duration': event_time,
            f'{self.args.backbone}': risk_factor,
        })

        if hasattr(self, 'cox_df'):
            self.cox_df = pd.concat([self.cox_df, fold_df], ignore_index=True)
        else:
            self.cox_df = fold_df

        if fold == self.kfold - 1:
            if os.path.exists(df_path):
                existing_df = pd.read_excel(df_path)
                existing_df[f'{self.args.backbone}'] = None  # Initialize the new column

                for _, row in self.cox_df.iterrows():
                    slide_id = row['Slide.ID']
                    if slide_id in existing_df['Slide.ID'].values:
                        existing_df.loc[existing_df['Slide.ID'] == slide_id, f'{self.args.backbone}'] = row[f'{self.args.backbone}']
            else:
                existing_df = self.cox_df
            existing_df.to_excel(df_path, index=False)

        self.model.train(training)       
            