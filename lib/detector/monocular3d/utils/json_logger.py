import os
import json
from datetime import datetime

class LocalLogger:
    def __init__(self, path_to_log_folder, filename="train_log.json"):
        self.path_to_log_folder = path_to_log_folder
        self.path_to_file = os.path.join(path_to_log_folder, filename)

        os.makedirs(path_to_log_folder, exist_ok=True)

        # Load existing logs or initialize an empty list
        if os.path.isfile(self.path_to_file):
            with open(self.path_to_file, "r") as f:
                self.logs = json.load(f)
        else:
            self.logs = []

    def log(self, **kwargs):
        """
        Flexible logging method that accepts any keyword arguments
        Automatically adds timestamp to each log entry
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            **kwargs  # Accept any keyword arguments
        }

        self.logs.append(entry)

        # Write the full list to file
        with open(self.path_to_file, "w") as f:
            json.dump(self.logs, f, indent=4)

    def log_epoch(self, epoch, train_loss, train_cls_loss, train_box_loss, 
                  train_object_loss, train_rpn_loss, mAP, learning_rate):
        """
        Convenience method for epoch-wise logging
        """
        self.log(
            log_type="epoch",
            epoch=epoch,
            train_total_loss=train_loss,
            train_cls_loss=train_cls_loss,
            train_box_loss=train_box_loss,
            train_object_loss=train_object_loss,
            train_rpn_loss=train_rpn_loss,
            mAP=mAP,
            learning_rate=learning_rate
        )

    def log_iteration(self, iteration, total_loss, cls_loss, box_loss, 
                      object_loss, rpn_loss, learning_rate):
        """
        Convenience method for iteration-wise logging
        """
        self.log(
            log_type="iteration",
            iteration=iteration,
            iter_total_loss=total_loss,
            iter_cls_loss=cls_loss,
            iter_box_loss=box_loss,
            iter_object_loss=object_loss,
            iter_rpn_loss=rpn_loss,
            learning_rate=learning_rate
        )

    def log_mAP(self, iteration, mAP_score):
        """
        Convenience method for mAP logging
        """
        self.log(
            log_type="mAP_evaluation",
            iteration=iteration,
            mAP=mAP_score
        )

    def get_best_mAP(self):
        """
        Get the best mAP score from logs
        """
        mAP_scores = []
        for log in self.logs:
            if "mAP" in log:
                mAP_scores.append(log["mAP"])
        return max(mAP_scores) if mAP_scores else 0.0

    def get_latest_metrics(self):
        """
        Get the latest logged metrics
        """
        if not self.logs:
            return None
        return self.logs[-1]