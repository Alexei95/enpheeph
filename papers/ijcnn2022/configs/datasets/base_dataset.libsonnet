local defaults = import "../defaults.libsonnet";
local utils = import "../utils.libsonnet";

{
    local config = self,

    dataset_batch_size:: defaults.dataset_batch_size,
    dataset_name:: null,
    dataset_num_classes:: null,
    dataset_num_workers:: defaults.dataset_num_workers,
    dataset_root:: defaults.dataset_root,
    dataset_subcommand_name:: null,
    dataset_val_split:: defaults.dataset_val_split,

    dataset_complete_dir:: utils.joinPath(self.dataset_root, self.dataset_name),
}
