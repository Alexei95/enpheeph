local carla = import "./carla.libsonnet";
local cifar10 = import "./cifar10.libsonnet";
local coco = import "./coco.libsonnet";

local datasets = {
    [carla.dataset_name]: carla,
    [cifar10.dataset_name]: cifar10,
    [coco.dataset_name]: coco,
};

function(dataset_name)
    if std.objectHas(datasets, dataset_name) then
        datasets[dataset_name]
    else
        error
            "invalid dataset name, choose between the following: " +
            std.join(
                ",",
                std.objectKeys(datasets),
            )
