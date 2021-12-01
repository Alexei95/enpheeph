local image_classifier = import "./image_classifier.libsonnet";
local object_detector = import "./object_detector.libsonnet";
local semantic_segmenter = import "./semantic_segmenter.libsonnet";

local models = {
    [image_classifier.model_task]: image_classifier,
    [object_detector.model_task]: object_detector,
    [semantic_segmenter.model_task]: semantic_segmenter,
};

function(model_task)
    if std.objectHas(models, model_task) then
        models[model_task]
    else
        error
            "invalid dataset name, choose between the following: " +
            std.join(
                ",",
                std.objectKeys(models),
            )
