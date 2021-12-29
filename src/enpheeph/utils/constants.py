# -*- coding: utf-8 -*-
import enpheeph.utils.enums
import enpheeph.utils.typings


NORSE_DIMENSION_DICT: enpheeph.utils.typings.DimensionDictType = {
    enpheeph.utils.enums.DimensionType.Time: 0,
    enpheeph.utils.enums.DimensionType.Batch: 1,
    enpheeph.utils.enums.DimensionType.Tensor: ...,
}
PYTORCH_DIMENSION_DICT: enpheeph.utils.typings.DimensionDictType = {
    enpheeph.utils.enums.DimensionType.Batch: 0,
    enpheeph.utils.enums.DimensionType.Tensor: ...,
}
