# coding: utf-8
# pyfair.plain.


from .utils_saver import elegant_print, get_elogger
from .utils_timer import (
    fantasy_timer, fantasy_durat, elegant_durat, elegant_dated)

from .ensem_voting import (
    plurality_voting, majority_voting, weighted_voting)
from .metric_cont import (
    contingency_tab_bi, contg_tab_mu_type3, contg_tab_mu_merge,
    contg_tab_mu_type2, contg_tab_mu_type1)
from .data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_cross_valid,
    manual_repetitive, scale_normalize_helper, scale_normalize_data)


__all__ = [
    'weighted_voting',
    'plurality_voting',
    'majority_voting',

    'elegant_print',
    'get_elogger',

    'fantasy_timer',
    'fantasy_durat',
    'elegant_durat',
    'elegant_dated',

    'manual_cross_valid',
    'manual_repetitive',
    'scale_normalize_helper',
    'scale_normalize_data',
    'sklearn_k_fold_cv',
    'sklearn_stratify',

    'contingency_tab_bi',
    'contg_tab_mu_type3',
    'contg_tab_mu_merge',
    'contg_tab_mu_type2',
    'contg_tab_mu_type1',
]
