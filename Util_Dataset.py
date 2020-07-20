import numpy as np
# ['filename','leter',code]
list_dataset_items = [
	["no_0_", 0, "۰"],
    ["no_1_", 1, "۱"],
    ["no_2_", 2, "۲"],
    ["no_3_", 3, "۳"],
    ["no_4_", 4, "۴"],
    ["no_5_", 5, "۵"],
    ["no_6_", 6, "۶"],
    ["no_7_", 7, "۷"],
    ["no_8_", 8, "۸"],
    ["no_9_", 9, "۹"],
    ["al_al_", 10, "ا"],
    ["al_be_", 11, "ب"],
    ["al_pe_", 12, "پ"],
    ["al_te_", 13, "ت"],
    ["al_sn_", 14, "ث"],
    ["al_jm_", 15, "ج"],
    ["al_ch_", 16, "چ"],
    ["al_hh_", 17, "ح"],
    ["al_kh_", 18, "خ"],
    ["al_dl_", 19, "د"],
    ["al_zl_", 20, "ذ"],
    ["al_rr_", 21, "ر"],
    ["al_zz_", 22, "ز"],
    ["al_jz_", 23, "ژ"],
    ["al_sn_", 24, "س"],
    ["al_shn_", 25, "ش"],
    ["al_sd_", 26, "ص"],
    ["al_zd_", 27, "ض"],
    ["al_ta_", 28, "ط"],
    ["al_za_", 29, "ظ"],
    ["al_ay_", 30, "ع"],
    ["al_ghy_", 31, "غ"],
    ["al_fe_", 32, "ف"],
    ["al_gh_", 33, "ق"],
    ["al_kf_", 34, "ک"],
    ["al_gf_", 35, "گ"],
    ["al_lm_", 36, "ل"],
    ["al_mm_", 37, "م"],
    ["al_nn_", 38, "ن"],
    ["al_vv_", 39, "و"],
    ["al_he_", 40, "ه"],
    ["al_ye_", 40, "ی"]
]
list_dataset_items = np.array(list_dataset_items)


def get_char_ds(filename:str):
    for row in list_dataset_items:
        print(row[0], type(row[0]))
        print(filename, type(filename))
        if filename in row[0]:
            return row[1] , row[2]
    return "X"