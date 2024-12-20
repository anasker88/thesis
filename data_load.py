import pandas as pd


def load_and_create_data():
    """
    Load template data and dictionary data to create a dataset
    """
    # load template data
    template_data = pd.read_csv("data/template.csv")
    # load dictionary data
    dictionary_data = pd.read_csv("data/dictionary.csv")
    vocabulary_num = len(dictionary_data)
    en_m, en_f, jp_m, jp_f = [], [], [], []
    for i, row in dictionary_data.iterrows():
        en_m_new = row["EN_M"]
        en_f_new = row["EN_F"]
        jp_m_new = row["JP_M"]
        jp_f_new = row["JP_F"]
        if row["EN_prefix"] == "_":
            en_m_new = en_m_new.capitalize()
            en_f_new = en_f_new.capitalize()
        else:
            en_m_new = row["EN_prefix"] + " " + en_m_new
            en_f_new = row["EN_prefix"] + " " + en_f_new
        if row["JP_prefix"] != "_":
            jp_m_new = row["JP_prefix"] + jp_m_new
            jp_f_new = row["JP_prefix"] + jp_f_new
        en_m.append(en_m_new)
        en_f.append(en_f_new)
        jp_m.append(jp_m_new)
        jp_f.append(jp_f_new)
    # create dataset
    # en,jp
    # stereotypical,anti-stereotypical
    en_data = {"st": [], "anti-st": [], "tgt": []}
    jp_data = {"st": [], "anti-st": [], "tgt": []}
    print(template_data["Known_stereotyped_groups"])
    for i, row in template_data.iterrows():
        if row["Known_stereotyped_groups"] == '["F"]':
            for j in range(vocabulary_num):
                en_data["st"].append(row["EN_tpl_neg_st"].replace("_", en_f[j]))
                en_data["anti-st"].append(row["EN_tpl_neg_st"].replace("_", en_m[j]))
                en_data["tgt"].append(row["EN_tgt_neg_st"])
                en_data["st"].append(row["EN_tpl_non_neg_st"].replace("_", en_m[j]))
                en_data["anti-st"].append(
                    row["EN_tpl_non_neg_st"].replace("_", en_f[j])
                )
                en_data["tgt"].append(row["EN_tgt_non_neg_st"])

                jp_data["st"].append(row["JP_tpl_neg_st"].replace("_", jp_f[j]))
                jp_data["anti-st"].append(row["JP_tpl_neg_st"].replace("_", jp_m[j]))
                jp_data["tgt"].append(row["JP_tgt_neg_st"])
                jp_data["st"].append(row["JP_tpl_non_neg_st"].replace("_", jp_m[j]))
                jp_data["anti-st"].append(
                    row["JP_tpl_non_neg_st"].replace("_", jp_f[j])
                )
                jp_data["tgt"].append(row["JP_tgt_non_neg_st"])
        elif row["Known_stereotyped_groups"][0] == '["M"]':
            for j in range(vocabulary_num):
                en_data["st"].append(row["EN_tpl_neg_st"].replace("_", en_m[j]))
                en_data["anti-st"].append(row["EN_tpl_neg_st"].replace("_", en_f[j]))
                en_data["tgt"].append(row["EN_tgt_neg_st"])
                en_data["st"].append(row["EN_tpl_non_neg_st"].replace("_", en_f[j]))
                en_data["anti-st"].append(
                    row["EN_tpl_non_neg_st"].replace("_", en_m[j])
                )
                en_data["tgt"].append(row["EN_tgt_non_neg_st"])

                jp_data["st"].append(row["JP_tpl_neg_st"].replace("_", jp_m[j]))
                jp_data["anti-st"].append(row["JP_tpl_neg_st"].replace("_", jp_f[j]))
                jp_data["tgt"].append(row["JP_tgt_neg_st"])
                jp_data["st"].append(row["JP_tpl_non_neg_st"].replace("_", jp_f[j]))
                jp_data["anti-st"].append(
                    row["JP_tpl_non_neg_st"].replace("_", jp_m[j])
                )
                jp_data["tgt"].append(row["JP_tgt_non_neg_st"])
    en_data = pd.DataFrame(en_data)
    jp_data = pd.DataFrame(jp_data)
    en_data.to_csv("data/en_data.csv")
    jp_data.to_csv("data/jp_data.csv")


def __main__():
    load_and_create_data()


__main__()
