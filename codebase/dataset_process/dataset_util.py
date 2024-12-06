import os
import pandas as pd
import pickle as pkl


def main():

    df_fmow = pkl.load(open("/media/zilun/mx500/ImageRAG_database/cropped_img/info_train_fmow.pkl", "rb"))
    df_fmow['img_name_list'] = df_fmow.apply(
        lambda row: "fmow_" + row['img_name_list'][:-4] + "_{}-{}-{}-{}".format(row['bbox_list'][0], row['bbox_list'][1], row['bbox_list'][2], row['bbox_list'][3]) + row['img_name_list'][-4:], axis=1
    )

    df_millionaid = pkl.load(open("/media/zilun/mx500/ImageRAG_database/cropped_img/info_train_millionaid.pkl", "rb"))
    df_millionaid_renamed = df_millionaid.rename(columns={
        "level3_class": "cls_list"
    })
    df_fmow["dataset"] = "fmow"
    df_millionaid_renamed["dataset"] = "millionaid"
    df_fmow_short = df_fmow[["img_name_list", "cls_list", "dataset"]]
    df_millionaid_short = df_millionaid_renamed[["img_name_list", "cls_list", "dataset"]]

    merged_df = pd.concat([df_fmow_short, df_millionaid_short])
    pkl.dump(merged_df, open("/media/zilun/mx500/ImageRAG_database/cropped_img/meta_df.pkl", "wb"))


if __name__ == "__main__":
    main()