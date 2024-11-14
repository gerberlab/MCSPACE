import pandas as pd
import argparse

def load_sintax(sintax_file):
    sintax = pd.read_table(sintax_file, header=None)
    sintax = sintax.loc[:, :1]
    sintax.columns = ['OTU', 'tax']
    sintax = sintax.set_index("OTU")
    return sintax

def build_tax_table(sintax, tax_process_info):
    # new tax df
    tax = sintax[[]].copy(deep=True)
    # iterate over taxonomy levels and add to tax df
    for key in tax_process_info.keys():
        level_df = sintax.tax.str.split(',').str[tax_process_info[key]["split_level"]]
        level_df = level_df.str.split(':').str[1]
        level_df = level_df.str.split('(', expand=True)
        level_df.columns = [key, tax_process_info[key]["conf"]]
        level_df[tax_process_info[key]["conf"]] = level_df[tax_process_info[key]["conf"]].str.strip(')')
        level_df[key] = level_df[key].str.replace("\"", "")
        tax = tax.join(level_df)
    return tax

tax_process_info = {
   "domain" : {
        "split_level" : -6,
        "conf" : "dconf"
    },
    "phylum" : {
        "split_level" : -5,
        "conf" : "pconf"
    },
   "class" : {
        "split_level" : -4,
        "conf" : "cconf"
    },
   "order" : {
        "split_level" : -3,
        "conf" : "oconf"
    },
   "family" : {
        "split_level" : -2,
        "conf" : "fconf"
    },
   "genus" : {
        "split_level" : -1,
        "conf" : "gconf"
    }
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', '--sintax',
                    help="SINTAX taxonomy output",
                    required=True)
    
    parser.add_argument('-t', '--taxtable',
                    help="CSV filepath to write out structured taxonomy table",
                    required=True)

    args = parser.parse_args()

    sintax = load_sintax(args.sintax)
    tax = build_tax_table(sintax, tax_process_info)
    tax.to_csv(args.taxtable)