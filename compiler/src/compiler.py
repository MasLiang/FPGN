import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))
from top_layer import generate_rtl
from solver import heuristic_h_search_best
from json_parser import json_parser

limit = {
    "LUT": 2357000,
    #"LUT": 22408,
    "FF":  5000000
}
json_path = "model_execution_info_6g.json"
# data = json_parser(json_path)
data = heuristic_h_search_best(json_path, limit)
os.system("rm -rf ./verilog")
os.makedirs("./verilog")
generate_rtl("./verilog", data)
