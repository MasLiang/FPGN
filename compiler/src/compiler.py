import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'template'))
from top_layer import generate_rtl
from solver import heuristic_h_search_best
from json_parser import json_parser

limit = {
    "LUT": 2000000,
    "FF":  4000000
}
json_path = "model_execution_info_6g.json"
data = heuristic_h_search_best(json_path, limit)
os.system("rm -rf ./verilog")
os.makedirs("./verilog")
generate_rtl("./verilog", data)
